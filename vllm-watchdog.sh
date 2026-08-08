#!/bin/bash
# vLLM watchdog: poll health, save crash logs, redeploy
# Usage: ./vllm-watchdog.sh [interval_secs] [max_fails]
INTERVAL="${1:-300}"
MAX_FAILS="${2:-2}"
DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$DIR/crash-logs"
URL="http://192.168.110.2:8000/v1/models"
METRICS="http://192.168.110.2:8000/metrics"
CHAT="http://192.168.110.2:8000/v1/chat/completions"
f=0
last_gen=""
mkdir -p "$LOG_DIR"
echo "$(date): watchdog start (${INTERVAL}s, ${MAX_FAILS} fails)"

save_crash_logs() {
  TS=$(date +%Y%m%d_%H%M%S)
  D="$LOG_DIR/$TS"
  mkdir -p "$D"
  ssh spark-2 "docker inspect vllm_node --format '{{json .State}}'" > "$D/spark2-state.json" 2>&1
  ssh spark-3 "docker inspect vllm_node --format '{{json .State}}'" > "$D/spark3-state.json" 2>&1
  ssh spark-2 "docker logs vllm_node" > "$D/spark2.log" 2>&1
  ssh spark-3 "docker logs vllm_node" > "$D/spark3.log" 2>&1
  ssh spark-2 "journalctl -k --since '1 hour ago'" > "$D/spark2-dmesg.log" 2>&1
  ssh spark-3 "journalctl -k --since '1 hour ago'" > "$D/spark3-dmesg.log" 2>&1
  echo "$(date): crash logs saved to $D"
}

check_ok() {
  # layer 1: API up at all?
  curl -s -m30 "$URL" | python3 -c \
    "import sys,json; json.load(sys.stdin)" 2>/dev/null || return 1
  # layer 2: engine wedge (Aug 8: engine core died at 8.5->0 tok/s
  # while /v1/models kept answering 200 for 40+ min). Metrics-gated:
  # requests running but the generation counter frozen since the
  # last poll -> disambiguate with ONE live completion (sleep-idle
  # engines wake and answer; a wedged engine times out). Never
  # probes with inference during normal operation (DAG caution).
  m=$(curl -s -m15 "$METRICS") || { last_gen=""; return 0; }
  run=$(echo "$m" | awk '/^vllm:num_requests_running/{s+=$2} END{printf "%d", s+0}')
  gen=$(echo "$m" | awk '/^vllm:generation_tokens_total/{s+=$2} END{printf "%d", s+0}')
  if [ "${run:-0}" -gt 0 ] && [ -n "$last_gen" ] && \
     [ "$gen" = "$last_gen" ]; then
    echo "$(date): $run running, generation frozen -- live probe"
    model=$(curl -s -m10 "$URL" | python3 -c \
      "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])" \
      2>/dev/null)
    out=$(curl -s -m90 "$CHAT" -H 'content-type: application/json' \
      -d '{"model":"'"$model"'","messages":[{"role":"user","content":"say ok"}],"max_tokens":3}')
    echo "$out" | grep -q '"content"' || { last_gen=$gen; return 1; }
  fi
  last_gen=$gen
  return 0
}

while true; do
  if check_ok; then
    [ $f -gt 0 ] && echo "$(date): recovered"
    f=0
  else
    f=$((f+1))
    echo "$(date): fail ($f/$MAX_FAILS)"
  fi
  if [ $f -ge $MAX_FAILS ]; then
    echo "$(date): saving crash logs and redeploying"
    save_crash_logs
    bash "$DIR/deploy.sh" --no-build
    f=0
    sleep 600
  fi
  sleep "$INTERVAL"
done
