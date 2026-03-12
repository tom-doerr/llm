#!/bin/bash
# vLLM watchdog: poll health, save crash logs, redeploy
# Usage: ./vllm-watchdog.sh [interval_secs] [max_fails]
INTERVAL="${1:-300}"
MAX_FAILS="${2:-2}"
DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$DIR/crash-logs"
URL="http://192.168.110.2:8000/v1/models"
f=0
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

while true; do
  if curl -s -m30 "$URL" | python3 -c "import sys,json; json.load(sys.stdin)" 2>/dev/null; then
    [ $f -gt 0 ] && echo "$(date): recovered"
    f=0
  else
    f=$((f+1))
    echo "$(date): fail ($f/$MAX_FAILS)"
  fi
  if [ $f -ge $MAX_FAILS ]; then
    echo "$(date): saving crash logs and redeploying"
    save_crash_logs
    bash "$DIR/deploy-122b-fp8.sh"
    f=0
    sleep 600
  fi
  sleep "$INTERVAL"
done
