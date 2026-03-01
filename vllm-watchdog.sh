#!/bin/bash
# vLLM watchdog: send test request every 5 min, restart after 2 consecutive failures
L=/tmp/vllm-watchdog.log
API=${VLLM_API_URL:-http://192.168.110.2:8000/v1/chat/completions}
D='{"model":"Qwen/Qwen3.5-122B-A10B-FP8","messages":[{"role":"user","content":"1+1="}],"max_tokens":5}'
f=0
echo "$(date): watchdog start"|tee -a $L
while true; do
  r=$(curl -s -m120 $API -H"Content-Type: application/json" -d"$D")
  if echo "$r"|python3 -c"import sys,json;json.load(sys.stdin)" 2>/dev/null; then
    [ $f -gt 0 ] && echo "$(date): recovered"|tee -a $L
    f=0
  else
    f=$((f+1))
    echo "$(date): fail($f/2) ${r:0:80}"|tee -a $L
  fi
  if [ $f -ge 2 ]; then
    echo "$(date): RESTART"|tee -a $L
    cd /home/tom/llm && bash start-vllm-multinode.sh >>$L 2>&1
    f=0; sleep 900
  fi
  sleep 300
done
