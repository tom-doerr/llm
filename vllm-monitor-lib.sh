check_health() {
    local s=$(ssh spark-2 \
      "docker ps --filter name=vllm-head \
      --format '{{.Status}}'" 2>/dev/null)
    [[ -z "$s" ]] && echo "FAIL: down" && return 1
    local m=$(curl -s -m10 \
      "$API/v1/models" 2>/dev/null)
    [[ -z "$m" ]] && echo "FAIL: API" && return 1
    local r=$(curl -s -m300 "$API/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d '{"model":"Qwen/Qwen3.5-122B-A10B-FP8",
      "messages":[{"role":"user","content":"hi"}],
      "max_tokens":1}' 2>/dev/null)
    [[ -z "$r" ]] && echo "FAIL: hang" && return 1
    echo "OK"; return 0
}
switch_to_nightly() {
    echo "$(date): Switch to $NEW"
    sed -i \
      "s|CONTAINER=.*|CONTAINER=\"\${CONTAINER:-$NEW}\"|" \
      "$SCRIPT"
    bash "$SCRIPT"; sleep 900
}
