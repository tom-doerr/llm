#!/bin/bash
# Deploy Qwen3.5-0.8B on spark-1, port 8000
# API: http://localhost:8000/v1
# Usage: ./deploy-0.8b.sh [stop]
set -e

CONTAINER="vllm-0.8b"

if [ "${1:-}" = "stop" ]; then
    docker stop "$CONTAINER" 2>/dev/null && docker rm "$CONTAINER" 2>/dev/null || true
    echo "Stopped."; exit 0
fi

docker stop "$CONTAINER" 2>/dev/null && docker rm "$CONTAINER" 2>/dev/null || true

docker run -d --name "$CONTAINER" \
  --gpus all \
  --shm-size 4g \
  -p 8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  vllm/vllm-openai:cu130-nightly \
  --model Qwen/Qwen3.5-0.8B \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.1 \
  --enforce-eager

echo "API: http://localhost:8000/v1"
echo "Logs: docker logs -f $CONTAINER"
