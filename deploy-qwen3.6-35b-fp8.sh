#!/bin/bash
# Deploy Qwen3.6-35B-A3B-FP8 single-node (TP=1) on spark-2, gpu-mem 0.50.
# API: http://192.168.110.2:8000/v1
# Usage: ./deploy-qwen3.6-35b-fp8.sh [stop|--no-build]
set -e

if [ "${1:-}" = "stop" ]; then
    ssh spark-2 'docker rm -f vllm_node 2>/dev/null' || true
    ssh spark-3 'docker rm -f vllm_node 2>/dev/null' || true
    echo "Stopped."; exit 0
fi

# Update repo + rebuild image (skip both with --no-build, e.g. watchdog)
if [ "${1:-}" != "--no-build" ]; then
    ssh spark-2 'cd ~/spark-vllm-docker && git checkout main && git pull --ff-only'
    ssh spark-2 'cd ~/spark-vllm-docker && ./build-and-copy.sh'
fi

# Clean restart
ssh spark-2 'docker rm -f vllm_node 2>/dev/null' || true
ssh spark-3 'docker rm -f vllm_node 2>/dev/null' || true

# Solo deploy on spark-2, TP=1, gpu-mem 0.50 (lowered from 0.70: 0.70x128=~90GiB left
# spark-2 at ~0 free / OOM edge; 0.50=~64GiB leaves ~25GiB headroom, ~29GiB for KV).
ssh spark-2 "cd ~/spark-vllm-docker && python3 run-recipe.py qwen3.6-35b-a3b-fp8 \
    --solo --tp 1 \
    -e HF_HUB_OFFLINE=1 \
    -e LD_PRELOAD=/usr/local/lib/python3.12/dist-packages/ray/core/libjemalloc.so \
    -e MALLOC_CONF=background_thread:true,dirty_decay_ms:1000,muzzy_decay_ms:1000 \
    -e UCX_MEM_MMAP_HOOK_MODE=none \
    -e VLLM_SLEEP_WHEN_IDLE=1 \
    --gpu-mem 0.50 -d -- --generation-config auto --override-generation-config '{\"presence_penalty\":1.0}' --max-num-batched-tokens 65536"

echo "API: http://192.168.110.2:8000/v1"
echo "Logs: ssh spark-2 'docker logs -f vllm_node'"
