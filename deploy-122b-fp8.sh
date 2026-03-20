#!/bin/bash
# Deploy Qwen3.5-122B-FP8 TP=2: spark-2 (head) + spark-3 (worker)
# API: http://192.168.110.2:8000/v1
# Usage: ./deploy-122b-fp8.sh [stop|--no-build]
set -e
NODES="192.168.100.10,192.168.100.11"

if [ "${1:-}" = "stop" ]; then
    ssh spark-2 'docker rm -f vllm_node 2>/dev/null' || true
    ssh spark-3 'docker rm -f vllm_node 2>/dev/null' || true
    echo "Stopped."; exit 0
fi

# Update repo + rebuild image (skip both with --no-build, e.g. watchdog)
if [ "${1:-}" != "--no-build" ]; then
    ssh spark-2 'cd ~/spark-vllm-docker && git checkout main && git pull --ff-only'
    ssh spark-2 'cd ~/spark-vllm-docker && ./build-and-copy.sh --copy-to spark-3'
fi

# Clean restart
ssh spark-2 'docker rm -f vllm_node 2>/dev/null' || true
ssh spark-3 'docker rm -f vllm_node 2>/dev/null' || true

# Deploy via run-recipe.py (generates launch script from YAML + calls launch-cluster.sh)
ssh spark-2 "cd ~/spark-vllm-docker && python3 run-recipe.py qwen3.5-122b-fp8 \
    -n $NODES --ib-if rocep1s0f1,roceP2p1s0f1 --eth-if enp1s0f1np1 \
    -e NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 -e HF_HUB_OFFLINE=1 \
    --no-ray -d"

echo "API: http://192.168.110.2:8000/v1"
echo "Logs: ssh spark-2 'docker logs -f vllm_node'"
