#!/bin/bash
# Deploy Qwen3.5-122B-FP8 TP=2: spark-2 (head) + spark-3 (worker)
# API: http://192.168.110.2:8000/v1
# Usage: ./deploy-122b-fp8.sh [stop|--no-build]
set -e
CD="$(cd "$(dirname "$0")/spark-vllm-docker" && pwd)"

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

# Sync launch script to spark-2
scp "$(dirname "$0")/launch-122b-fp8.sh" spark-2:~/spark-vllm-docker/launch-122b-fp8.sh

# Clean restart
ssh spark-2 'docker rm -f vllm_node 2>/dev/null' || true
ssh spark-3 'docker rm -f vllm_node 2>/dev/null' || true

ssh spark-2 "cd ~/spark-vllm-docker && ./launch-cluster.sh \
    -n 192.168.100.10,192.168.100.11 \
    --eth-if enp1s0f1np1 \
    --ib-if rocep1s0f1,roceP2p1s0f1 \
    --apply-mod mods/fix-qwen3.5-chat-template \
    --launch-script launch-122b-fp8.sh -d"

echo "API: http://192.168.110.2:8000/v1"
echo "Logs: ssh spark-2 'docker logs -f vllm_node'"
