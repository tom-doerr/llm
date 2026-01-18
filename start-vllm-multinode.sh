#!/bin/bash
# Start vLLM multi-node on spark-2 (head) + spark-3 (worker)
set -e

MODEL="QuantTrio/Qwen3-VL-235B-A22B-Instruct-AWQ"
CONTAINER="nvcr.io/nvidia/vllm:25.11-py3"
RDMA_IF="enp1s0f1np1"
HEAD_IP="192.168.100.10"

ENV="-e NCCL_SOCKET_IFNAME=$RDMA_IF -e GLOO_SOCKET_IFNAME=$RDMA_IF"
ENV="$ENV -e UCX_NET_DEVICES=$RDMA_IF -e RAY_memory_monitor_refresh_ms=0"
ENV="$ENV -e HF_HUB_OFFLINE=1"

VOLS="-v /home/tom/.cache/huggingface:/root/.cache/huggingface"
VOLS="$VOLS -v /home/tom/sitecustomize.py:/usr/lib/python3.12/sitecustomize.py:ro"

echo "=== Cleanup ==="
ssh spark-2 "docker rm -f vllm-head 2>/dev/null || true"
ssh spark-3 "docker rm -f vllm-worker 2>/dev/null || true"

# echo "=== Drop caches ==="
# ssh spark-2 "sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'"
# ssh spark-3 "sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'"

echo "=== Starting head on spark-2 ==="
ssh spark-2 "docker run -d --name vllm-head --gpus all --shm-size 16g \\
    --network host --ipc host $ENV $VOLS $CONTAINER \\
    vllm serve $MODEL --tensor-parallel-size 2 --trust-remote-code \\
    --enforce-eager --quantization awq --gpu-memory-utilization 0.75 \\
    --kv-cache-dtype fp8 --limit-mm-per-prompt.video 0 \\
    --host 0.0.0.0 --port 8000"

echo "Waiting for Ray head..."
sleep 10

echo "=== Starting worker on spark-3 ==="
ssh spark-3 "docker run -d --name vllm-worker --gpus all --shm-size 16g \\
    --network host --ipc host $ENV -e RAY_ADDRESS=$HEAD_IP:6379 $VOLS \\
    $CONTAINER ray start --address=$HEAD_IP:6379 --block"

echo "=== Done ==="
echo "Logs: ssh spark-2 'docker logs -f vllm-head'"
echo "API:  http://192.168.102.11:8000/v1/chat/completions"
