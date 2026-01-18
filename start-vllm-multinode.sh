#!/bin/bash
# Start vLLM multi-node on spark-2 (head) + spark-3 (worker)
set -e

MODEL="QuantTrio/Qwen3-VL-235B-A22B-Instruct-AWQ"
CONTAINER="nvcr.io/nvidia/vllm:25.11-py3"
RDMA_IF="enp1s0f1np1"
HEAD_IP="192.168.100.10"
WORKER_IP="192.168.100.11"

ENV="-e NCCL_SOCKET_IFNAME=$RDMA_IF -e GLOO_SOCKET_IFNAME=$RDMA_IF"
ENV="$ENV -e UCX_NET_DEVICES=$RDMA_IF -e RAY_memory_monitor_refresh_ms=0"
ENV="$ENV -e HF_HUB_OFFLINE=1"

ENV_HEAD="$ENV -e VLLM_HOST_IP=$HEAD_IP"
ENV_WORKER="$ENV -e VLLM_HOST_IP=$WORKER_IP"

VOLS="-v /home/tom/.cache/huggingface:/root/.cache/huggingface"
VOLS="$VOLS -v /home/tom/llm/sitecustomize.py:/usr/lib/python3.12/sitecustomize.py:ro"

echo "=== Cleanup ==="
ssh spark-2 "docker rm -f vllm-head 2>/dev/null || true"
ssh spark-3 "docker rm -f vllm-worker 2>/dev/null || true"

# echo "=== Drop caches ==="
# ssh spark-2 "sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'"
# ssh spark-3 "sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'"

echo "=== Starting Ray head on spark-2 ==="
ssh spark-2 "docker run -d --name vllm-head --gpus all --shm-size 16g \\
    --network host --ipc host $ENV_HEAD $VOLS $CONTAINER \\
    ray start --head --port=6379 --node-ip-address=$HEAD_IP --block"

echo "Waiting for Ray head..."
sleep 10

echo "=== Starting Ray worker on spark-3 ==="
ssh spark-3 "docker run -d --name vllm-worker --gpus all --shm-size 16g \\
    --network host --ipc host $ENV_WORKER $VOLS $CONTAINER \\
    ray start --address=$HEAD_IP:6379 --node-ip-address=$WORKER_IP --block"

echo "Waiting for worker to join..."
sleep 5

echo "=== Starting vLLM server ==="
VLLM_ARGS="--tensor-parallel-size 2 --trust-remote-code --enforce-eager"
VLLM_ARGS="$VLLM_ARGS --quantization awq --gpu-memory-utilization 0.75"
VLLM_ARGS="$VLLM_ARGS --kv-cache-dtype fp8 --limit-mm-per-prompt.video 0"
VLLM_ARGS="$VLLM_ARGS --host 0.0.0.0 --port 8000"

ssh spark-2 "docker exec -d -e RAY_ADDRESS=$HEAD_IP:6379 vllm-head \\
    vllm serve $MODEL $VLLM_ARGS"

echo "=== Done ==="
echo "Logs: ssh spark-2 'docker exec -it vllm-head bash' then check"
echo "API:  http://192.168.102.11:8000/v1/chat/completions"
