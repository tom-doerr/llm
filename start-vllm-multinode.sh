#!/bin/bash
# Start vLLM multi-node on spark-2 (head) + spark-3 (worker)
# Usage: ./start-vllm-multinode.sh [--debug] [--pp]
set -e

DEBUG_NCCL=${DEBUG_NCCL:-0}
USE_PP=${USE_PP:-0}
for arg in "$@"; do
  [[ "$arg" == "--debug" ]] && DEBUG_NCCL=1
  [[ "$arg" == "--pp" ]] && USE_PP=1
done

MODEL="QuantTrio/Qwen3-VL-235B-A22B-Instruct-AWQ"
CONTAINER="nvcr.io/nvidia/vllm:25.11-py3"
OOB_IF="enp1s0f1np1"  # Control plane interface
HEAD_IP="192.168.100.10"
WORKER_IP="192.168.100.11"
# NCCL IB with GDR disabled (host-staged RDMA for lower latency)
ENV="-e NCCL_NET_PLUGIN=none -e NCCL_DMABUF_ENABLE=0"
ENV="$ENV -e NCCL_NET_GDR_LEVEL=LOC -e NCCL_NET_GDR_C2C=0"
ENV="$ENV -e NCCL_IB_HCA='=rocep1s0f1:1'"  # Single-rail (faster than dual with GDR off)
ENV="$ENV -e NCCL_SOCKET_IFNAME=$OOB_IF"
ENV="$ENV -e GLOO_SOCKET_IFNAME=$OOB_IF"
ENV="$ENV -e UCX_NET_DEVICES=$OOB_IF -e RAY_memory_monitor_refresh_ms=0"
ENV="$ENV -e HF_HUB_OFFLINE=1"
ENV="$ENV -e VLLM_SLEEP_WHEN_IDLE=1"  # Reduce CPU when idle (small latency cost)
ENV="$ENV -e OMP_NUM_THREADS=1"  # Reduce threading overhead (vLLM Qwen3-VL recommended)
# Note: VLLM_USE_RAY_COMPILED_DAG=0 doesn't work for multi-node - vLLM forces it to 1

if [ "$DEBUG_NCCL" = "1" ]; then
  ENV="$ENV -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,NET"
  ENV="$ENV -e NCCL_DEBUG_FILE=/tmp/nccl.%h.%p.log"
  echo "NCCL debug enabled (logs: /tmp/nccl.*.log)"
fi

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

RDMA="--device=/dev/infiniband --ulimit memlock=-1 --cap-add=IPC_LOCK"

echo "=== Starting Ray head on spark-2 ==="
ssh spark-2 "docker run -d --name vllm-head --gpus all --shm-size 16g \\
    --network host --ipc host $RDMA $ENV_HEAD $VOLS $CONTAINER \\
    ray start --head --port=6379 --node-ip-address=$HEAD_IP --block"

echo "Waiting for Ray head..."
sleep 10

echo "=== Starting Ray worker on spark-3 ==="
ssh spark-3 "docker run -d --name vllm-worker --gpus all --shm-size 16g \\
    --network host --ipc host $RDMA $ENV_WORKER $VOLS $CONTAINER \\
    ray start --address=$HEAD_IP:6379 --node-ip-address=$WORKER_IP --block"

echo "Waiting for worker to join..."
sleep 5

echo "=== Starting vLLM server ==="
if [ "$USE_PP" = "1" ]; then
  echo "Mode: Pipeline Parallel (PP=2)"
  VLLM_ARGS="--pipeline-parallel-size 2 --trust-remote-code --enforce-eager"
else
  echo "Mode: Tensor Parallel (TP=2)"
  VLLM_ARGS="--tensor-parallel-size 2 --trust-remote-code --enforce-eager"
fi
VLLM_ARGS="$VLLM_ARGS --quantization awq --gpu-memory-utilization 0.75"
VLLM_ARGS="$VLLM_ARGS --kv-cache-dtype fp8 --limit-mm-per-prompt.video 0"
VLLM_ARGS="$VLLM_ARGS --mm-encoder-tp-mode data"
VLLM_ARGS="$VLLM_ARGS --max-num-batched-tokens 8192"
VLLM_ARGS="$VLLM_ARGS --host 0.0.0.0 --port 8000"

ssh spark-2 "docker exec -d -e RAY_ADDRESS=$HEAD_IP:6379 -e VLLM_ATTENTION_BACKEND=TRITON_ATTN vllm-head \\
    vllm serve $MODEL $VLLM_ARGS"

echo "=== Done ==="
echo "Logs: ssh spark-2 'docker exec -it vllm-head bash' then check"
echo "API:  http://192.168.102.11:8000/v1/chat/completions"
