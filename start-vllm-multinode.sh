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

MODEL="QuantTrio/Qwen3-VL-235B-A22B-Thinking-AWQ"
CONTAINER="nvcr.io/nvidia/vllm:25.11-py3"
OOB_IF="enp1s0f1np1"  # Control plane interface
HEAD_IP="192.168.100.10"
WORKER_IP="192.168.100.11"
# NCCL IB with GDR disabled (host-staged RDMA for lower latency)
ENV="-e NCCL_NET_PLUGIN=none -e NCCL_DMABUF_ENABLE=0"
ENV="$ENV -e NCCL_NET_GDR_LEVEL=LOC -e NCCL_NET_GDR_C2C=0"
ENV="$ENV -e NCCL_IB_HCA='=rocep1s0f1:1,roceP2p1s0f1:1'"  # Dual-rail
ENV="$ENV -e NCCL_SOCKET_IFNAME=$OOB_IF"
ENV="$ENV -e GLOO_SOCKET_IFNAME=$OOB_IF"
ENV="$ENV -e UCX_NET_DEVICES=$OOB_IF -e RAY_memory_monitor_refresh_ms=0"
ENV="$ENV -e HF_HUB_OFFLINE=1"
ENV="$ENV -e VLLM_SLEEP_WHEN_IDLE=1"  # Reduce CPU when idle (small latency cost)
ENV="$ENV -e OMP_NUM_THREADS=1"  # Reduce threading overhead
ENV="$ENV -e VLLM_USE_RAY_COMPILED_DAG=0"  # Disable compiled DAG
# Note: VLLM_USE_RAY_COMPILED_DAG=0 may not work for multi-node - vLLM V1 forces it to 1

if [ "$DEBUG_NCCL" = "1" ]; then
  ENV="$ENV -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,NET"
  ENV="$ENV -e NCCL_DEBUG_FILE=/tmp/nccl.%h.%p.log"
  echo "NCCL debug enabled (logs: /tmp/nccl.*.log)"
fi

ENV_HEAD="$ENV -e VLLM_HOST_IP=$HEAD_IP"
ENV_WORKER="$ENV -e VLLM_HOST_IP=$WORKER_IP"

VOLS="-v /home/tom/.cache/huggingface:/root/.cache/huggingface"
VOLS="$VOLS -v /home/tom/llm/sitecustomize.py:/usr/lib/python3.12/sitecustomize.py:ro"
VOLS="$VOLS -v /tmp/vllm-head-entrypoint.sh:/entrypoint.sh:ro"
VOLS="$VOLS -v /tmp/vllm-serve-cmd.sh:/vllm-serve-cmd.sh:ro"
VOLS_WORKER="-v /home/tom/.cache/huggingface:/root/.cache/huggingface"
VOLS_WORKER="$VOLS_WORKER -v /home/tom/llm/sitecustomize.py:/usr/lib/python3.12/sitecustomize.py:ro"
VOLS_WORKER="$VOLS_WORKER -v /tmp/vllm-worker-entrypoint.sh:/entrypoint.sh:ro"

echo "=== Deploying entrypoint scripts ==="
scp -q /home/tom/llm/vllm-head-entrypoint.sh spark-2:/tmp/vllm-head-entrypoint.sh
scp -q /home/tom/llm/vllm-worker-entrypoint.sh spark-3:/tmp/vllm-worker-entrypoint.sh

echo "=== Cleanup ==="
ssh spark-2 "docker rm -f vllm-head 2>/dev/null || true"
ssh spark-3 "docker rm -f vllm-worker 2>/dev/null || true"

# echo "=== Drop caches ==="
# ssh spark-2 "sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'"
# ssh spark-3 "sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'"

RDMA="--device=/dev/infiniband --ulimit memlock=-1 --cap-add=IPC_LOCK"

echo "=== Building vLLM args ==="
if [ "$USE_PP" = "1" ]; then
  echo "Mode: Pipeline Parallel (PP=2)"
  VLLM_ARGS="--pipeline-parallel-size 2 --trust-remote-code"
else
  echo "Mode: Tensor Parallel (TP=2)"
  VLLM_ARGS="--tensor-parallel-size 2 --trust-remote-code"
fi
VLLM_ARGS="$VLLM_ARGS --quantization awq --gpu-memory-utilization 0.70"
VLLM_ARGS="$VLLM_ARGS --kv-cache-dtype fp8"
VLLM_ARGS="$VLLM_ARGS --max-num-batched-tokens 4096"
VLLM_ARGS="$VLLM_ARGS --scheduling-policy priority"  # Lower priority value = higher priority
VLLM_ARGS="$VLLM_ARGS --distributed-executor-backend ray"  # Required for multi-node
VLLM_ARGS="$VLLM_ARGS --mm-encoder-tp-mode data"  # GPU data parallel for vision encoder
VLLM_ARGS="$VLLM_ARGS --limit-mm-per-prompt '{\"video\": 0}'"  # Disable video input
VLLM_ARGS="$VLLM_ARGS --enforce-eager"  # Disable CUDA graphs (test CPU usage)
VLLM_ARGS="$VLLM_ARGS --enable-auto-tool-choice --tool-call-parser hermes"
VLLM_ARGS="$VLLM_ARGS --host 0.0.0.0 --port 8000"

echo "#!/bin/bash" > /home/tom/llm/vllm-serve-cmd.sh
echo "exec vllm serve $MODEL $VLLM_ARGS" >> /home/tom/llm/vllm-serve-cmd.sh
scp -q /home/tom/llm/vllm-serve-cmd.sh spark-2:/tmp/vllm-serve-cmd.sh
ENV_WORKER="$ENV_WORKER -e RAY_HEAD_IP=$HEAD_IP"

echo "=== Starting worker on spark-3 ==="
ssh spark-3 "docker run -d --name vllm-worker --gpus all --shm-size 16g \\
    --network host --ipc host --restart=on-failure:10 \\
    $RDMA $ENV_WORKER $VOLS_WORKER $CONTAINER bash /entrypoint.sh"

echo "=== Starting head on spark-2 ==="
ssh spark-2 "docker run -d --name vllm-head --gpus all --shm-size 16g \\
    --network host --ipc host --restart=on-failure:10 \\
    $RDMA $ENV_HEAD $VOLS $CONTAINER bash /entrypoint.sh"

echo "=== Containers started (auto-restart on failure) ==="
echo "Head logs:   ssh spark-2 'docker logs -f vllm-head'"
echo "Worker logs: ssh spark-3 'docker logs -f vllm-worker'"
echo "API:         http://192.168.102.11:8000/v1/chat/completions"
