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

MODEL="${MODEL:-Qwen/Qwen3.5-122B-A10B-FP8}"
CONTAINER="${CONTAINER:-vllm/vllm-openai:qwen3_5-cu130}"
OOB_IF="enp1s0f1np1"  # Control plane interface
HEAD_IP="192.168.100.10"
WORKER_IP="192.168.100.11"
# NCCL IB with GDR disabled (host-staged RDMA for lower latency)
ENV="-e NCCL_NET_PLUGIN=none -e NCCL_DMABUF_ENABLE=0"
ENV="$ENV -e NCCL_NET_GDR_LEVEL=LOC -e NCCL_NET_GDR_C2C=0"
ENV="$ENV -e NCCL_IB_HCA='=rocep1s0f1:1'"  # Single-rail (reduces pinned buffers on UMA)
ENV="$ENV -e NCCL_SOCKET_IFNAME=$OOB_IF"
ENV="$ENV -e GLOO_SOCKET_IFNAME=$OOB_IF"
ENV="$ENV -e UCX_NET_DEVICES=$OOB_IF -e RAY_memory_monitor_refresh_ms=0"
ENV="$ENV -e HF_HUB_OFFLINE=1"
ENV="$ENV -e VLLM_SLEEP_WHEN_IDLE=0"  # Disabled: stale requests prevent wake-up, causing hung requests
ENV="$ENV -e OMP_NUM_THREADS=1"  # Reduce threading overhead
ENV="$ENV -e RAY_CGRAPH_get_timeout=3600"  # 1hr compiled DAG timeout
ENV="$ENV -e RAY_CGRAPH_submit_timeout=3600"  # 1hr submit timeout
# PyTorch NCCL flight recorder (forensic data on stalls)
ENV="$ENV -e TORCH_NCCL_TRACE_BUFFER_SIZE=2000"
ENV="$ENV -e TORCH_NCCL_DUMP_ON_TIMEOUT=1"
ENV="$ENV -e TORCH_NCCL_DESYNC_DEBUG=1"
ENV="$ENV -e TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200"
ENV="$ENV -e TORCH_NCCL_ENABLE_MONITORING=0"
ENV="$ENV -e VLLM_TEST_FORCE_FP8_MARLIN=1"  # Force Marlin MoE backend - CUTLASS crashes on sm_121a
ENV="$ENV -e VLLM_ENCODER_CACHE_TOKENS=131072"  # 128K encoder cache (~0.75 GiB), decoupled from max_num_batched_tokens
# ENV="$ENV -e VLLM_NVFP4_GEMM_BACKEND=marlin"  # NVFP4-only, not needed for FP8
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
VOLS="$VOLS -v /tmp/vllm-head-ep.sh:/entrypoint.sh:ro"
VOLS="$VOLS -v /tmp/vllm-serve-cmd.sh:/vllm-serve-cmd.sh:ro"
# qwen3_5.py: use container's built-in version (imports transformers inline, not from transformers.models.qwen3_5)
VOLS="$VOLS -v /tmp/vllm_scheduler_patched.py:/usr/local/lib/python3.12/dist-packages/vllm/config/scheduler.py:ro"
VOLS_WORKER="-v /home/tom/.cache/huggingface:/root/.cache/huggingface"
VOLS_WORKER="$VOLS_WORKER -v /home/tom/llm/sitecustomize.py:/usr/lib/python3.12/sitecustomize.py:ro"
VOLS_WORKER="$VOLS_WORKER -v /tmp/vllm-worker-ep.sh:/entrypoint.sh:ro"
# qwen3_5.py: use container's built-in version
VOLS_WORKER="$VOLS_WORKER -v /tmp/vllm_scheduler_patched.py:/usr/local/lib/python3.12/dist-packages/vllm/config/scheduler.py:ro"

echo "=== Cleanup ==="
ssh spark-2 "docker rm -f vllm-head 2>/dev/null; for f in /tmp/vllm-head-ep.sh /tmp/vllm-serve-cmd.sh /tmp/vllm_scheduler_patched.py; do [ -d \"\$f\" ] && rmdir \"\$f\"; done; true"
ssh spark-3 "docker rm -f vllm-worker 2>/dev/null; for f in /tmp/vllm-worker-ep.sh /tmp/vllm_scheduler_patched.py; do [ -d \"\$f\" ] && rmdir \"\$f\"; done; true"

echo "=== Deploying entrypoint scripts ==="
scp -q /home/tom/llm/vllm-head-entrypoint.sh spark-2:/tmp/vllm-head-ep.sh
scp -q /home/tom/llm/vllm-worker-entrypoint.sh spark-3:/tmp/vllm-worker-ep.sh
scp -q /home/tom/llm/vllm_scheduler_patched.py spark-2:/tmp/vllm_scheduler_patched.py
scp -q /home/tom/llm/vllm_scheduler_patched.py spark-3:/tmp/vllm_scheduler_patched.py
# qwen3_5.py: using container's built-in version

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
VLLM_ARGS="$VLLM_ARGS --gpu-memory-utilization 0.70"
# VLLM_ARGS="$VLLM_ARGS --kv-cache-dtype fp8"  # Disabled: suspected cause of hard crashes on Spark
VLLM_ARGS="$VLLM_ARGS --max-num-batched-tokens 4096"
VLLM_ARGS="$VLLM_ARGS --distributed-executor-backend ray"
VLLM_ARGS="$VLLM_ARGS --enforce-eager"
VLLM_ARGS="$VLLM_ARGS --max-num-seqs 32"
VLLM_ARGS="$VLLM_ARGS --limit-mm-per-prompt '{\"video\": 0}'"
VLLM_ARGS="$VLLM_ARGS --reasoning-parser qwen3"
VLLM_ARGS="$VLLM_ARGS --host 0.0.0.0 --port 8000"

echo "#!/bin/bash" > /home/tom/llm/vllm-serve-cmd.sh
echo "exec vllm serve $MODEL $VLLM_ARGS" >> /home/tom/llm/vllm-serve-cmd.sh
scp -q /home/tom/llm/vllm-serve-cmd.sh spark-2:/tmp/vllm-serve-cmd.sh
ENV_WORKER="$ENV_WORKER -e RAY_HEAD_IP=$HEAD_IP"

echo "=== Starting worker on spark-3 ==="
ssh spark-3 "docker run -d --name vllm-worker --gpus all --shm-size 16g \\
    --network host --ipc host --restart=no --entrypoint bash \\
    $RDMA $ENV_WORKER $VOLS_WORKER $CONTAINER /entrypoint.sh"

echo "=== Starting head on spark-2 ==="
ssh spark-2 "docker run -d --name vllm-head --gpus all --shm-size 16g \\
    --network host --ipc host --restart=no --entrypoint bash \\
    $RDMA $ENV_HEAD $VOLS $CONTAINER /entrypoint.sh"

echo "=== Containers started ==="
echo "Head logs:   ssh spark-2 'docker logs -f vllm-head'"
echo "Worker logs: ssh spark-3 'docker logs -f vllm-worker'"
echo "API:         http://192.168.110.2:8000/v1/chat/completions"
