#!/bin/bash
# Start vLLM PP=3 across spark-1 + spark-2 (head) + spark-3
# Model: nvidia/Qwen3.5-397B-A17B-NVFP4 (~224 GB)
# Usage: ./start-vllm-pp3.sh [--debug]
set -e

DEBUG_NCCL=${DEBUG_NCCL:-0}
for arg in "$@"; do
  [[ "$arg" == "--debug" ]] && DEBUG_NCCL=1
done

MODEL="nvidia/Qwen3.5-397B-A17B-NVFP4"
CONTAINER="${CONTAINER:-vllm/vllm-openai:qwen3_5-cu130}"
HEAD_IP="192.168.110.2"
WORKER1_IP="192.168.110.1"
WORKER2_IP="192.168.100.11"

# Common env: NCCL IB with GDR disabled, Marlin for NVFP4
ENV="-e NCCL_NET_PLUGIN=none -e NCCL_DMABUF_ENABLE=0"
ENV="$ENV -e NCCL_NET_GDR_LEVEL=LOC -e NCCL_NET_GDR_C2C=0"
ENV="$ENV -e NCCL_SOCKET_IFNAME=enp1s0f0np0,enp1s0f1np1"
ENV="$ENV -e GLOO_SOCKET_IFNAME=enp1s0f0np0,enp1s0f1np1"
ENV="$ENV -e RAY_memory_monitor_refresh_ms=0"
ENV="$ENV -e HF_HUB_OFFLINE=1"
ENV="$ENV -e VLLM_SLEEP_WHEN_IDLE=0"
ENV="$ENV -e OMP_NUM_THREADS=1"
ENV="$ENV -e VLLM_USE_RAY_COMPILED_DAG=0"
ENV="$ENV -e RAY_CGRAPH_get_timeout=86400"
ENV="$ENV -e TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200"
ENV="$ENV -e TORCH_NCCL_ENABLE_MONITORING=0"
ENV="$ENV -e VLLM_TEST_FORCE_FP8_MARLIN=1"
ENV="$ENV -e VLLM_USE_FLASHINFER_MOE_FP4=0"
ENV="$ENV -e VLLM_NVFP4_GEMM_BACKEND=marlin"

if [ "$DEBUG_NCCL" = "1" ]; then
  ENV="$ENV -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,NET"
  ENV="$ENV -e NCCL_DEBUG_FILE=/tmp/nccl.%h.%p.log"
  echo "NCCL debug enabled"
fi

# Per-node IB HCAs
ENV_HEAD="$ENV -e VLLM_HOST_IP=$HEAD_IP -e EXPECTED_GPUS=3"
ENV_HEAD="$ENV_HEAD -e NCCL_IB_HCA='=rocep1s0f0:1,roceP2p1s0f0:1,rocep1s0f1:1,roceP2p1s0f1:1'"
ENV_W1="$ENV -e VLLM_HOST_IP=$WORKER1_IP"
ENV_W1="$ENV_W1 -e NCCL_IB_HCA='=rocep1s0f0:1,roceP2p1s0f0:1'"
ENV_W2="$ENV -e VLLM_HOST_IP=$WORKER2_IP"
ENV_W2="$ENV_W2 -e NCCL_IB_HCA='=rocep1s0f1:1,roceP2p1s0f1:1'"

RDMA="--device=/dev/infiniband --ulimit memlock=-1 --cap-add=IPC_LOCK"

VOLS="-v /home/tom/.cache/huggingface:/root/.cache/huggingface"
VOLS="$VOLS -v /home/tom/llm/sitecustomize.py:/usr/lib/python3.12/sitecustomize.py:ro"
VOLS_HEAD="$VOLS -v /tmp/vllm-head-ep.sh:/entrypoint.sh:ro"
VOLS_HEAD="$VOLS_HEAD -v /tmp/vllm-serve-cmd.sh:/vllm-serve-cmd.sh:ro"
VOLS_W1="$VOLS -v /tmp/vllm-worker-ep.sh:/entrypoint.sh:ro"
VOLS_W2="$VOLS -v /tmp/vllm-worker-ep.sh:/entrypoint.sh:ro"

echo "=== Cleanup ==="
docker rm -f vllm-worker1 2>/dev/null || true
ssh spark-2 "docker rm -f vllm-head 2>/dev/null; for f in /tmp/vllm-head-ep.sh /tmp/vllm-serve-cmd.sh; do [ -d \"\$f\" ] && rmdir \"\$f\"; done; true"
ssh spark-3 "docker rm -f vllm-worker2 2>/dev/null; for f in /tmp/vllm-worker-ep.sh; do [ -d \"\$f\" ] && rmdir \"\$f\"; done; true"

echo "=== Deploying entrypoint scripts ==="
scp -q /home/tom/llm/vllm-head-entrypoint.sh spark-2:/tmp/vllm-head-ep.sh
scp -q /home/tom/llm/vllm-worker-entrypoint.sh spark-3:/tmp/vllm-worker-ep.sh
cp /home/tom/llm/vllm-worker-entrypoint.sh /tmp/vllm-worker-ep.sh

echo "=== Building vLLM args ==="
echo "Mode: Pipeline Parallel (PP=3)"
VLLM_ARGS="--pipeline-parallel-size 3 --trust-remote-code"
VLLM_ARGS="$VLLM_ARGS --quantization modelopt_fp4"
VLLM_ARGS="$VLLM_ARGS --gpu-memory-utilization 0.85"
VLLM_ARGS="$VLLM_ARGS --kv-cache-dtype fp8"
VLLM_ARGS="$VLLM_ARGS --max-num-batched-tokens 4096"
VLLM_ARGS="$VLLM_ARGS --distributed-executor-backend ray"
VLLM_ARGS="$VLLM_ARGS --enforce-eager"
VLLM_ARGS="$VLLM_ARGS --limit-mm-per-prompt '{\"video\": 0}'"
VLLM_ARGS="$VLLM_ARGS --reasoning-parser qwen3"
VLLM_ARGS="$VLLM_ARGS --host 0.0.0.0 --port 8000"

echo "#!/bin/bash" > /home/tom/llm/vllm-serve-cmd.sh
echo "exec vllm serve $MODEL $VLLM_ARGS" >> /home/tom/llm/vllm-serve-cmd.sh
scp -q /home/tom/llm/vllm-serve-cmd.sh spark-2:/tmp/vllm-serve-cmd.sh

# Workers connect to head via their respective subnets
ENV_W1="$ENV_W1 -e RAY_HEAD_IP=192.168.110.2"   # spark-2 port 0
ENV_W2="$ENV_W2 -e RAY_HEAD_IP=192.168.100.10"   # spark-2 port 1

echo "=== Starting worker1 on spark-1 (local) ==="
docker run -d --name vllm-worker1 --gpus all --shm-size 16g \
    --network host --ipc host --restart=no --entrypoint bash \
    $RDMA $ENV_W1 $VOLS_W1 $CONTAINER /entrypoint.sh

echo "=== Starting worker2 on spark-3 ==="
ssh spark-3 "docker run -d --name vllm-worker2 --gpus all --shm-size 16g \\
    --network host --ipc host --restart=no --entrypoint bash \\
    $RDMA $ENV_W2 $VOLS_W2 $CONTAINER /entrypoint.sh"

echo "=== Starting head on spark-2 ==="
ssh spark-2 "docker run -d --name vllm-head --gpus all --shm-size 16g \\
    --network host --ipc host --restart=no --entrypoint bash \\
    $RDMA $ENV_HEAD $VOLS_HEAD $CONTAINER /entrypoint.sh"

echo "=== Containers started (PP=3) ==="
echo "Head logs:    ssh spark-2 'docker logs -f vllm-head'"
echo "Worker1 logs: docker logs -f vllm-worker1"
echo "Worker2 logs: ssh spark-3 'docker logs -f vllm-worker2'"
echo "API:          http://192.168.102.11:8000/v1/chat/completions"
