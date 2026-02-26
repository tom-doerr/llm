#!/bin/bash
# Start vLLM GGUF multi-node on spark-2 (head) + spark-3 (worker)
set -e

MODEL="/root/.cache/huggingface/hub/models--unsloth--Qwen3.5-397B-A17B-GGUF/snapshots/abcbf9b1686cbdec98d678f3db51f04abb7a7ca2/Q3_K_M/Qwen3.5-397B-A17B-Q3_K_M-00001-of-00005.gguf"
CFG="/root/llm-gguf-config"
CONTAINER="vllm-qwen35-gguf:latest"
OOB_IF="enp1s0f1np1"
HEAD_IP="192.168.100.10"
WORKER_IP="192.168.100.11"

ENV="-e NCCL_NET_PLUGIN=none -e NCCL_DMABUF_ENABLE=0"
ENV="$ENV -e NCCL_NET_GDR_LEVEL=LOC -e NCCL_NET_GDR_C2C=0"
ENV="$ENV -e NCCL_IB_HCA='=rocep1s0f1:1,roceP2p1s0f1:1'"
ENV="$ENV -e NCCL_SOCKET_IFNAME=$OOB_IF -e GLOO_SOCKET_IFNAME=$OOB_IF"
ENV="$ENV -e UCX_NET_DEVICES=$OOB_IF -e RAY_memory_monitor_refresh_ms=0"
ENV="$ENV -e HF_HUB_OFFLINE=1 -e VLLM_SLEEP_WHEN_IDLE=1 -e OMP_NUM_THREADS=1"
ENV="$ENV -e VLLM_ATTENTION_BACKEND=TRITON_ATTN"
for arg in "$@"; do
  [[ "$arg" == "--debug" ]] && ENV="$ENV -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,NET"
done

ENV_HEAD="$ENV -e VLLM_HOST_IP=$HEAD_IP"
ENV_WORKER="$ENV -e VLLM_HOST_IP=$WORKER_IP -e RAY_HEAD_IP=$HEAD_IP"

# Volume mounts - patches baked into image, no vllm clone needed
VOLS="-v /home/tom/.cache/huggingface:/root/.cache/huggingface"
VOLS="$VOLS -v /home/tom/llm/sitecustomize.py:/usr/lib/python3.12/sitecustomize.py:ro"
VOLS="$VOLS -v /home/tom/llm-gguf-config:/root/llm-gguf-config:ro"
VOLS="$VOLS -v /tmp/vllm-head-entrypoint.sh:/entrypoint.sh:ro"
VOLS="$VOLS -v /tmp/vllm-serve-cmd.sh:/vllm-serve-cmd.sh:ro"
VOLS_WORKER="-v /home/tom/.cache/huggingface:/root/.cache/huggingface"
VOLS_WORKER="$VOLS_WORKER -v /home/tom/llm/sitecustomize.py:/usr/lib/python3.12/sitecustomize.py:ro"
VOLS_WORKER="$VOLS_WORKER -v /home/tom/llm-gguf-config:/root/llm-gguf-config:ro"
VOLS_WORKER="$VOLS_WORKER -v /tmp/vllm-worker-entrypoint.sh:/entrypoint.sh:ro"

RDMA="--device=/dev/infiniband --ulimit memlock=-1 --cap-add=IPC_LOCK"

echo "=== Building vLLM serve command ==="
VLLM_ARGS="--hf-config-path $CFG --tokenizer $CFG"
VLLM_ARGS="$VLLM_ARGS --tensor-parallel-size 2 --trust-remote-code"
VLLM_ARGS="$VLLM_ARGS --quantization gguf --dtype bfloat16 --enforce-eager"
VLLM_ARGS="$VLLM_ARGS --gpu-memory-utilization 0.85 --block-size 16"
VLLM_ARGS="$VLLM_ARGS --mamba-block-size 16"
VLLM_ARGS="$VLLM_ARGS --max-model-len 4096 --max-num-batched-tokens 4096"
VLLM_ARGS="$VLLM_ARGS --distributed-executor-backend ray"
VLLM_ARGS="$VLLM_ARGS --host 0.0.0.0 --port 8000"

echo "#!/bin/bash" > /home/tom/llm/vllm-serve-cmd.sh
echo "exec vllm serve $MODEL $VLLM_ARGS" >> /home/tom/llm/vllm-serve-cmd.sh

echo "=== Deploying entrypoint scripts ==="
scp -q /home/tom/llm/vllm-head-entrypoint.sh spark-2:/tmp/vllm-head-entrypoint.sh
scp -q /home/tom/llm/vllm-worker-entrypoint.sh spark-3:/tmp/vllm-worker-entrypoint.sh
scp -q /home/tom/llm/vllm-serve-cmd.sh spark-2:/tmp/vllm-serve-cmd.sh

echo "=== Cleanup ==="
ssh spark-2 "docker rm -f vllm-head 2>/dev/null || true"
ssh spark-3 "docker rm -f vllm-worker 2>/dev/null || true"

# echo "=== Drop caches (run manually with TTY if needed) ==="
# ssh -t spark-2 "sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'"
# ssh -t spark-3 "sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'"

echo "=== Starting worker on spark-3 ==="
ssh spark-3 "docker run -d --name vllm-worker --gpus all --shm-size 16g \
    --network host --ipc host --restart=on-failure:10 \
    $RDMA $ENV_WORKER $VOLS_WORKER --entrypoint bash $CONTAINER /entrypoint.sh"

echo "=== Starting head on spark-2 ==="
ssh spark-2 "docker run -d --name vllm-head --gpus all --shm-size 16g \
    --network host --ipc host --restart=on-failure:10 \
    $RDMA $ENV_HEAD $VOLS --entrypoint bash $CONTAINER /entrypoint.sh"

echo "=== Containers started ==="
echo "Head logs:   ssh spark-2 'docker logs -f vllm-head'"
echo "Worker logs: ssh spark-3 'docker logs -f vllm-worker'"
echo "API:         http://192.168.102.11:8000/v1/chat/completions"
