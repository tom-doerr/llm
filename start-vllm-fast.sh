#!/bin/bash
# Start fast single-node vLLM on spark-2 (Qwen3.5-35B-A3B-FP8)
# Runs on port 8001 alongside the multi-node 122B on port 8000
set -e
MODEL="${MODEL:-Qwen/Qwen3.5-35B-A3B-FP8}"
CONTAINER="${CONTAINER:-vllm/vllm-openai:qwen3_5-cu130}"
PORT=8001; NAME="vllm-fast"
ENV="-e HF_HUB_OFFLINE=1 -e VLLM_SLEEP_WHEN_IDLE=1"
ENV="$ENV -e OMP_NUM_THREADS=1 -e VLLM_TEST_FORCE_FP8_MARLIN=1"
VOLS="-v /home/tom/.cache/huggingface:/root/.cache/huggingface"
VOLS="$VOLS -v /home/tom/llm/sitecustomize.py:/usr/lib/python3.12/sitecustomize.py:ro"
VOLS="$VOLS -v /tmp/vllm_scheduler_patched.py:/usr/local/lib/python3.12/dist-packages/vllm/config/scheduler.py:ro"
ARGS="--trust-remote-code --gpu-memory-utilization 0.50 --kv-cache-dtype fp8"
ARGS="$ARGS --enforce-eager --host 0.0.0.0 --port $PORT"
echo "=== Setup ==="
scp -q /home/tom/llm/vllm_scheduler_patched.py spark-2:/tmp/vllm_scheduler_patched.py
ssh spark-2 "docker rm -f $NAME 2>/dev/null || true"
# Write serve command to a script to avoid quoting issues over SSH
cat > /tmp/vllm-fast-serve.sh <<SCRIPT
#!/bin/bash
exec vllm serve $MODEL $ARGS
SCRIPT
scp -q /tmp/vllm-fast-serve.sh spark-2:/tmp/vllm-fast-serve.sh
echo "=== Starting $NAME on spark-2 (port $PORT) ==="
ssh spark-2 "docker run -d --name $NAME --gpus all --shm-size 16g \
    --network host --ipc host --restart=on-failure:10 \
    -v /tmp/vllm-fast-serve.sh:/serve.sh:ro \
    --entrypoint bash $ENV $VOLS $CONTAINER /serve.sh"
echo "API: http://192.168.102.11:$PORT/v1/chat/completions"
