#!/bin/bash
# Head node entrypoint: Ray head + vLLM serve with retry loop
set -u

echo "=== Starting Ray head ==="
ray start --head --port=6379 --node-ip-address="$VLLM_HOST_IP"

echo "=== Waiting for worker node with GPU to join ==="
for i in $(seq 1 120); do
    GPUS=$(ray status 2>/dev/null | grep -oP '[\d.]+/[\d.]+\s+GPU' | head -1 | grep -oP '[\d.]+(?=/)' || echo 0)
    TOTAL_GPUS=$(ray status 2>/dev/null | grep -oP '[\d.]+/[\d.]+\s+GPU' | head -1 | grep -oP '(?<=/)[\d.]+' || echo 0)
    if [ "$(echo "$TOTAL_GPUS >= 2" | bc)" -eq 1 ]; then
        echo "Worker joined (${TOTAL_GPUS} GPUs available)."
        break
    fi
    [ "$i" -eq 120 ] && { echo "ERROR: Worker timeout (only ${TOTAL_GPUS} GPUs)"; exit 1; }
    sleep 2
done

echo "=== Starting vLLM (restarts on failure) ==="
RETRY=0
MAX_RETRIES=10
while [ "$RETRY" -lt "$MAX_RETRIES" ]; do
    VLLM_ATTENTION_BACKEND=TRITON_ATTN \
        RAY_ADDRESS="${VLLM_HOST_IP}:6379" \
        bash /vllm-serve-cmd.sh
    EXIT_CODE=$?
    RETRY=$((RETRY + 1))
    [ "$EXIT_CODE" -eq 0 ] && break

    # Wait for worker GPU if it disconnected
    TOTAL_GPUS=$(ray status 2>/dev/null | grep -oP '[\d.]+/[\d.]+\s+GPU' | head -1 | grep -oP '(?<=/)[\d.]+' || echo 0)
    if [ "$(echo "$TOTAL_GPUS < 2" | bc)" -eq 1 ]; then
        echo "Worker GPU lost (${TOTAL_GPUS} GPUs). Waiting for reconnect..."
        for j in $(seq 1 120); do
            TOTAL_GPUS=$(ray status 2>/dev/null | grep -oP '[\d.]+/[\d.]+\s+GPU' | head -1 | grep -oP '(?<=/)[\d.]+' || echo 0)
            [ "$(echo "$TOTAL_GPUS >= 2" | bc)" -eq 1 ] && break
            sleep 2
        done
        [ "$(echo "$TOTAL_GPUS < 2" | bc)" -eq 1 ] && { echo "ERROR: Worker gone (${TOTAL_GPUS} GPUs)"; exit 1; }
    fi
    echo "vLLM exited ($EXIT_CODE), retry $RETRY/$MAX_RETRIES in 15s..."
    sleep 15
done
[ "$RETRY" -ge "$MAX_RETRIES" ] && { echo "ERROR: Max retries"; exit 1; }
