#!/bin/bash
# Head node entrypoint: Ray head + vLLM serve with retry loop
set -u

EXPECTED_GPUS="${EXPECTED_GPUS:-2}"

# Prevent Ray from overwriting per-node NCCL/GLOO settings
mkdir -p /root/.config/vllm
echo '["NCCL_SOCKET_IFNAME","NCCL_IB_HCA","GLOO_SOCKET_IFNAME"]' > /root/.config/vllm/ray_non_carry_over_env_vars.json

echo "=== Starting Ray head ==="
ray start --head --port=6379 --node-ip-address="$VLLM_HOST_IP" \
    --object-store-memory=2000000000 --include-dashboard=false

gpu_count() { ray status 2>/dev/null | grep -oP '[\d.]+/[\d.]+\s+GPU' | head -1 | grep -oP '(?<=/)[\d.]+' | cut -d. -f1 || echo 0; }

echo "=== Waiting for $EXPECTED_GPUS GPUs ==="
for i in $(seq 1 120); do
    TOTAL_GPUS=$(gpu_count)
    if [ "${TOTAL_GPUS:-0}" -ge "$EXPECTED_GPUS" ]; then
        echo "Worker joined (${TOTAL_GPUS} GPUs available)."
        break
    fi
    [ "$i" -eq 120 ] && { echo "ERROR: Worker timeout (only ${TOTAL_GPUS} GPUs)"; exit 1; }
    sleep 2
done

echo "=== Starting vLLM (restarts on failure) ==="
RETRY=0
MAX_RETRIES=3
while [ "$RETRY" -lt "$MAX_RETRIES" ]; do
    VLLM_ATTENTION_BACKEND=TRITON_ATTN \
        RAY_ADDRESS="${VLLM_HOST_IP}:6379" \
        bash /vllm-serve-cmd.sh
    EXIT_CODE=$?
    RETRY=$((RETRY + 1))
    [ "$EXIT_CODE" -eq 0 ] && break

    # Wait for worker GPU if it disconnected
    TOTAL_GPUS=$(gpu_count)
    if [ "${TOTAL_GPUS:-0}" -lt "$EXPECTED_GPUS" ]; then
        echo "Worker GPU lost (${TOTAL_GPUS} GPUs). Waiting for reconnect..."
        for j in $(seq 1 120); do
            TOTAL_GPUS=$(gpu_count)
            [ "${TOTAL_GPUS:-0}" -ge "$EXPECTED_GPUS" ] && break
            sleep 2
        done
        [ "${TOTAL_GPUS:-0}" -lt 2 ] && { echo "ERROR: Worker gone (${TOTAL_GPUS} GPUs)"; exit 1; }
    fi
    echo "vLLM exited ($EXIT_CODE), retry $RETRY/$MAX_RETRIES in 15s..."
    sleep 15
done
[ "$RETRY" -ge "$MAX_RETRIES" ] && { echo "ERROR: Max retries"; exit 1; }
