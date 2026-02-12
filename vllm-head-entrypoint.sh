#!/bin/bash
# Head node entrypoint: Ray head + vLLM serve with retry loop
set -u

echo "=== Starting Ray head ==="
ray start --head --port=6379 --node-ip-address="$VLLM_HOST_IP"

echo "=== Waiting for worker node to join ==="
for i in $(seq 1 120); do
    NODES=$(ray status 2>/dev/null | grep -c "node_" || true)
    if [ "$NODES" -ge 2 ]; then
        echo "Worker joined (${NODES} nodes)."
        break
    fi
    [ "$i" -eq 120 ] && { echo "ERROR: Worker timeout"; exit 1; }
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

    # Wait for worker if it disconnected
    NODES=$(ray status 2>/dev/null | grep -c "node_" || true)
    if [ "$NODES" -lt 2 ]; then
        echo "Worker lost. Waiting for reconnect..."
        for j in $(seq 1 120); do
            NODES=$(ray status 2>/dev/null | grep -c "node_" || true)
            [ "$NODES" -ge 2 ] && break
            sleep 2
        done
        [ "$NODES" -lt 2 ] && { echo "ERROR: Worker gone"; exit 1; }
    fi
    echo "vLLM exited ($EXIT_CODE), retry $RETRY/$MAX_RETRIES in 15s..."
    sleep 15
done
[ "$RETRY" -ge "$MAX_RETRIES" ] && { echo "ERROR: Max retries"; exit 1; }
