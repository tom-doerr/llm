#!/bin/bash
# Worker node entrypoint: retry Ray join until head is available
set -u

HEAD_ADDR="${RAY_HEAD_IP}:6379"

echo "=== Waiting for Ray head at $HEAD_ADDR ==="
for i in $(seq 1 120); do
    if ray start --address="$HEAD_ADDR" \
        --node-ip-address="$VLLM_HOST_IP" \
        --object-store-memory=2000000000 --block; then
        echo "Ray worker exited cleanly."
        exit 0
    fi
    echo "Ray worker exited ($?), reconnecting in 5s... ($i/120)"
    ray stop --force 2>/dev/null || true
    sleep 5
done
echo "ERROR: Could not connect after 120 attempts."
exit 1
