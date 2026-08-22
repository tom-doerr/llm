#!/bin/bash
# Deploy Qwen3.8-27B-FP8 single-node (TP=1) on spark-2.
# Clients:   http://spark-2:8000/v1  (later: http://vllm.tail620cfa.ts.net:8000/v1)
# Direct:    http://192.168.110.2:8000/v1  (RDMA link, host-management tools only)
# Usage: ./deploy-qwen3.8-27b-fp8.sh [stop|--no-build]
#   KV_NVME=1 [KV_NVME_MTP=1] ./deploy-qwen3.8-27b-fp8.sh -> experimental: KV cache spills to
#            spark-2's NVMe (vLLM OffloadingConnector fs tier, MTP OFF because
#            offload+spec-decode is broken upstream). See ~/llm/CLAUDE.md.
#   GPU_MEM=0.60  -> override --gpu-memory-utilization (default 0.60)
set -e
cd "$(dirname "$0")"
REMOTE_REPO=/home/tom/spark-vllm-docker-aug   # fresh upstream clone (Aug 2026), NOT ~/spark-vllm-docker
RECIPE=qwen3.8-27b-fp8
[ "${KV_NVME:-0}" = 1 ] && RECIPE=qwen3.8-27b-fp8-kvnvme
[ "${KV_NVME:-0}" = 1 ] && [ "${KV_NVME_MTP:-0}" = 1 ] && RECIPE=qwen3.8-27b-fp8-kvnvme-mtp
GPU_MEM="${GPU_MEM:-0.60}"

if [ "${1:-}" = "stop" ]; then
    ssh spark-2 'docker rm -f vllm_node 2>/dev/null' || true
    echo "Stopped."; exit 0
fi

# Always sync the recipe(s) from this repo (source of truth) to spark-2.
scp -q qwen3.8-27b-fp8.yaml qwen3.8-27b-fp8-kvnvme.yaml qwen3.8-27b-fp8-kvnvme-mtp.yaml "spark-2:$REMOTE_REPO/recipes/"

# Rebuild the image from upstream prebuilt wheels (skip with --no-build, e.g. watchdog).
# ~40 min on spark-2's WAN; the old image stays as vllm-node:20260409.
if [ "${1:-}" != "--no-build" ]; then
    ssh spark-2 "cd $REMOTE_REPO && git pull -q --ff-only && ./build-and-copy.sh -t vllm-node-aug --use-wheels"
fi

ssh spark-2 'docker rm -f vllm_node 2>/dev/null' || true

EXTRA_ENV=""
# jemalloc (if the image ships one) keeps the EngineCore host heap from growing
# without bound under prompt_logprobs load (see ~/llm/CLAUDE.md memory-leak note).
JEMALLOC=$(ssh spark-2 "docker run --rm --entrypoint sh vllm-node-aug -c 'ls /usr/local/lib/python3.12/dist-packages/ray/core/libjemalloc.so /usr/lib/aarch64-linux-gnu/libjemalloc.so.2 2>/dev/null | head -1'" 2>/dev/null || true)
if [ -n "$JEMALLOC" ]; then
    EXTRA_ENV="-e LD_PRELOAD=$JEMALLOC -e MALLOC_CONF=background_thread:true,dirty_decay_ms:1000,muzzy_decay_ms:1000"
    echo "jemalloc: $JEMALLOC"
else
    echo "jemalloc: not found in image, running without LD_PRELOAD"
fi
if [ "${KV_NVME:-0}" = 1 ]; then
    # fs-tier block files are keyed by Python hashes -> must be stable across restarts.
    EXTRA_ENV="$EXTRA_ENV -e PYTHONHASHSEED=0"
    ssh spark-2 'mkdir -p ~/.cache/vllm/kv_nvme'
fi

# Solo, TP=1. gpu-mem 0.60 = ~77 GiB: 28 GiB weights + ~12 activations + ~37 KV
# (3x the Qwen3.6 deployment's 12.6 GiB). Raise only after checking
# MemAvailable on spark-2 under load; the engine grows ~20-30 GB outside the
# budget over days (CUDA caching allocator / prompt_logprobs), and 0.85 hard-froze the box.
ssh spark-2 "cd $REMOTE_REPO && python3 run-recipe.py $RECIPE \
    --solo --tp 1 \
    -e HF_HUB_OFFLINE=1 \
    -e UCX_MEM_MMAP_HOOK_MODE=none \
    -e VLLM_SLEEP_WHEN_IDLE=1 \
    $EXTRA_ENV \
    --gpu-mem $GPU_MEM -d -- --generation-config auto --override-generation-config '{\"presence_penalty\":1.0}'"

echo "API (clients): http://spark-2:8000/v1   direct: http://192.168.110.2:8000/v1"
echo "Logs: ssh spark-2 'docker logs -f vllm_node'"
