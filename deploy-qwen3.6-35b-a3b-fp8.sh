#!/bin/bash
# Deploy Qwen3.6-35B-A3B-FP8 single-node (TP=1) on spark-2, on the Aug-2026 image.
# Switched back to this model Aug 23 2026 (user call): the dense Qwen3.8-27B measured
# 3.4x slower per token / -46% requests-per-hour on the same workload (see CLAUDE.md).
# Kept from the Qwen3.8 era: vllm-node-aug image (vLLM 0.26.1 nightly).
# Clients:   http://spark-2:8000/v1  (later: http://vllm.tail620cfa.ts.net:8000/v1)
# Direct:    http://192.168.110.2:8000/v1  (RDMA link, host-management tools only)
# Usage: ./deploy-qwen3.6-35b-a3b-fp8.sh [stop|--no-build]
#   (the KV_NVME variants live in deploy-qwen3.8-27b-fp8.sh)
#   GPU_MEM=...   -> override --gpu-memory-utilization (default 0.70; user call Aug 28).
#            0.85 hard-froze spark-2 historically; watch MemAvailable under load.
#   EXTRA_VLLM_ARGS='...' -> appended to the vllm serve command (vLLM takes the LAST
#            value of a repeated flag, so e.g. '--max-num-batched-tokens 32768' overrides the recipe)
set -e
cd "$(dirname "$0")"
REMOTE_REPO=/home/tom/spark-vllm-docker-aug   # fresh upstream clone (Aug 2026), NOT ~/spark-vllm-docker
RECIPE=qwen3.6-35b-a3b-fp8      # upstream recipe, ships with the clone
GPU_MEM="${GPU_MEM:-0.70}"

if [ "${1:-}" = "stop" ]; then
    ssh spark-2 'docker rm -f vllm_node 2>/dev/null' || true
    echo "Stopped."; exit 0
fi

# No recipe scp: qwen3.6-35b-a3b-fp8.yaml ships with the upstream clone.

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

# Solo, TP=1, gpu-mem 0.70: ~35 GiB weights leaves ~37 GiB of KV (about 3.69M tokens).
# 65536 batched tokens is safe for this
# MoE (its per-expert intermediate is small, so the int32 overflow of vllm#53390 —
# which caps the dense 27B at 32768 — cannot trigger here).
# Watch MemAvailable on spark-2 under load; 0.85 hard-froze this box historically.
ssh spark-2 "cd $REMOTE_REPO && python3 run-recipe.py $RECIPE \
    --solo --tp 1 -t vllm-node-aug \
    -e HF_HUB_OFFLINE=1 \
    -e UCX_MEM_MMAP_HOOK_MODE=none \
    -e VLLM_SLEEP_WHEN_IDLE=1 \
    $EXTRA_ENV \
    --gpu-mem $GPU_MEM -d -- --generation-config auto --override-generation-config '{\"presence_penalty\":1.0}' --max-num-batched-tokens 65536 ${EXTRA_VLLM_ARGS:-}"

echo "API (clients): http://spark-2:8000/v1   direct: http://192.168.110.2:8000/v1"
echo "Logs: ssh spark-2 'docker logs -f vllm_node'"
