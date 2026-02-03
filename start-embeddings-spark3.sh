#!/bin/bash
# Deploy BGE embeddings on spark-3 (port 8001)
# Runs alongside vLLM worker with minimal memory footprint
set -e
MODEL="BAAI/bge-base-en-v1.5"
IMG="nvcr.io/nvidia/vllm:25.11-py3"
VOLS="-v /home/tom/.cache/huggingface:/root/.cache/huggingface"
ssh spark-3 "docker rm -f vllm-embed 2>/dev/null || true"
# Minimal memory settings to coexist with Qwen3-VL worker on unified memory
ssh spark-3 "docker run -d --name vllm-embed --gpus all --shm-size 1g -p 8001:8001 $VOLS $IMG \
    vllm serve $MODEL --task embed --host 0.0.0.0 --port 8001 --trust-remote-code \
    --gpu-memory-utilization 0.02 --max-num-seqs 1 --enforce-eager"
echo "API: http://spark-3:8001/v1/embeddings"
