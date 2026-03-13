#!/bin/bash
# Generated from recipe: Qwen3.5-122B-FP8

# Environment variables
export NCCL_IB_HCA="rocep1s0f1,roceP2p1s0f1"

# Run the model
vllm serve Qwen/Qwen3.5-122B-A10B-FP8 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.7 \
  --port 8000 \
  --host 0.0.0.0 \
  --load-format fastsafetensors \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --chat-template unsloth.jinja \
  -tp 2 --distributed-executor-backend ray \
  --max-num-batched-tokens 8192
