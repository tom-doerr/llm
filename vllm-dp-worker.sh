#!/bin/bash
MODEL="/root/.cache/huggingface/hub/models--Intel--Qwen3.5-122B-A10B-int4-AutoRound/snapshots/3f4ba633bf17c19e41f874def42760bf50925a3e"
exec vllm serve "$MODEL" \
  --trust-remote-code \
  --gpu-memory-utilization ${GPU_MEM_UTIL:-0.60} \
  --reasoning-parser qwen3 \
  --enforce-eager \
  --max-num-seqs 32 \
  --max-num-batched-tokens 4096 \
  --limit-mm-per-prompt '{"video": 0}' \
  --data-parallel-size 2 \
  --data-parallel-size-local 1 \
  --data-parallel-address 192.168.100.10 \
  --data-parallel-rpc-port 13345 \
  --data-parallel-start-rank 1 \
  --headless
