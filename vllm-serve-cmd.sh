#!/bin/bash
exec vllm serve Qwen/Qwen3.5-122B-A10B-FP8 --tensor-parallel-size 2 --trust-remote-code --gpu-memory-utilization 0.70 --max-num-batched-tokens 4096 --distributed-executor-backend ray --enforce-eager --max-num-seqs 32 --limit-mm-per-prompt '{"video": 0}' --reasoning-parser qwen3 --host 0.0.0.0 --port 8000
