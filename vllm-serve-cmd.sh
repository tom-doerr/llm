#!/bin/bash
exec vllm serve Qwen/Qwen3.5-122B-A10B-FP8 --tensor-parallel-size 2 --trust-remote-code --gpu-memory-utilization 0.70 --kv-cache-dtype fp8 --max-num-batched-tokens 4096 --distributed-executor-backend ray --enforce-eager --limit-mm-per-prompt '{"video": 0}' --host 0.0.0.0 --port 8000
