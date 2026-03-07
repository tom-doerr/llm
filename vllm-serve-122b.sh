#!/bin/bash
exec vllm serve Intel/Qwen3.5-122B-A10B-int4-AutoRound --trust-remote-code --gpu-memory-utilization 0.70 --reasoning-parser qwen3 --enforce-eager --host 0.0.0.0 --port 8000
