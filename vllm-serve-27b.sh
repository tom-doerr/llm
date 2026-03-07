#!/bin/bash
exec vllm serve Qwen/Qwen3.5-27B-FP8 --trust-remote-code --gpu-memory-utilization 0.70 --reasoning-parser qwen3 --enforce-eager --host 0.0.0.0 --port 8000
