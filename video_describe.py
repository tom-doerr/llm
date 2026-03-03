#!/usr/bin/env python3
"""Send video to vLLM and get per-second description."""
import time, base64, json, requests, os

V = '/home/tom/Videos/britney_max.mp4'
U = 'http://192.168.110.2:8000/v1/chat/completions'
M = 'Intel/Qwen3.5-122B-A10B-int4-AutoRound'
O = os.path.expanduser('~/llm/benchmark_results/britney_max_description.txt')
P = ('Describe every second of this video individually, '
     'one line per second. Format: 00:00 - description. '
     'Be specific about what is visible in each second.')

def main():
    from video_describe_run import run
    run(V, U, M, O, P)

if __name__ == '__main__':
    main()
