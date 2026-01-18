#!/usr/bin/env python3
"""Benchmark vLLM server."""
import argparse, asyncio, time, aiohttp, random

async def make_request(session, url, model, prompt, max_tokens):
    start_time = time.perf_counter()
    prompt_text = f"[{random.randint(0,999999)}] {prompt}"
    async with session.post(url, json={"model": model,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": max_tokens, "stream": False}) as resp:
        data = await resp.json()
        usage = data["usage"]
        return usage["prompt_tokens"], usage["completion_tokens"], time.perf_counter()-start_time

async def run_benchmark(url, model, prompt, max_tokens, num_requests, concurrency):
    semaphore = asyncio.Semaphore(concurrency)
    async def bounded_request(session):
        async with semaphore: return await make_request(session, url, model, prompt, max_tokens)
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        return await asyncio.gather(*[bounded_request(session) for _ in range(num_requests)])

URL = "http://192.168.102.11:8000/v1/chat/completions"
MODEL = "QuantTrio/Qwen3-VL-235B-A22B-Instruct-AWQ"

def run(prompt, max_tokens, num_requests, concurrency):
    start = time.perf_counter()
    results = asyncio.run(run_benchmark(URL, MODEL, prompt, max_tokens, num_requests, concurrency))
    elapsed = time.perf_counter() - start
    encode_toks = sum(pt for pt,_,_ in results)
    decode_toks = sum(ct for _,ct,_ in results)
    return elapsed, encode_toks/elapsed, decode_toks/elapsed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="Write a haiku about AI.")
    parser.add_argument("-t", type=int, default=32)
    parser.add_argument("-n", type=int, default=10)
    parser.add_argument("-c", type=int, default=1)
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()
    if args.sweep:
        print("c\treq/s\tenc/s\tdec/s")
        for conc in [1,2,4,8,16,32,64,128,256]:
            elapsed,enc,dec = run(args.prompt, args.t, conc, conc)
            print(f"{conc}\t{conc/elapsed:.2f}\t{enc:.1f}\t{dec:.1f}")
    else:
        elapsed,enc,dec = run(args.prompt, args.t, args.n, args.c)
        print(f"c={args.c} req/s={args.n/elapsed:.2f} enc={enc:.1f} dec={dec:.1f}")

if __name__ == "__main__": main()
