#!/usr/bin/env python3
"""Benchmark vLLM server."""
import argparse, asyncio, time, aiohttp, random, string, base64, io, json, os
from datetime import datetime
from PIL import Image
import numpy as np

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
    timeout = aiohttp.ClientTimeout(total=1800)  # 30 min for long prefills
    async with aiohttp.ClientSession(timeout=timeout) as session:
        return await asyncio.gather(*[bounded_request(session) for _ in range(num_requests)])

URL = "http://192.168.102.11:8000/v1/chat/completions"
BASE_URL = "http://192.168.102.11:8000"
MODEL = "QuantTrio/Qwen3-VL-235B-A22B-Instruct-AWQ"
RESULTS_DIR = os.path.expanduser("~/llm/benchmark_results")
WORDS = ["the","be","to","of","and","a","in","that","have","it","for","not","on",
    "with","as","you","do","at","this","but","by","from","they","we","say","or"]

def gen_image_b64(w, h):
    """Generate random noise image as base64 JPEG."""
    img = Image.new('RGB', (w, h))
    img.putdata([(random.randint(0,255),)*3 for _ in range(w*h)])
    buf = io.BytesIO(); img.save(buf, format='JPEG', quality=80)
    return base64.b64encode(buf.getvalue()).decode()

def gen_prompt(target_tokens):
    """Generate random text targeting approximate token count."""
    return " ".join(random.choice(WORDS) for _ in range(int(target_tokens * 1.3)))

def run(prompt, max_tokens, num_requests, concurrency):
    start = time.perf_counter()
    results = asyncio.run(run_benchmark(URL, MODEL, prompt, max_tokens, num_requests, concurrency))
    elapsed = time.perf_counter() - start
    encode_toks = sum(pt for pt,_,_ in results)
    decode_toks = sum(ct for _,ct,_ in results)
    lats = sorted([l for _,_,l in results])
    return elapsed, encode_toks/elapsed, decode_toks/elapsed, lats

def run_prefill(target_tokens, num_requests=5, concurrency=1):
    """Benchmark prefill speed at given prompt length."""
    prompt = gen_prompt(target_tokens)
    start = time.perf_counter()
    results = asyncio.run(run_benchmark(URL, MODEL, prompt, 1, num_requests, concurrency))
    elapsed = time.perf_counter() - start
    actual_toks = sum(pt for pt,_,_ in results)
    return actual_toks / num_requests, actual_toks / elapsed

async def img_req(s, url, model, b64, mt):
    c = [{"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}},
         {"type":"text","text":"Describe."}]
    async with s.post(url, json={"model":model,"messages":[{"role":"user","content":c}],
        "max_tokens":mt}) as r: d=await r.json(); return d["usage"]["prompt_tokens"]

def run_img(w,h,n,c):
    b=gen_image_b64(w,h);sem=asyncio.Semaphore(c);t=aiohttp.ClientTimeout(total=600)
    async def go():
        async with aiohttp.ClientSession(timeout=t) as s:
            st=time.perf_counter();r=await asyncio.gather(*[img_req(s,URL,MODEL,b,32) for _ in range(n)])
            return time.perf_counter()-st,sum(r)
    return asyncio.run(go())

def get_vllm_config():
    """Fetch vLLM server config."""
    import requests
    try:
        r = requests.get(f"{BASE_URL}/v1/models", timeout=5)
        return {"models": r.json(), "url": BASE_URL}
    except: return {"url": BASE_URL}

def save_results(name, data):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"{RESULTS_DIR}/{ts}_{name}.json"
    with open(path, "w") as f: json.dump(data, f, indent=2)
    print(f"Saved: {path}"); return path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="Continue writing numbers: 1, 2, 3, 4, 5,")
    parser.add_argument("-t", type=int, default=32)
    parser.add_argument("-n", type=int, default=10)
    parser.add_argument("-c", type=int, default=1)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--prefill", action="store_true", help="Benchmark prefill at various lengths")
    parser.add_argument("--image", action="store_true", help="Benchmark image throughput")
    args = parser.parse_args()
    if args.image:
        print("res\tc\treq/s\timg_tok/s", flush=True)
        for w in [256,512,1024,2048]:
            for c in [1,4,8,16,32]:
                n=max(c,4); el,toks = run_img(w,w,n,c)
                print(f"{w}x{w}\t{c}\t{n/el:.2f}\t{toks/el:.1f}", flush=True)
    elif args.prefill:
        print("target\tactual\tenc_tok/s", flush=True)
        for toks in [128, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]:
            actual, rate = run_prefill(toks)
            print(f"{toks}\t{actual:.0f}\t{rate:.1f}", flush=True)
    elif args.sweep:
        print("c\treq/s\tenc/s\tdec/s\tp50\tp95\tp99"); rows=[]
        for conc in [1,2,4,8,16,32,64,128,256]:
            el,enc,dec,lats = run(args.prompt, args.t, conc, conc)
            p50,p95,p99 = np.percentile(lats,[50,95,99])
            print(f"{conc}\t{conc/el:.2f}\t{enc:.1f}\t{dec:.1f}\t{p50:.2f}\t{p95:.2f}\t{p99:.2f}")
            rows.append({"c":conc,"req_s":conc/el,"enc":enc,"dec":dec,"p50":p50,"p95":p95,"p99":p99})
        save_results("sweep", {"config": get_vllm_config(), "results": rows})
    else:
        el,enc,dec,lats = run(args.prompt, args.t, args.n, args.c)
        p50,p95,p99 = np.percentile(lats,[50,95,99])
        print(f"c={args.c} req/s={args.n/el:.2f} enc={enc:.1f} dec={dec:.1f} p50={p50:.2f}s")

if __name__ == "__main__": main()
