#!/usr/bin/env python3
"""Benchmark llama.cpp with parallel requests (encode + decode)."""
import base64, io, json, time, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request
from PIL import Image
import numpy as np

API = "http://192.168.102.11:8000/v1/chat/completions"

def make_png(w, h):
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:,:,0] = np.linspace(0, 255, w, dtype=np.uint8)
    arr[:,:,1] = np.linspace(0, 255, h, dtype=np.uint8).reshape(-1,1)
    arr[:,:,2] = 128
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()

def do_req(payload):
    data = json.dumps(payload).encode()
    r = urllib.request.Request(API, data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(r, timeout=600) as resp:
            result = json.loads(resp.read())
        dt = time.time() - t0
        u = result.get("usage", {})
        t = result.get("timings", {})
        return {"ok": True, "dt": dt, "pt": u.get("prompt_tokens", 0),
                "ct": u.get("completion_tokens", 0),
                "prompt_ms": t.get("prompt_ms", 0),
                "pred_ms": t.get("predicted_ms", 0)}
    except Exception as e:
        return {"ok": False, "dt": time.time() - t0, "error": str(e)[:80]}

def make_text_payload(mt=64):
    return {"model": "qwen3.5", "max_tokens": mt, "temperature": 0.7,
        "messages": [{"role": "user", "content": "Write a haiku about the ocean."}]}

def make_img_payload(b64, mt=64):
    return {"model": "qwen3.5", "max_tokens": mt, "temperature": 0.7,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": "Describe briefly."}]}]}

def run_parallel(payloads, label):
    n = len(payloads)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n) as ex:
        results = list(ex.map(do_req, payloads))
    wall = time.time() - t0
    ok = [r for r in results if r["ok"]]
    fail = n - len(ok)
    tot_ct = sum(r["ct"] for r in ok) if ok else 0
    tps = tot_ct / wall if wall > 0 else 0
    avg_dt = sum(r["dt"] for r in ok) / len(ok) if ok else 0
    print(f"  {label:>16} | c={n:>3} | {wall:>6.1f}s | "
          f"{tps:>6.1f} tok/s | avg {avg_dt:>5.1f}s | fail={fail}")
    return results

concurrencies = [1, 2, 4, 8, 16]
MT = 64  # max tokens per request

# Warmup
print("Warming up...")
do_req(make_text_payload(1))

# Text-only sweep
print(f"\n=== Text decode (max_tokens={MT}) ===")
print(f"  {'mode':>16} | {'c':>5} | {'wall':>6} | {'tok/s':>9} | {'avg':>9} | fail")
print("  " + "-" * 70)
for c in concurrencies:
    run_parallel([make_text_payload(MT)] * c, "text")

# Image encode sweep (512x512, max_tokens=1 to isolate encode)
print(f"\n=== Image encode only (512x512, max_tokens=1) ===")
print(f"  {'mode':>16} | {'c':>5} | {'wall':>6} | {'tok/s':>9} | {'avg':>9} | fail")
print("  " + "-" * 70)
b64_512 = base64.b64encode(make_png(512, 512)).decode()
for c in concurrencies:
    run_parallel([make_img_payload(b64_512, 1)] * c, "img-enc-512")

# Image encode+decode (512x512, max_tokens=64)
print(f"\n=== Image encode+decode (512x512, max_tokens={MT}) ===")
print(f"  {'mode':>16} | {'c':>5} | {'wall':>6} | {'tok/s':>9} | {'avg':>9} | fail")
print("  " + "-" * 70)
for c in concurrencies:
    run_parallel([make_img_payload(b64_512, MT)] * c, "img-full-512")

# Large image encode (1024x1024, max_tokens=1)
print(f"\n=== Image encode only (1024x1024, max_tokens=1) ===")
print(f"  {'mode':>16} | {'c':>5} | {'wall':>6} | {'tok/s':>9} | {'avg':>9} | fail")
print("  " + "-" * 70)
b64_1k = base64.b64encode(make_png(1024, 1024)).decode()
for c in concurrencies:
    run_parallel([make_img_payload(b64_1k, 1)] * c, "img-enc-1024")
