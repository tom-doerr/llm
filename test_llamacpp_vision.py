#!/usr/bin/env python3
"""Test llama.cpp vision capabilities with various image sizes."""
import base64, io, json, time, sys
import urllib.request
from PIL import Image
import numpy as np

API = "http://192.168.102.11:8000/v1/chat/completions"

def make_png(w, h):
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:,:,0] = np.linspace(0, 255, w, dtype=np.uint8)
    arr[:,:,1] = np.linspace(0, 255, h, dtype=np.uint8).reshape(-1,1)
    arr[:,:,2] = 128
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def req(payload):
    data = json.dumps(payload).encode()
    r = urllib.request.Request(API, data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(r, timeout=300) as resp:
            result = json.loads(resp.read())
        dt = time.time() - t0
        c = result["choices"][0]["message"]["content"]
        u = result.get("usage", {})
        t = result.get("timings", {})
        return {"ok": True, "dt": dt, "content": c, "pt": u.get("prompt_tokens", 0),
                "ct": u.get("completion_tokens", 0), "prompt_ms": t.get("prompt_ms", 0)}
    except Exception as e:
        err = str(e)
        try: err += " | " + e.read().decode()[:200]
        except: pass
        return {"ok": False, "dt": time.time() - t0, "error": err}

def img_req(b64, prompt="What do you see?", mt=1):
    return req({"model": "qwen3.5", "max_tokens": mt, "temperature": 0,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": prompt}]}]})

def txt_req(prompt, mt=1):
    return req({"model": "qwen3.5", "max_tokens": mt, "temperature": 0,
        "messages": [{"role": "user", "content": prompt}]})

sizes = [
    (128,128),(256,256),(512,512),(768,768),(1024,1024),
    (1536,1536),(2048,2048),(2560,2560),(3072,3072),(4096,4096),
]

print("=== Text baseline (max_tokens=1) ===")
r = txt_req("Say hello.", 1)
print(f"  Time: {r['dt']:.2f}s, tokens: {r.get('pt',0)}" if r["ok"] else f"  Error: {r['error']}")

print(f"\n{'Size':>12} | {'Wall':>7} | {'Prompt':>8} | {'Tok':>5} | {'KB':>7}")
print("-" * 55)
for w, h in sizes:
    img = make_png(w, h); kb = len(img)/1024
    b64 = base64.b64encode(img).decode()
    r = img_req(b64, "What do you see?", 1)
    if r["ok"]:
        pm = r.get("prompt_ms", 0)
        print(f"{w}x{h:>4} | {r['dt']:>6.2f}s | {pm:>7.0f}ms | {r['pt']:>5} | {kb:>6.1f}")
    else:
        print(f"{w}x{h:>4} | {r['dt']:>6.2f}s | {'ERR':>8} | {'':>5} | {kb:>6.1f}")
        print(f"  Error: {r.get('error','')[:80]}")

if "--full" in sys.argv:
    print("\n=== Full response (512x512, max_tokens=256) ===")
    b64 = base64.b64encode(make_png(512, 512)).decode()
    r = img_req(b64, "Describe this image in detail.", 256)
    if r["ok"]:
        print(f"  {r['dt']:.2f}s, {r['pt']}+{r['ct']} tok")
        print(f"  {r['content']}")
