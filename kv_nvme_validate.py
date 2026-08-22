#!/usr/bin/env python3
"""Validate the KV-cache-on-NVMe deployment (KV_NVME=1 deploy of Qwen3.8).

Sends the SAME long prompt (> 2 hybrid attention blocks, i.e. >> 784 tokens)
several times at temperature 0 and reports, per round: TTFT, completion text
equality vs round 1, and the vLLM prefix-cache / offloading counters from
/metrics. Run it once, then `docker restart vllm_node`, then run it again with
--expect-restart-hit: if the fs tier works, round 1 after the restart already
has a low TTFT and the text is identical to the pre-restart baseline saved in
--state. Any text divergence at temperature 0 = the connector is corrupting
hits (the upstream bug class #4701/#4674) -> do NOT keep KV_NVME on.
"""
import argparse, json, os, sys, time, urllib.request

def metrics(base):
    out = {}
    for line in urllib.request.urlopen(base + "/metrics", timeout=30).read().decode().splitlines():
        if line.startswith("#"): continue
        if any(k in line for k in ("prefix_cache", "offload", "external", "kv_cache_usage", "connector")):
            name, _, val = line.rpartition(" ")
            key = name.split("{")[0]
            try: out[key] = out.get(key, 0.0) + float(val)
            except ValueError: pass
    return out

def chat(base, model, prompt, max_tokens):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": max_tokens, "seed": 1,
            "chat_template_kwargs": {"enable_thinking": False}, "stream": True}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    t0 = time.time(); ttft = None; text = []
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            raw = raw.decode().strip()
            if not raw.startswith("data:") or raw == "data: [DONE]": continue
            d = json.loads(raw[5:])
            delta = d["choices"][0].get("delta", {}).get("content") or ""
            if delta and ttft is None: ttft = time.time() - t0
            text.append(delta)
    return ttft, time.time() - t0, "".join(text)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("VLLM_BASE_URL", "http://192.168.110.2:8000"))
    ap.add_argument("--tokens", type=int, default=6000, help="approx prompt length in tokens")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--state", default=os.path.expanduser("~/llm/benchmark_results/kv_nvme_state.json"))
    ap.add_argument("--expect-restart-hit", action="store_true")
    a = ap.parse_args()
    model = json.load(urllib.request.urlopen(a.base + "/v1/models", timeout=30))["data"][0]["id"]
    # Deterministic, incompressible-ish filler: numbered pseudo-random facts.
    import random; rnd = random.Random(42)
    words = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november".split()
    filler = " ".join(f"[{i}] {' '.join(rnd.choice(words) for _ in range(7))}." for i in range(a.tokens // 9))
    prompt = f"Here is a numbered list of notes.\n\n{filler}\n\nReply with exactly the text of note [17] and nothing else."
    base_text = None
    if os.path.exists(a.state):
        base_text = json.load(open(a.state)).get("text")
    m0 = metrics(a.base)
    results = []
    for i in range(a.rounds):
        ttft, total, text = chat(a.base, model, prompt, a.max_tokens)
        m1 = metrics(a.base)
        delta = {k: round(m1.get(k, 0) - m0.get(k, 0), 1) for k in m1 if m1.get(k, 0) != m0.get(k, 0)}
        m0 = m1
        same = (base_text is None) if i == 0 and base_text is None else (text == (base_text if base_text else results[0]["text"]))
        if base_text is None and i == 0: base_text = text; same = True
        results.append({"round": i + 1, "ttft_s": round(ttft or -1, 3), "total_s": round(total, 2),
                        "same_text": bool(same), "text": text, "metric_delta": delta})
        print(f"round {i+1}: TTFT {ttft:.3f}s total {total:.2f}s same_text={same} metrics_delta={delta}")
    if not os.path.exists(a.state):
        os.makedirs(os.path.dirname(a.state), exist_ok=True)
        json.dump({"model": model, "text": base_text, "prompt_tokens_approx": a.tokens}, open(a.state, "w"))
        print(f"baseline saved to {a.state}")
    if a.expect_restart_hit:
        r1 = results[0]
        ok = r1["same_text"] and r1["ttft_s"] < results[-1]["ttft_s"] * 3
        print("RESTART-HIT CHECK:", "PASS" if ok else "FAIL", f"(round1 TTFT {r1['ttft_s']}s, same_text={r1['same_text']})")
        sys.exit(0 if ok else 1)
    if not all(r["same_text"] for r in results):
        print("TEXT DIVERGENCE at temperature 0 -> cached vs fresh differ"); sys.exit(1)

if __name__ == "__main__":
    main()
