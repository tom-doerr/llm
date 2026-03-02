"""Register Qwen3_5MoeForCausalLM (text-only) in vLLM model registry."""
P = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/registry.py"
with open(P) as f:
    s = f.read()
old = '    "Qwen3_5MoeForConditionalGeneration": (\n        "qwen3_5",\n        "Qwen3_5MoeForConditionalGeneration",\n    ),'
new = ('    "Qwen3_5MoeForCausalLM": ("qwen3_5", "Qwen3_5MoeForCausalLM"),\n'
       + '    ' + old.lstrip())
if 'Qwen3_5MoeForCausalLM' not in s:
    s = s.replace(old, new)
    with open(P, "w") as f:
        f.write(s)
    print("OK: Qwen3_5MoeForCausalLM registered")
