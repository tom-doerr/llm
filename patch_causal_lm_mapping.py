"""Add qwen35moe to MODEL_FOR_CAUSAL_LM_MAPPING_NAMES."""
P = "/usr/local/lib/python3.12/dist-packages/transformers/models/auto/modeling_auto.py"
with open(P) as f:
    s = f.read()
old = '("qwen3_moe", "Qwen3MoeForCausalLM")'
new = '("qwen35moe", "Qwen3MoeForCausalLM")'
if old in s and new not in s:
    s = s.replace(old, new + ',\n        ' + old)
    with open(P, "w") as f:
        f.write(s)
    print("OK: qwen35moe in CAUSAL_LM mapping")
