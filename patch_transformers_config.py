"""Patch transformers config: add qwen35moe (use flat Qwen3MoeConfig)."""
P = "/usr/local/lib/python3.12/dist-packages/transformers/models/auto/configuration_auto.py"
with open(P) as f:
    s = f.read()
pairs = [
    ('("qwen3_moe", "Qwen3MoeConfig")', '("qwen35moe", "Qwen3MoeConfig")'),
    ('("qwen3_moe", "Qwen3MoE")', '("qwen35moe", "Qwen3MoE")'),
]
for old, new in pairs:
    if old in s and new not in s:
        s = s.replace(old, new + ',\n        ' + old, 1)
with open(P, "w") as f:
    f.write(s)
print("OK: qwen35moe mapped to Qwen3MoeConfig")
