"""Patch transformers ggml.py: add qwen35moe arch."""
import re
P = "/usr/local/lib/python3.12/dist-packages/transformers/integrations/ggml.py"
with open(P) as f:
    s = f.read()
blocks = re.findall(r'"qwen3_moe": (\{[^}]+\})', s)
assert blocks, "qwen3_moe block not found"
s = s.replace('"qwen3_moe": {',
    '"qwen35moe": ' + blocks[0] + ',\n    "qwen3_moe": {')
with open(P, "w") as f:
    f.write(s)
print("OK: qwen35moe added to ggml.py")
