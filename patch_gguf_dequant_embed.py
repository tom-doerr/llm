"""Dequantize embedding weights in GGUF iterator."""
P = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/weight_utils.py"
with open(P) as f:
    s = f.read()
if "import gguf.quants" not in s:
    s = "import gguf.quants as _gguf_quants\n" + s
# First pass: skip qweight_type for embeddings
old1 = ('                weight_type_name = name.replace'
        '("weight", "qweight_type")\n'
        '                weight_type = torch.tensor(weight_type)\n'
        '                yield weight_type_name, weight_type')
new1 = ('                if "embed_tokens" not in name'
        ' and not name.startswith("lm_head"):\n'
        '                    weight_type_name = name.replace'
        '("weight", "qweight_type")\n'
        '                    weight_type = torch.tensor(weight_type)\n'
        '                    yield weight_type_name, weight_type')
s = s.replace(old1, new1, 1)
# Second pass: dequantize embeddings, keep .weight name
old2 = ('            if weight_type.name not in ("F32", "BF16", "F16"):\n'
        '                name = name.replace("weight", "qweight")')
new2 = ('            _is_embed = "embed_tokens" in name'
        ' or name.startswith("lm_head")\n'
        '            if weight_type.name not in ("F32", "BF16", "F16"):\n'
        '                if _is_embed:\n'
        '                    weight = _gguf_quants.dequantize'
        '(weight, weight_type)\n'
        '                else:\n'
        '                    name = name.replace("weight", "qweight")')
s = s.replace(old2, new2, 1)
with open(P, "w") as f:
    f.write(s)
print("OK: embedding dequantization patched")
