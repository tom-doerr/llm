"""Dequant MXFP4: convert to Q8_0, keep qweight path."""
P = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/weight_utils.py"
with open(P) as f:
    s = f.read()
o = ('if _is_embed:\n'
     '                    weight = _gguf_quants.dequantize'
     '(weight, weight_type)\n'
     '                else:')
n = ('_mxfp4 = (weight_type.value == 39)\n'
     '                if _is_embed:\n'
     '                    weight = _gguf_quants.dequantize'
     '(weight, weight_type)\n'
     '                else:\n'
     '                    if _mxfp4:\n'
     '                        weight = _mxfp4_q8'
     '(weight, tensor.shape)')
s = s.replace(o, n, 1)
with open(P, "w") as f:
    f.write(s)
print("OK: MXFP4 -> Q8_0 + qweight")
