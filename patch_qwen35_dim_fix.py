"""Fix 1D→2D dimension mismatch in Qwen3_5 load_weights."""
P = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_5.py"
with open(P) as f:
    s = f.read()
old = '                    weight_loader(param, loaded_weight)'
new = ('                    _ig = getattr(param, "is_gguf_weight", False)\n'
       '                    if not _ig and loaded_weight.shape != param.shape'
       ' and loaded_weight.numel() == param.numel():\n'
       '                        loaded_weight = loaded_weight.reshape(param.shape)\n'
       '                    weight_loader(param, loaded_weight)')
s = s.replace(old, new, 1)
with open(P, "w") as f:
    f.write(s)
print("OK: dim mismatch fix added")
