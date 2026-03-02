"""Patch load_model to use all shards for weight_type and extra_tensors."""
P = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/gguf_loader.py"
with open(P) as f:
    s = f.read()
old = '        if "lm_head.weight" in get_gguf_extra_tensor_names(\n            local_model_path, gguf_weights_map\n        ):'
new = ('        _extra = set()\n'
       '        for _sf in _gguf_shards(local_model_path):\n'
       '            _extra.update(get_gguf_extra_tensor_names(_sf, gguf_weights_map))\n'
       '        if "lm_head.weight" in _extra:')
s = s.replace(old, new)
with open(P, "w") as f:
    f.write(s)
print("OK: sharded extra_tensors patched")
