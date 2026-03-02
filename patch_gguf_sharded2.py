"""Patch gguf_loader to iterate all shard files."""
P = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/gguf_loader.py"
with open(P) as f:
    s = f.read()
old = "        yield from gguf_quant_weights_iterator(model_name_or_path, gguf_to_hf_name_map)"
new = ("        for _sf in _gguf_shards(model_name_or_path):\n"
       "            yield from gguf_quant_weights_iterator(_sf, gguf_to_hf_name_map)")
s = s.replace(old, new)
with open(P, "w") as f:
    f.write(s)
print("OK: sharded weight iteration patched")
