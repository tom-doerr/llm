"""Patch _get_gguf_weight_type to use all shards."""
P = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/gguf_loader.py"
with open(P) as f:
    s = f.read()
old = "        weight_type_map = get_gguf_weight_type_map(\n            model_name_or_path, gguf_to_hf_name_map\n        )"
new = ("        weight_type_map = {}\n"
       "        for _sf in _gguf_shards(model_name_or_path):\n"
       "            weight_type_map.update(get_gguf_weight_type_map(_sf, gguf_to_hf_name_map))")
s = s.replace(old, new, 1)
with open(P, "w") as f:
    f.write(s)
print("OK: sharded weight_type patched")
