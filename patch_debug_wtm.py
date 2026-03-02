"""Add debug print for weight_type_map in load_model."""
P = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/gguf_loader.py"
with open(P) as f:
    s = f.read()
old = "        # filter out unquantized modules to skip"
new = ('        print(f"DBG wtm:{len(weight_type_map)} map:{len(gguf_weights_map)}")\n'
       '        # filter out unquantized modules to skip')
# Also patch _get_gguf_weight_type
old2 = "            weight_type_map.update(get_gguf_weight_type_map(_sf, gguf_to_hf_name_map))"
new2 = ('            _w = get_gguf_weight_type_map(_sf, gguf_to_hf_name_map)\n'
        '            print(f"DBG shard {_sf[-30:]}: {len(_w)} types")\n'
        '            weight_type_map.update(_w)')
s = s.replace(old2, new2, 1)
s = s.replace(old, new, 1)
with open(P, "w") as f:
    f.write(s)
print("OK: debug print added")
