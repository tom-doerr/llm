"""Handle tuple shard_ids in MergedColumnParallelLinear for GGUF."""
P = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/linear.py"
_TUPLE_SHARD_FIX = """\
        if isinstance(loaded_shard_id, tuple):
            if getattr(param, "is_gguf_weight_type", False):
                for _s in loaded_shard_id:
                    self.weight_loader(param, loaded_weight, _s)
                return
            elif getattr(param, "is_gguf_weight", False):
                _sz = [self.output_sizes[i] for i in loaded_shard_id]
                _od = getattr(param, "output_dim", 0)
                for _s, _c in zip(loaded_shard_id, loaded_weight.split(_sz, dim=_od)):
                    self.weight_loader(param, _c, _s)
                return
            raise NotImplementedError("tuple shard_id unsupported")"""
with open(P) as f:
    s = f.read()
old = '        if isinstance(loaded_shard_id, tuple):\n'
old += '            raise NotImplementedError(\n'
old += '                "Shard id with multiple indices is not supported in weight_loader, "\n'
old += '                "please use weight_loader_v2 instead."\n'
old += '            )'
new = _TUPLE_SHARD_FIX
s = s.replace(old, new, 1)
with open(P, "w") as f:
    f.write(s)
print("OK: tuple shard_id support added")
