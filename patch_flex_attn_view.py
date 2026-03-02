"""Patch flex_attention.py: use reshape instead of view for non-contiguous KV cache."""
P = "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/flex_attention.py"
with open(P) as f:
    s = f.read()
o = "key_cache = key_cache.view(-1, self.num_kv_heads, self.head_size)"
n = "key_cache = key_cache.reshape(-1, self.num_kv_heads, self.head_size)"
s = s.replace(o, n)
o2 = "value_cache = value_cache.view(-1, self.num_kv_heads, self.head_size)"
n2 = "value_cache = value_cache.reshape(-1, self.num_kv_heads, self.head_size)"
s = s.replace(o2, n2)
with open(P, "w") as f:
    f.write(s)
print("OK: flex_attention view->reshape patched")
