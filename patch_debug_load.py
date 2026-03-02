"""Add debug logging to Qwen3_5Model.load_weights."""
P = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_5.py"
with open(P) as f:
    s = f.read()
old = "        for name, loaded_weight in weights:"
new = ("        _total = _loaded = 0\n"
       "        _skipped = []\n"
       "        for name, loaded_weight in weights:\n"
       "            _total += 1")
s = s.replace(old, new, 1)
with open(P, "w") as f:
    f.write(s)
print("OK: debug counter added")
