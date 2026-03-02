"""NaN detection in compute_logits, log to file."""
P="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_5.py"
with open(P) as f: s=f.read()
o="return self.logits_processor(self.lm_head, hidden_states)"
n='import torch as _t\n        _lg=self.logits_processor(self.lm_head, hidden_states)\n        if _lg is not None:\n            _a=_lg.argmax(-1)\n            open("/tmp/lg.log","a").write(f"LG sh={_lg.shape} a0={_a[0].item()} a-1={_a[-1].item()} mx={_lg.max().item():.3f} mn={_lg.min().item():.3f}\\n")\n        return _lg'
s=s.replace(o,n,1)
with open(P,"w") as f: f.write(s)
print("OK: logits file debug")
