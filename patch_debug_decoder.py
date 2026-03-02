"""Debug decoder layer forward for NaN in full attention."""
P="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_next.py"
with open(P) as f: s=f.read()
o='hidden_states = self_attention_output'
n='hidden_states = self_attention_output\n        if self.layer_type=="full_attention":\n            open("/tmp/fa.log","a").write(f"FA L{self.layer_idx} attn_nan={hidden_states.isnan().any().item()} mx={hidden_states.abs().max().item():.1f}\\n")'
s=s.replace(o,n,1)
with open(P,"w") as f: f.write(s)
print("OK: decoder debug")
