"""NaN detection in GDN forward, log to file."""
P="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_next.py"
with open(P) as f: s=f.read()
o="projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)"
n='open("/tmp/gdn.log","a").write(f"GDN_IN l={self.layer_idx} sh={hidden_states.shape} nan={hidden_states.isnan().any().item()} mx={hidden_states.max().item():.3f}\\n")\n        projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)'
s=s.replace(o,n,1)
with open(P,"w") as f: f.write(s)
print("OK: GDN file debug")
