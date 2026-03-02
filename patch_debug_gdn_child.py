"""Debug GDN child forward to find NaN source."""
P="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_5.py"
with open(P) as f: s=f.read()
# After in_proj_qkvz
o="mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)"
n='mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)\n        open("/tmp/gdn.log","a").write(f"L{self.layer_idx} hs={hidden_states.isnan().any().item()} qkvz={mixed_qkvz.isnan().any().item()}\\n")'
s=s.replace(o,n,1)
# After gdn_attention_core
o="torch.ops.vllm.gdn_attention_core("
n='open("/tmp/gdn.log","a").write(f"L{self.layer_idx} qkv={mixed_qkv.isnan().any().item()} b={b.isnan().any().item()} a={a.isnan().any().item()}\\n")\n        torch.ops.vllm.gdn_attention_core('
s=s.replace(o,n,1)
with open(P,"w") as f: f.write(s)
print("OK: GDN child debug")
