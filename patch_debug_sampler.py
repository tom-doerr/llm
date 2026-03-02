"""Debug sampler to file."""
P="/usr/local/lib/python3.12/dist-packages/vllm/v1/sample/sampler.py"
with open(P) as f: s=f.read()
o="logits = self.apply_logits_processors("
n='open("/tmp/sam.log","a").write(f"PRE a0={logits.argmax(-1)[0].item()} g={sampling_metadata.all_greedy} sh={logits.shape}\\n")\n        logits = self.apply_logits_processors('
s=s.replace(o,n,1)
o2="sampled = sampled.long()"
n2='sampled = sampled.long()\n        open("/tmp/sam.log","a").write(f"POST s={sampled[:5].tolist()} a={logits.argmax(-1)[:5].tolist()}\\n")'
s=s.replace(o2,n2,1)
with open(P,"w") as f: f.write(s)
print("OK: sampler file debug")
