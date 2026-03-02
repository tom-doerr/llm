"""Allow bfloat16 for GGUF on Blackwell."""
P="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/gguf.py"
with open(P) as f: s=f.read()
o="return [torch.half, torch.float32]"
n="return [torch.half, torch.bfloat16, torch.float32]"
s=s.replace(o,n,1)
with open(P,"w") as f: f.write(s)
print("OK: GGUF bf16 on Blackwell")
