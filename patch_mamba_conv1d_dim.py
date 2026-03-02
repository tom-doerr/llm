"""Fix 2D->3D conv1d weight dim in mamba_mixer2 weight_loader."""
P = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/mamba_mixer2.py"
with open(P) as f:
    s = f.read()
old = ('    def loader(param: torch.Tensor, loaded_weight:'
       ' torch.Tensor) -> None:\n'
       '        # - track boundary')
new = ('    def loader(param: torch.Tensor, loaded_weight:'
       ' torch.Tensor) -> None:\n'
       '        if loaded_weight.dim() == 2 and param.dim() == 3:\n'
       '            loaded_weight = loaded_weight.unsqueeze(1)\n'
       '        # - track boundary')
s = s.replace(old, new, 1)
with open(P, "w") as f:
    f.write(s)
print("OK: conv1d 2D->3D dim fix added")
