"""Dequant MXFP4: import + yield Q8_0 type for MXFP4."""
P="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/weight_utils.py"
with open(P) as f:
    s = f.read()
if "mxfp4_dequant" not in s:
    s = "from mxfp4_dequant import mxfp4_to_q8_0 as _mxfp4_q8\n" + s
o = 'weight_type = torch.tensor(weight_type)'
n = 'weight_type = torch.tensor(8 if weight_type==39 else weight_type)'
s = s.replace(o, n, 1)
with open(P, "w") as f:
    f.write(s)
print("OK: MXFP4 import + Q8_0 type")
