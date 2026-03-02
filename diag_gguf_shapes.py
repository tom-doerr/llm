"""Check GGUF tensor shapes for expert weights."""
import sys, gguf, torch, numpy as np
f = sys.argv[1]
reader = gguf.GGUFReader(f)
for t in reader.tensors:
    if "blk.0." in t.name:
        data = torch.tensor(t.data)
        print(f"{t.name}: type={t.tensor_type.name} "
              f"shape={list(t.shape)} data_shape={list(data.shape)} "
              f"dtype={data.dtype}")
