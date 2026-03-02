"""Diagnostic: check GGUF weight names vs model param names."""
import sys, os, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""
cfg_dir = sys.argv[1]
with open(f"{cfg_dir}/config.json") as f:
    cfg = json.load(f)
sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader")
from qwen35_gguf_map import build

class FakeCfg:
    def __init__(self, d):
        self.__dict__.update(d)
    def get_text_config(self):
        return self

fake = FakeCfg(cfg)
gguf_map = build(fake)
hf_names = set(gguf_map.values())
print(f"Map: {len(hf_names)} HF names")
for prefix in ["layers.0.mlp.experts", "layers.0.linear_attn", "layers.3.self_attn"]:
    names = sorted(n for n in hf_names if prefix in n)
    print(f"\n{prefix}:")
    for n in names:
        print(f"  {n}")
