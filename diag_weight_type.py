"""Check weight_type_map from GGUF shards."""
import sys, os, json, gguf
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader")
from qwen35_gguf_map import build

with open(f"{sys.argv[1]}/config.json") as f:
    cfg = json.load(f)
class FC:
    def __init__(s, d): s.__dict__.update(d)
    def get_text_config(s): return s
m = build(FC(cfg))
reader = gguf.GGUFReader(sys.argv[2])
wt = {m[t.name]: t.tensor_type.name for t in reader.tensors if t.name in m}
f32 = {k: v for k, v in wt.items() if v in ("F32","F16","BF16")}
print(f"Types: {len(wt)} total, {len(f32)} F32+")
for k, v in sorted(f32.items())[:15]:
    print(f"  {v}: {k}")
