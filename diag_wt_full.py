"""Weight type check."""
import sys,os,json,glob,re,gguf
os.environ["CUDA_VISIBLE_DEVICES"]=""
sys.path.insert(0,"/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader")
from qwen35_gguf_map import build
from gguf_loader import get_gguf_weight_type_map
with open(f"{sys.argv[1]}/config.json") as f: cfg=json.load(f)
class F:
    def __init__(s,d): s.__dict__.update(d)
    def get_text_config(s): return s
m=build(F(cfg)); p=sys.argv[2]
d,b=p.rsplit('/',1)
pat=re.sub(r'-\d+-of-(\d+)\.gguf$',r'-*-of-\1.gguf',b)
wt={}
for s in sorted(glob.glob(f"{d}/{pat}")):
    w=get_gguf_weight_type_map(s,m);wt.update(w)
    print(f"{os.path.basename(s)}:{len(w)}")
f=[k for k,v in wt.items() if v in("F32","F16","BF16") and ".weight" in k]
print(f"Total:{len(wt)} F32w:{len(f)}")
