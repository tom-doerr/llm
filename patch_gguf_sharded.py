"""Patch gguf_loader.py to support sharded GGUF files."""
import re
P = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/gguf_loader.py"
with open(P) as f:
    s = f.read()
# Add glob import
if "import glob as _glob" not in s:
    s = "import glob as _glob\n" + s
# Add shard discovery helper before the class
fn = '''
def _gguf_shards(path):
    import re as _re
    m = _re.search(r'-(\d+)-of-(\d+)\.gguf$', path)
    if not m:
        return [path]
    d, base = path.rsplit('/', 1)
    pat = _re.sub(r'-\d+-of-(\d+)\.gguf$', '-*-of-\\\\1.gguf', base)
    return sorted(_glob.glob(f"{d}/{pat}"))
'''
s = s.replace("\nclass GGUFModelLoader", fn + "\nclass GGUFModelLoader")
with open(P, "w") as f:
    f.write(s)
print("OK: sharded GGUF support added")
