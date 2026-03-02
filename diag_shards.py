"""Check if _gguf_shards works with actual GGUF path."""
import sys, glob, re
path = sys.argv[1]
m = re.search(r'-(\d+)-of-(\d+)\.gguf$', path)
print(f"Path: {path}")
print(f"Regex match: {m is not None}")
if m:
    d, base = path.rsplit('/', 1)
    pat = re.sub(r'-\d+-of-(\d+)\.gguf$', r'-*-of-\1.gguf', base)
    full = f"{d}/{pat}"
    print(f"Glob pattern: {full}")
    shards = sorted(glob.glob(full))
    print(f"Found {len(shards)} shards:")
    for s in shards:
        print(f"  {s}")
