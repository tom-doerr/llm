"""Patch kv_cache_utils.py: fix page size unification for hybrid models."""
import math, re
P = "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py"
with open(P) as f:
    s = f.read()
if "import math" not in s:
    s = "import math\n" + s
# Find and replace the function
pat = r'def unify_kv_cache_spec_page_size\(.*?\n    return new_kv_cache_spec'
m = re.search(pat, s, re.DOTALL)
assert m, "Could not find unify_kv_cache_spec_page_size"
NEW = '''def unify_kv_cache_spec_page_size(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> dict[str, KVCacheSpec]:
    """Unify page sizes for hybrid models with padding."""
    page_sizes = {l.page_size_bytes for l in kv_cache_spec.values()}
    if len(page_sizes) <= 1:
        return kv_cache_spec
    # Debug: log unique page sizes
    _ps = {}
    for n, sp in kv_cache_spec.items():
        ps = sp.page_size_bytes
        _ps.setdefault(ps, []).append(type(sp).__name__)
    for ps, ts in sorted(_ps.items()):
        print(f"DBG page_size={ps} cnt={len(ts)} type={ts[0]}")
    # Compute target divisible by all page sizes
    srt = sorted(page_sizes)
    target = srt[-1]
    for ps in srt[:-1]:
        if target % ps != 0:
            target = math.ceil(target / ps) * ps
    if target != srt[-1]:
        print(f"DBG padded: {srt[-1]} -> {target}")
    new_kv = {}
    for ln, ls in kv_cache_spec.items():
        ps = ls.page_size_bytes
        if ps == target:
            new_kv[ln] = ls
            continue
        if target % ps == 0:
            r = target // ps
            ns = replace(ls, block_size=ls.block_size * r)
            if ns.page_size_bytes == target:
                new_kv[ln] = ns
                continue
        if hasattr(ls, 'page_size_padded'):
            ns = replace(ls, page_size_padded=target)
            if ns.page_size_bytes == target:
                new_kv[ln] = ns
                continue
        raise NotImplementedError(
            f"Cannot unify {ln}: ps={ps} target={target}")
    return new_kv'''
s = s[:m.start()] + NEW + s[m.end():]
with open(P, "w") as f:
    f.write(s)
print("OK: kv_cache page size unification patched")
