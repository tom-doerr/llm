import builtins
_ri = builtins.__import__
_patched_gguf = False
_patched_tf = False
def _i(n, *a, **k):
    global _patched_gguf, _patched_tf
    m = _ri(n, *a, **k)
    if n == 'torch' and hasattr(m, 'backends'):
        m.backends.cudnn.enabled = False
    if n == 'gguf.constants' and not _patched_gguf:
        _patched_gguf = True
        _patch_gguf()
    if n == 'transformers.integrations.ggml' and not _patched_tf:
        _patched_tf = True
        _patch_tf()
    return m
builtins.__import__ = _i
