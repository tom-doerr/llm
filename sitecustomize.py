import builtins
_ri = builtins.__import__
def _i(n, *a, **k):
    m = _ri(n, *a, **k)
    if n == 'torch' and hasattr(m, 'backends'):
        m.backends.cudnn.enabled = False
    return m
builtins.__import__ = _i
