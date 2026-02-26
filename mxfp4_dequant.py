"""MXFP4 dequantization → Q8_0 requantization for GGUF weights."""
import numpy as np

_LUT = np.array(
    [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12],
    dtype=np.int8)


def mxfp4_to_q8_0(data, shape):
    QK = 32
    n = 1
    for d in shape:
        n *= int(d)
    nb = n // QK
    flat = np.asarray(data).reshape(-1)[:nb*17]
    raw = flat.reshape(nb, 17)
    e = raw[:, 0].astype(np.uint32)
    bits = np.where(e < 2,
        np.uint32(0x00200000) << e,
        (e - 1).astype(np.uint32) << np.uint32(23))
    sc = bits.view(np.float32).reshape(nb, 1)
    qs = raw[:, 1:17]
    vals = np.empty((nb, QK), dtype=np.float32)
    vals[:, 0::2] = _LUT[qs & 0x0F].astype(np.float32) * sc
    vals[:, 1::2] = _LUT[qs >> 4].astype(np.float32) * sc
    amax = np.max(np.abs(vals), axis=1, keepdims=True)
    amax = np.where(amax == 0, 1.0, amax)
    d = amax / 127.0
    qi = np.round(vals / d).clip(-128, 127).astype(np.int8)
    d_f16 = d.astype(np.float16)
    out = np.empty(nb * 34, dtype=np.uint8)
    ov = out.reshape(nb, 34)
    ov[:, :2] = d_f16.view(np.uint8).reshape(nb, 2)
    ov[:, 2:] = qi.view(np.uint8)
    rows = int(shape[-1]) if len(shape) > 1 else 1
    return out.reshape(rows, -1)
