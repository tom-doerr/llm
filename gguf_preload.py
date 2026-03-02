"""Pre-load patches for Qwen3.5 MoE GGUF support.
Import this before loading any GGUF model."""
import gguf.constants as gc

E = gc.GGMLQuantizationType
if 39 not in E._value2member_map_:
    m = int.__new__(E, 39)
    m._name_ = 'MXFP4'
    m._value_ = 39
    E._member_map_['MXFP4'] = m
    E._value2member_map_[39] = m
    E._member_names_.append('MXFP4')
if hasattr(gc, 'GGML_QUANT_SIZES'):
    gc.GGML_QUANT_SIZES[E(39)] = (32, 17)
