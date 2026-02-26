"""Fix GemmaRMSNorm double +1 offset for GGUF loading.

GGUF adds +1 to norms. GemmaRMSNorm adds +1 again. Use RMSNorm instead.
"""
import importlib.util, pathlib

spec = importlib.util.find_spec("vllm.model_executor.models.qwen3_5")
p = pathlib.Path(spec.origin)
src = p.read_text()

old = "GemmaRMSNorm as Qwen3_5RMSNorm,"
new = "RMSNorm as Qwen3_5RMSNorm,"

assert old in src, f"Pattern not found in {p}"
src = src.replace(old, new)
p.write_text(src)
print(f"Patched {p}: GemmaRMSNorm -> RMSNorm")
