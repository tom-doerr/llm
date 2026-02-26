"""Fix GemmaRMSNorm in qwen3_next.py for GGUF loading."""
import importlib.util, pathlib

spec = importlib.util.find_spec("vllm.model_executor.models.qwen3_next")
p = pathlib.Path(spec.origin)
src = p.read_text()

old = "GemmaRMSNorm as Qwen3NextRMSNorm,"
new = "RMSNorm as Qwen3NextRMSNorm,"

assert old in src, f"Pattern not found in {p}"
src = src.replace(old, new)
p.write_text(src)
print(f"Patched {p}: GemmaRMSNorm -> RMSNorm")
