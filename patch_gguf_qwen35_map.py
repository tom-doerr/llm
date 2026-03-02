"""Inject Qwen3.5 GGUF map into gguf_loader."""
import shutil
shutil.copy2("/tmp/qwen35_gguf_map.py",
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/qwen35_gguf_map.py")
P = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/gguf_loader.py"
with open(P) as f:
    s = f.read()
s = "from vllm.model_executor.model_loader.qwen35_gguf_map import build as _q35_map\n" + s
old = "        config = model_config.hf_config"
new = (old + '\n        mt = config.get_text_config().model_type'
    '\n        if mt in ("qwen3_5_moe_text","qwen35moe","qwen3_5_moe"):'
    '\n            return _q35_map(config)')
s = s.replace(old, new, 1)
with open(P, "w") as f:
    f.write(s)
print("OK: Qwen3.5 GGUF map injected")
