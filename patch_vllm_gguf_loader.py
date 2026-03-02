"""Patch vLLM gguf_loader.py for qwen35moe."""
P = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/gguf_loader.py"
with open(P) as f:
    s = f.read()
s = s.replace(
    'if model_type in ("qwen2_moe", "qwen3_moe"):',
    'if model_type in ("qwen2_moe", "qwen3_moe", "qwen35moe", "qwen3_5_moe", "qwen3_5_moe_text"):')
old2 = '        model_type = config.model_type\n'
remap = (old2
    + '        if model_type in ("qwen35moe", "qwen3_5_moe", "qwen3_5_moe_text"):\n'
    + '            model_type = "qwen35moe"\n')
s = s.replace(old2, remap, 1)
# Use text_config for num_hidden_layers (composite VLM configs)
s = s.replace('range(config.num_hidden_layers)', 'range(text_config.num_hidden_layers)')
with open(P, "w") as f:
    f.write(s)
print("OK: patched gguf_loader.py")
