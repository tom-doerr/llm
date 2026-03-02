"""Add qwen35moe to vLLM config GGUF defaults."""
P = "/usr/local/lib/python3.12/dist-packages/vllm/transformers_utils/config.py"
with open(P) as f:
    s = f.read()
old = 'if config.model_type in {"qwen3_moe"}:'
new = 'if config.model_type in {"qwen3_moe", "qwen35moe"}:'
if old in s:
    s = s.replace(old, new)
    with open(P, "w") as f:
        f.write(s)
    print("OK: qwen35moe in vLLM config defaults")
