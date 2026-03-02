"""Add qwen35moe arch to gguf constants.py."""
P = "/usr/local/lib/python3.12/dist-packages/gguf/constants.py"
with open(P) as f:
    s = f.read()
if 'QWEN35MOE' not in s:
    s = s.replace(
        '    ARCEE            = auto()\n',
        '    ARCEE            = auto()\n    QWEN35MOE        = 200\n')
    s = s.replace(
        '    MODEL_ARCH.QWEN3MOE:         "qwen3moe",\n',
        '    MODEL_ARCH.QWEN3MOE:         "qwen3moe",\n    MODEL_ARCH.QWEN35MOE:        "qwen35moe",\n')
with open(P, "w") as f:
    f.write(s)
print("OK: QWEN35MOE arch added")
