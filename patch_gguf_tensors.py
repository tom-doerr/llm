"""Add qwen35moe tensor map (copy of qwen3moe)."""
P = "/usr/local/lib/python3.12/dist-packages/gguf/constants.py"
with open(P) as f:
    s = f.read()
old = "    MODEL_ARCH.QWEN3MOE: [\n"
end = s.find("],\n", s.find(old)) + 3
block = s[s.find(old):end]
new_block = block.replace("QWEN3MOE", "QWEN35MOE")
if "QWEN35MOE: [" not in s:
    s = s[:end] + "\n" + new_block + s[end:]
    with open(P, "w") as f:
        f.write(s)
    print("OK: QWEN35MOE tensor map added")
