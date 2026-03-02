"""Create text-only config for GGUF from NVFP4 config."""
import json, shutil, os, sys
src = sys.argv[1]  # NVFP4 config dir
dst = sys.argv[2]  # output dir
os.makedirs(dst, exist_ok=True)
with open(f"{src}/config.json") as f:
    c = json.load(f)
tc = c["text_config"]
tc["architectures"] = ["Qwen3_5MoeForCausalLM"]
tc.pop("quantization_config", None)
with open(f"{dst}/config.json", "w") as f:
    json.dump(tc, indent=2, fp=f)
for fn in ["tokenizer.json", "tokenizer_config.json"]:
    p = f"{src}/{fn}"
    if os.path.exists(p):
        shutil.copy2(p, f"{dst}/{fn}")
print(f"OK: text-only config in {dst}")
