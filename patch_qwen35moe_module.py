"""Create qwen35moe module alias for transformers."""
import os
P = "/usr/local/lib/python3.12/dist-packages/transformers/models"
d = os.path.join(P, "qwen35moe")
os.makedirs(d, exist_ok=True)
init = 'from transformers.models.qwen3_moe import *\n'
with open(os.path.join(d, "__init__.py"), "w") as f:
    f.write(init)
print("OK: qwen35moe module alias created")
