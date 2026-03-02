"""Fix list/set type mismatch for rope validation."""
import glob
P = "/usr/local/lib/python3.12/dist-packages/vllm/transformers_utils/configs/"
for f in glob.glob(P + "*.py"):
    with open(f) as fh:
        s = fh.read()
    old = '"ignore_keys_at_rope_validation"] = ['
    if old in s:
        s = s.replace(old, old.replace("= [", "= {"))
        s = s.replace("]\n        self.vocab_size", "}\n        self.vocab_size")
        with open(f, "w") as fh:
            fh.write(s)
        print(f"OK: patched {f}")
