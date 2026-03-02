"""Patch gguf constants.py on disk: add MXFP4=39."""
P = "/usr/local/lib/python3.12/dist-packages/gguf/constants.py"
with open(P) as f:
    s = f.read()
old = "    TQ2_0   = 35\n"
new = "    TQ2_0   = 35\n    MXFP4   = 39\n"
assert old in s, "TQ2_0 line not found"
s = s.replace(old, new, 1)
old2 = "GGMLQuantizationType.TQ2_0:   (256, 2 + 64),\n}"
new2 = old2.replace("\n}", "\n    GGMLQuantizationType.MXFP4:   (32, 17),\n}")
s = s.replace(old2, new2, 1)
with open(P, "w") as f:
    f.write(s)
print("OK: MXFP4=39 added to gguf")
