"""Fix fix_query_key_value_ordering for flat GGUF layout."""
P="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_next.py"
with open(P) as f: lines=f.readlines()
s=None
for i,l in enumerate(lines):
    if 'def fix_query_key_value_ordering(' in l: s=i;break
assert s,"not found"
ind=len(lines[s])-len(lines[s].lstrip())
e=len(lines)
for i in range(s+2,len(lines)):
    si=len(lines[i])-len(lines[i].lstrip())
    if lines[i].strip() and si<=ind and lines[i].strip().startswith('def '):
        e=i;break
with open('/tmp/_qkv_body.py') as f: body=f.read()
lines[s:e]=[body]
with open(P,'w') as f: f.writelines(lines)
print(f"OK: patched lines {s+1}-{e}")
