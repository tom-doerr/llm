"""Video description runner."""
import time, base64, json, requests

def run(vpath, url, model, out, prompt):
    print('Encoding video...', flush=True)
    with open(vpath, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()
    print(f'Done ({len(b64)//1024//1024} MB b64)', flush=True)
    c = [{'type': 'video_url',
          'video_url': {'url': f'data:video/mp4;base64,{b64}'}},
         {'type': 'text', 'text': prompt}]
    _stream(url, model, c, out)

def _stream(u,m,c,o):
    print('Sending...',flush=True);s=time.perf_counter()
    eb={'chat_template_kwargs':{'enable_thinking':False}}
    r=requests.post(u,json={'model':m,'stream':True,
      'messages':[{'role':'user','content':c}],
      'max_tokens':16384,'extra_body':eb},stream=True,timeout=3600)
    ft=None;tx='';n=0
    for ln in r.iter_lines():
        t=ln.decode().strip()
        if not t.startswith('data: ') or t=='data: [DONE]':continue
        d=json.loads(t[6:])['choices'][0].get('delta',{})
        ch=d.get('content','')or d.get('reasoning','')
        if ch:
            n+=1;tx+=ch
            if not ft:ft=time.perf_counter()-s;print(f'TTFT:{ft:.2f}s')
    el=time.perf_counter()-s;print(f'Done:{el:.1f}s {n}tok')
    _save(o,ft,el,tx)

def _save(path,ttft,elapsed,text):
    with open(path,'w') as f:
        f.write(f'# TTFT:{ttft:.2f}s Total:{elapsed:.1f}s\n\n')
        f.write(text)
    print(f'Saved:{path}',flush=True)
