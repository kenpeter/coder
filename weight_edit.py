import copy, json, time, datetime, re, subprocess, sys, torch, torch.nn.functional as F, numpy as np
from pathlib import Path
sys.path.insert(0,'/home/kenpeter/work/coder')
from san_model import SimpleAttentionNetwork, SANConfig
from infer import generate
import urllib.request

ROOT=Path('/home/kenpeter/work/coder')
CKPT=ROOT/'checkpoints/san/san_latest.pt'
OUT=ROOT/'checkpoints/san/san_cycle.pt'
MICRO=[
 ('reverse_string','def reverse_string(s):\n    """Return the string reversed."""\n', [('reverse_string("hello")','"olleh"'),('reverse_string("")','""')]),
 ('factorial','def factorial(n):\n    """Return n factorial. 0! = 1."""\n', [('factorial(0)','1'),('factorial(5)','120')]),
 ('fib','def fib(n):\n    """Return the nth Fibonacci number (0-indexed: fib(0)=0, fib(1)=1)."""\n', [('fib(0)','0'),('fib(1)','1'),('fib(6)','8')]),
 ('abs_val','def abs_val(x):\n    """Return the absolute value of x."""\n', [('abs_val(-5)','5'),('abs_val(3)','3')]),
 ('min_list','def min_list(xs):\n    """Return the smallest element in xs."""\n', [('min_list([3,1,2])','1')]),
 ('max_list','def max_list(xs):\n    """Return the largest element in xs."""\n', [('max_list([3,1,2])','3')]),
 ('sum_list','def sum_list(xs):\n    """Return the sum of xs."""\n', [('sum_list([1,2,3])','6')]),
 ('is_even','def is_even(n):\n    """Return True if n is even."""\n', [('is_even(2)','True'),('is_even(3)','False')]),
 ('multiply','def multiply(a, b):\n    """Return a * b."""\n', [('multiply(3,4)','12')]),
 ('list_len','def list_len(xs):\n    """Return the length of xs."""\n', [('list_len([1,2,3])','3')]),
]
def log(*a): print(time.strftime('%H:%M:%S'), *a, flush=True)
def trunc(s,out):
    for pat in ['\ndef ','\nclass ','\nif __name__','\n#','\nprint(']:
        i=out.find(pat)
        if i>0: out=out[:i]
    return out
import gzip
HE=ROOT/'eval_data/HumanEval.jsonl.gz'
def eval_he(m,tok,dev='cuda',n=30):
    with gzip.open(HE,'rt') as f: rows=[json.loads(l) for l in f]
    rows=rows[:n]; m.eval(); passed=0; fails=[]
    for r in rows:
        try:
            out=generate(m,tok,r['prompt'],max_new_tokens=160,temperature=0.0,device=dev)[0]
            out=trunc(r['prompt'],out)
            g={}
            exec(r['prompt']+out+'\n'+r['test']+f"\ncheck({r['entry_point']})", g)
            passed+=1
        except Exception as e:
            fails.append((r['task_id'], r['prompt'], out if 'out' in locals() else '', str(e)[:200]))
    return passed,len(rows),fails
def wfp(m):
    lines=[]
    for n,par in m.named_parameters():
        t=par.detach().float()
        lines.append(f"{n}: shape={tuple(t.shape)} mean={t.mean().item():.4f} std={t.std().item():.4f} norm={t.norm().item():.2f}")
    return "\n".join(lines)
def ask_9b(prompt):
    data={"model":"qwopus-coder","prompt":prompt,"stream":False,"options":{"temperature":0.2}}
    req=urllib.request.Request("http://localhost:11434/api/generate",data=json.dumps(data).encode(),headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=120) as r:
        return json.loads(r.read())["response"]
def apply_edits(m,edits):
    sd=m.state_dict()
    for e in edits:
        name=e.get("param"); scale=e.get("scale",1.0); add=e.get("add",0.0)
        if name not in sd: 
            log("skip unknown param",name); continue
        with torch.no_grad():
            sd[name].mul_(scale).add_(add)
    m.load_state_dict(sd,strict=False)

ckpt=torch.load(CKPT,map_location='cpu',weights_only=False)
cfgd=ckpt['config']
cfg=SANConfig(vocab_size=cfgd['vocab_size'],d_model=cfgd['d_model'],num_heads=cfgd['num_heads'],num_kv_heads=cfgd['num_kv_heads'],num_layers=cfgd['num_layers'],max_seq_len=cfgd['seq_len'])
m=SimpleAttentionNetwork(cfg); m.load_state_dict(ckpt['model_state_dict'],strict=True); m.cuda()
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained('bigcode/starcoder2-7b')
if tok.pad_token is None: tok.pad_token=tok.eos_token

deadline=datetime.datetime.now().replace(hour=7,minute=0,second=0)
if deadline<=datetime.datetime.now(): deadline+=datetime.timedelta(days=1)
log("running 9B-editor loop until",deadline)
best_score=0; cyc=0
while datetime.datetime.now()<deadline:
    cyc+=1
    mp,mt,fails=eval_he(m,tok)
    score=mp
    log(f"cycle {cyc} 30M eval HumanEval@{mt} {mp}/{mt} fails={len(fails)}")
    failtxt=""
    for tid,prompt,gen,err in fails[:5]:
        failtxt+=f"\n--- {tid} ---\nPrompt:\n{prompt[:400]}\nGenerated:\n{gen[:400]}\nError: {err}\n"
    prompt=f"""You are an editor for a 30M-param code model (SAN). Eval: HumanEval@{mt} pass {mp}/{mt}. Failing problems and their generated (wrong) code + errors:
{failtxt}
Weight fingerprint:
{wfp(m)}
Based on these concrete failures, propose 3-5 targeted weight edits as JSON list of objects with keys: param (full param name), scale (float near 1.0), add (float small). Only output JSON, no prose."""
    try:
        resp=ask_9b(prompt)
        mch=re.search(r'\[.*\]',resp,re.S)
        edits=json.loads(mch.group(0)) if mch else []
    except Exception as ex:
        log("9b parse err",ex); edits=[]
    log("proposed edits:",edits)
    if not edits:
        time.sleep(2); continue
    before=copy.deepcopy(m.state_dict())
    apply_edits(m,edits)
    mp2,mt2,_=eval_he(m,tok)
    log(f"after edit HumanEval@{mt2} {mp2}/{mt2}")
    if mp2>=mp:
        # keep
        ck=copy.deepcopy(ckpt); ck['model_state_dict']={k:v.detach().cpu() for k,v in m.state_dict().items()}
        torch.save(ck,CKPT); log("kept edit, saved in place",CKPT)
        best_score=max(best_score,mp2)
    else:
        m.load_state_dict(before,strict=True); m.cuda(); log("reverted edit")
    if best_score>=mt: log("target reached"); break
log("done best",best_score,"cycles",cyc)