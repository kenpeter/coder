import copy, json, random, time, gzip, sys, re
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoTokenizer
sys.path.insert(0,'/home/kenpeter/work/coder')
from san_model import SimpleAttentionNetwork, SANConfig
from infer import generate

ROOT=Path('/home/kenpeter/work/coder')
CKPT=ROOT/'checkpoints/san/san_latest.pt'
OUT=ROOT/'checkpoints/san/san_cycle.pt'
DATA=Path('/home/kenpeter/work/data/_shards_final')
HE=ROOT/'eval_data/HumanEval.jsonl.gz'

def log(*a): print(time.strftime('%H:%M:%S'), *a, flush=True)

def load_model(path, device='cuda'):
    ckpt=torch.load(path,map_location='cpu',weights_only=False)
    cfgd=ckpt['config']
    cfg=SANConfig(vocab_size=cfgd['vocab_size'],d_model=cfgd['d_model'],
        num_heads=cfgd['num_heads'],num_kv_heads=cfgd['num_kv_heads'],
        num_layers=cfgd['num_layers'],max_seq_len=cfgd['seq_len'])
    m=SimpleAttentionNetwork(cfg); m.load_state_dict(ckpt['model_state_dict'],strict=True)
    return ckpt,cfg,m.to(device)

def trunc_completion(prompt, out):
    # keep completion up to next top-level def/class or end
    s = out
    # stop at "\ndef " or "\nclass " that appears after some chars
    for pat in ['\ndef ','\nclass ','\nif __name__','\n#','\nprint(']:
        i = s.find(pat)
        if i > 0: s = s[:i]
    return s

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

def eval_micro(m,tok,device='cuda'):
    m.eval(); passed=0
    for name,prompt,cases in MICRO:
        try:
            out=generate(m,tok,prompt,max_new_tokens=96,temperature=0.0,device=device)[0]
            out=trunc_completion(prompt,out)
            g={}; exec(prompt+out,g)
            ok=all(str(eval(expr,g))==ans for expr,ans in cases)
        except Exception: ok=False
        if ok: passed+=1
    return passed, len(MICRO)

def eval_humaneval(m,tok,device='cuda',n=30):
    with gzip.open(HE,'rt') as f: rows=[json.loads(l) for l in f]
    rows=rows[:n]; m.eval(); passed=0
    for r in rows:
        try:
            out=generate(m,tok,r['prompt'],max_new_tokens=160,temperature=0.0,device=device)[0]
            out=trunc_completion(r['prompt'],out)
            g={}
            exec(r['prompt']+out+'\n'+r['test']+f"\ncheck({r['entry_point']})", g)
            passed+=1
        except Exception: pass
    return passed,len(rows)

bins=[]
for tier in ['_shards_code_easy','_shards_code_medium','_shards_code_hard','_shards_code_gold']:
    d=DATA/tier
    if d.exists(): bins+=sorted(d.glob('*.bin'))

def sample_batch(seq_len,bs=2):
    xs,ys=[],[]
    while len(xs)<bs:
        b=random.choice(bins); data=np.memmap(b,dtype=np.uint16,mode='r')
        if len(data)<seq_len+2: continue
        i=random.randint(0,len(data)-seq_len-2)
        ch=np.array(data[i:i+seq_len+1],dtype=np.int64)
        xs.append(torch.from_numpy(ch[:-1])); ys.append(torch.from_numpy(ch[1:]))
    return torch.stack(xs).cuda(), torch.stack(ys).cuda()

def edit_cycle(model, base_ckpt, seq_len, steps, lr):
    model.train()
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0.01)
    losses=[]
    for s in range(steps):
        x,y=sample_batch(seq_len,2)
        logits=model(x)
        if isinstance(logits,tuple): logits=logits[0]
        loss=F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]), y.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        losses.append(loss.item())
        if (s+1)%100==0: log(f'  train {s+1}/{steps} loss {sum(losses[-100:])/100:.4f}')
    ck=copy.deepcopy(base_ckpt)
    ck['model_state_dict']={k:v.detach().cpu() for k,v in model.state_dict().items()}
    torch.save(ck, OUT)
    return sum(losses[-100:])/min(100,len(losses))

base,cfg,model=load_model(CKPT)
tok=AutoTokenizer.from_pretrained('bigcode/starcoder2-7b')
if tok.pad_token is None: tok.pad_token=tok.eos_token
log('start cycles; bins',len(bins))
import datetime as _dt
_now=_dt.datetime.now()
_deadline=_dt.datetime(_now.year,_now.month,_now.day,20,0,0)
if _deadline<=_now: _deadline+=_dt.timedelta(days=1)
log('running until 20:00 deadline', _deadline)
best=(0,0); cyc=0
while _dt.datetime.now()<_deadline:
    cyc+=1
    mp,mt=eval_micro(model,tok); hp,ht=eval_humaneval(model,tok,n=30)
    if (mp,hp)>best: best=(mp,hp)
    log(f'cycle {cyc} micro {mp}/{mt} he@30 {hp}/{ht} best={best}')
    if hp>=8 or mp>=8: log('target reached'); break
    lr = 4e-6 if cyc<=8 else 2e-6
    avg = edit_cycle(model, base, cfg.max_seq_len, steps=300, lr=lr)
    log(f'cycle {cyc} edit done loss {avg:.4f}')
log('done best', best, 'cycles', cyc)
