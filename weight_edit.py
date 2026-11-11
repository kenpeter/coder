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
DBG=ROOT/'weight_edit_debug.log'
def dlog(*a):
    """Full reasoning/feedback dump (prompt + raw editor response + decisions)."""
    with open(DBG,'a') as f:
        f.write(time.strftime('%H:%M:%S')+' '+' '.join(str(x) for x in a)+'\n')
def trunc(s,out):
    for pat in ['\ndef ','\nclass ','\nif __name__','\n#','\nprint(']:
        i=out.find(pat)
        if i>0: out=out[:i]
    return out
import gzip
HE=ROOT/'eval_data/HumanEval.jsonl.gz'
import ast
def _assert_stmts(test_src):
    """Split a HumanEval `check()` test block into individual assert statements.

    Uses ast so multi-line asserts (very common in HumanEval) stay intact —
    naive line-splitting would truncate them and grade the TEST as broken.
    """
    try:
        tree=ast.parse(test_src)
    except Exception:
        return [l.strip() for l in test_src.split('\n') if l.strip().startswith('assert ')]
    out=[]
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            seg=ast.get_source_segment(test_src, node)
            if seg: out.append(seg.strip())
    return out or [test_src.strip()]

def grade(r, completion):
    """Grade ONE HumanEval problem per-test (not just pass/fail).

    Returns (passed: bool, detail: dict) where detail has:
      stage      'syntax' | 'load' | 'tests'
      tests_passed / tests_total   (from the problem's assert block)
      fails      [ {'line', 'assert', 'err'} ]  first few failing assertions
    """
    src = r['prompt'] + completion
    g = {}
    try:
        exec(compile(src, '<gen>', 'exec'), g)
    except Exception as e:
        return False, {'stage':'syntax','error':f"{type(e).__name__}: {e}"[:200],
                       'tests_passed':0,'tests_total':0,'fails':[]}
    cand = g.get(r['entry_point'])
    if cand is None:
        return False, {'stage':'load','error':f"`{r['entry_point']}` not defined in the completion",
                       'tests_passed':0,'tests_total':0,'fails':[]}
    asserts=_assert_stmts(r['test'])
    ok=0; fails=[]
    for i,a in enumerate(asserts):
        try:
            exec(compile(a, '<test>', 'exec'), g)
            ok+=1
        except AssertionError as e:
            fails.append({'line':i+1,'assert':a[:180],'err':'AssertionError'})
        except Exception as e:
            fails.append({'line':i+1,'assert':a[:180],'err':f"{type(e).__name__}: {e}"[:140]})
    return (ok==len(asserts)), {'stage':'tests','tests_passed':ok,'tests_total':len(asserts),'fails':fails[:3]}

def eval_he(m,tok,dev='cuda',n=30,detail=False):
    with gzip.open(HE,'rt') as f: rows=[json.loads(l) for l in f]
    rows=rows[:n]; m.eval(); passed=0; fails=[]; report=[]
    for r in rows:
        try:
            out=generate(m,tok,r['prompt'],max_new_tokens=160,temperature=0.0,device=dev)[0]
        except Exception as e:
            out=''
        out=trunc(r['prompt'],out)
        ok,d=grade(r,out)
        rec={'task_id':r['task_id'],'passed':ok,'prompt':r['prompt'],'generated':out,
             'detail':d,'entry_point':r['entry_point']}
        report.append(rec)
        if ok: passed+=1
        else: fails.append((r['task_id'], r['prompt'], out, d))
    if detail:
        return passed,len(rows),fails,report
    return passed,len(rows),fails
# ─── Stage A: dense score vector ─────────────────────────────────────
# A bare "0/30" never changes, so the editor gets no signal. These four
# components all have headroom to move (see abstraction.md, finding F1).
import glob as _glob
CE_FILES=sorted(_glob.glob('/home/kenpeter/work/data/_shards_final/_shards_code_easy/*.bin'))[:6]
CE_WINS=[(f,o) for f in CE_FILES for o in (1000,9000,40000)]

def eval_ce(m):
    """Cross-entropy on held-out code_easy windows (continuous, cheap)."""
    m.eval(); tot=0.0
    for f,o in CE_WINS:
        a=np.memmap(f,dtype=np.uint16,mode='r')
        if o+2049>len(a): continue
        x=torch.tensor(np.array(a[o:o+2048],dtype=np.int64))[None,:].cuda()
        y=x[:,1:].clone()
        with torch.no_grad(): lg=m(x)[:,:-1,:].float()
        tot+=float(F.cross_entropy(lg.reshape(-1,lg.shape[-1]),y.reshape(-1)))
    return tot/max(len(CE_WINS),1)

def vec(report):
    """Dense score vector from one graded HumanEval report.

    syntax_rate  completions that exec cleanly        (191/210 fail here)
    defined_rate completions that define entry_point   (currently 0)
    assert_rate  individual asserts passed / total     (currently 0/103)
    """
    n=len(report) or 1
    syn=sum(1 for x in report if x['detail'].get('stage')!='syntax')
    dfr=sum(1 for x in report if x['detail'].get('stage')=='tests')
    tp=sum(x['detail'].get('tests_passed',0) for x in report)
    tt=sum(x['detail'].get('tests_total',0) for x in report)
    return {'n':len(report),'passed':sum(1 for x in report if x['passed']),
            'syntax_rate':round(syn/n,4),'defined_rate':round(dfr/n,4),
            'assert_rate':round(tp/tt,4) if tt else 0.0,
            'asserts':f'{tp}/{tt}'}

def fmt_vec(v,ce=None):
    s=(f"passed {v['passed']}/{v['n']} | syntax_rate {v['syntax_rate']:.0%} | "
       f"defined_rate {v['defined_rate']:.0%} | assert_rate {v['assert_rate']:.0%} "
       f"({v['asserts']})")
    if ce is not None: s+=f" | ce {ce:.3f}"
    return s

def wfp(m):
    lines=[]
    for n,par in m.named_parameters():
        t=par.detach().float()
        lines.append(f"{n}: shape={tuple(t.shape)} mean={t.mean().item():.4f} std={t.std().item():.4f} norm={t.norm().item():.2f}")
    return "\n".join(lines)
_GEMMA=None
_GEMMA_TOK=None
def ask_gemma(prompt):
    global _GEMMA,_GEMMA_TOK
    if _GEMMA is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _GEMMA_TOK=AutoTokenizer.from_pretrained("/home/kenpeter/work/coder/gemma4_local")
        _GEMMA=AutoModelForCausalLM.from_pretrained("/home/kenpeter/work/coder/gemma4_local", torch_dtype=torch.bfloat16, device_map="cpu")
        _GEMMA.eval()
    text=_GEMMA_TOK.apply_chat_template([{"role":"user","content":prompt}], tokenize=False, add_generation_prompt=True)
    inputs=_GEMMA_TOK(text, return_tensors="pt")
    with torch.no_grad():
        out=_GEMMA.generate(**inputs, max_new_tokens=512, do_sample=False)
    return _GEMMA_TOK.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
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

deadline=datetime.datetime.now().replace(hour=21,minute=0,second=0)
if deadline<=datetime.datetime.now(): deadline+=datetime.timedelta(days=1)
log("running editor loop until",deadline)
best_score=0; cyc=0; HIST=[]   # Stage A: per-cycle history shown to the editor
while datetime.datetime.now()<deadline:
    cyc+=1
    mp,mt,fails,report=eval_he(m,tok,detail=True)
    score=mp
    v0=vec(report); ce0=eval_ce(m)
    log(f"cycle {cyc} HumanEval@{mt} {mp}/{mt} | {fmt_vec(v0,ce0)}")
    rp=ROOT/f'eval_results/weight_edit_cycle_{cyc}.json'
    json.dump(report, open(rp,'w'), indent=1, default=str)
    dlog(f"graded report -> {rp}")
    dlog(f"SCORE VECTOR before: {fmt_vec(v0,ce0)}")
    failtxt=""
    for tid,prompt,gen,d in fails[:6]:
        failtxt+=(f"\n--- {tid} ---\nPrompt:\n{prompt[:350]}\nGenerated:\n{gen[:350]}\n"
                  f"Grade: stage={d.get('stage')}  tests passed {d.get('tests_passed')}/{d.get('tests_total')}\n")
        if d.get('error'): failtxt+=f"Error: {d['error']}\n"
        for f in d.get('fails',[]):
            failtxt+=f"  FAILING TEST #{f['line']}: {f['assert']}  ->  {f['err']}\n"
    hist=""
    if HIST:
        hist="\nYour previous attempts this session (what you tried, then what happened):\n"
        for h in HIST[-6:]:
            hist+=f"  cycle {h['cyc']}: tried {h['edits']}\n      before -> {h['before']}\n      after  -> {h['after']}\n"
        hist+=("  Do NOT repeat an edit that already produced no gain. Change the "
               "parameter, the direction, or the magnitude.\n")
    prompt=f"""You are an editor for a 30M-param code model (SAN). You edit weights to make it generate BETTER Python.
{hist}
DENSE SCORE VECTOR (the real signal — pass count alone is stuck at {mp}/{mt} and cannot move):
  before this cycle: {fmt_vec(v0,ce0)}
What each component means and why it can move:
  syntax_rate   % of completions that execute without SyntaxError. Currently LOW -> the model writes unclosed parens and brand-new 'def' statements instead of continuing the function body it was given. This is rung 1 of the ladder and the biggest headroom.
  defined_rate  % of completions that actually define the requested function (entry_point). Currently ~0 -> the model ignores the prompt and defines a DIFFERENT function.
  assert_rate   % of individual unit-test asserts passed. Currently 0 -> nothing has ever passed a test.
  ce            cross-entropy on held-out training code. Lower is better; continuous, so it responds to small edits even when pass count does not.
Ladder to climb: syntax_rate -> defined_rate -> assert_rate. Improving ce alone does NOT fix HumanEval.
{failtxt}
Weight fingerprint:
{wfp(m)}
Propose 3-5 targeted weight edits as a JSON list of objects with keys: param (full param name), scale (float near 1.0), add (float small).
Rules:
- Justify each edit from the graded failures above (which stage fails, which assert).
- Prefer params that control whether the model CONTINUES the prompt instead of restarting: attention q/k norms, attn_gate, embedding rows, norm scales.
- Vary direction: if you scaled a param down last cycle and nothing moved, try up.
Only output JSON, no prose."""
    dlog(f"=== cycle {cyc} | eval before: {mp}/{mt} | fails={len(fails)} ===")
    dlog("--- PROMPT SENT TO EDITOR ---"); dlog(prompt); dlog("--- END PROMPT ---")
    try:
        resp=ask_gemma(prompt)
        dlog("--- RAW EDITOR RESPONSE ---"); dlog(resp); dlog("--- END RESPONSE ---")
        mch=re.search(r'\[.*\]',resp,re.S)
        edits=json.loads(mch.group(0)) if mch else []
    except Exception as ex:
        log("editor parse err",ex); dlog("PARSE ERR:",ex); edits=[]
    log("proposed edits:",edits); dlog("PARSED EDITS:",json.dumps(edits))
    if not edits:
        dlog("cycle %d: no valid edits -> skipped (no weights changed)"%cyc)
        time.sleep(2); continue
    before=copy.deepcopy(m.state_dict())
    apply_edits(m,edits)
    mp2,mt2,_,rep2=eval_he(m,tok,detail=True)
    v1=vec(rep2); ce1=eval_ce(m)
    log(f"after edit HumanEval@{mt2} {mp2}/{mt2} | {fmt_vec(v1,ce1)}")
    dlog(f"cycle {cyc}: applied {len(edits)} -> {fmt_vec(v1,ce1)} (was {fmt_vec(v0,ce0)})")
    HIST.append({'cyc':cyc,'edits':edits,'before':fmt_vec(v0,ce0),'after':fmt_vec(v1,ce1)})
    # always keep: edits are applied in place to san_latest.pt regardless of
    # whether the score improved (single-checkpoint policy, no base copies).
    ck=copy.deepcopy(ckpt); ck['model_state_dict']={k:v.detach().cpu() for k,v in m.state_dict().items()}
    torch.save(ck,CKPT); log("kept edit, saved in place",CKPT)
    best_score=max(best_score,mp2); dlog(f"cycle {cyc}: DECISION=KEEP ({mp}->{mp2}, saved to {CKPT})")
    if best_score>=mt: log("target reached"); break
log("done best",best_score,"cycles",cyc)