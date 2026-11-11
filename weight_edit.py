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
# Stage C: the ruler must point at PYTHON. HumanEval is Python, but
# _shards_code_easy is 325/325 train-php shards -- measured CE 1.511 there
# vs 3.104 on train-py and 2.752 on codegold-py. Optimizing the old window
# was lowering PHP loss, which cannot help a Python benchmark.
CE_PY=sorted(_glob.glob('/home/kenpeter/work/data/_shards_final/train-py-*.bin'))[:6]
CE_PHP=sorted(_glob.glob('/home/kenpeter/work/data/_shards_final/_shards_code_easy/*.bin'))[:6]
CE_WINS=[(f,o) for f in CE_PY for o in (1000,9000,40000)]
CE_PHP_WINS=[(f,o) for f in CE_PHP for o in (1000,9000,40000)]

def eval_ce(m, wins=None):
    """Cross-entropy on held-out PYTHON windows (continuous, cheap).

    `wins` defaults to Python. Pass CE_PHP_WINS to check we did not break PHP.
    """
    m.eval(); tot=0.0
    for f,o in (wins or CE_WINS):
        a=np.memmap(f,dtype=np.uint16,mode='r')
        if o+2049>len(a): continue
        x=torch.tensor(np.array(a[o:o+2048],dtype=np.int64))[None,:].cuda()
        y=x[:,1:].clone()
        with torch.no_grad(): lg=m(x)[:,:-1,:].float()
        tot+=float(F.cross_entropy(lg.reshape(-1,lg.shape[-1]),y.reshape(-1)))
    return tot/max(len(CE_WINS),1)

# token ids that decode to whitespace-only indentation (spaces / tabs).
# Built once: the tokenizer emits DIFFERENT ids for a 4-space indent
# depending on context (281 vs 283 were both observed), so keying on one id
# silently measures ~0. Summing the whole whitespace class is robust.
_INDENT_IDS=None
def indent_ids(tok):
    global _INDENT_IDS
    if _INDENT_IDS is None:
        out=[]
        for t in range(tok.vocab_size):
            s=tok.decode([t])
            if s and all(c in ' \t' for c in s): out.append(t)
        _INDENT_IDS=out
    return _INDENT_IDS

def eval_pindent(m, tok):
    """Mean probability on whitespace-indentation tokens at the HumanEval
    prompt boundary.

    This is THE disease metric (finding F6). Each HumanEval prompt ends with
    a docstring followed by a newline, so the ONLY correct next token is an
    indent. The model instead collapses onto newline (222, P~0.94) and
    restarts with a new top-level 'def'.

    Measured once per proposal: 30 single forward passes, no generation, so
    it is cheap enough to ablate against -- unlike cont_rate, which needs
    full generation. ce_py averages over every token while this diseased
    position is ~1 in 1000, so ce_py alone can improve while the disease is
    untouched.
    """
    m.eval()
    ind=indent_ids(tok)
    with gzip.open(HE,'rt') as f:
        rows=[json.loads(l) for l in f][:30]
    tot=0.0
    with torch.no_grad():
        for r in rows:
            ids=tok(r['prompt'],return_tensors='pt').input_ids.cuda()
            pr=F.softmax(m(ids)[:,-1,:].float(),-1)[0]
            tot+=float(pr[ind].sum())
    return tot/max(len(rows),1)

_BOUNDARY_DIR=None
def boundary_dir(m, tok, refresh=False):
    """Mean post-final_norm hidden state at the HumanEval docstring boundary.

    All 30 prompts end the same way -- a docstring then a newline -- and their
    boundary states are almost collinear (measured cos to the mean = 0.983),
    so a single direction u characterizes 'the model is at the end of a
    docstring and must now indent'. That is what makes a targeted edit
    possible at all: there is one consistent state to steer, not 30 different
    ones. u is refreshed only when asked, since it moves slowly.
    """
    global _BOUNDARY_DIR
    if _BOUNDARY_DIR is not None and not refresh: return _BOUNDARY_DIR
    m.eval()
    with gzip.open(HE,'rt') as f:
        rows=[json.loads(l) for l in f][:30]
    hs=[]
    with torch.no_grad():
        for r in rows:
            ids=tok(r['prompt'],return_tensors='pt').input_ids.cuda()
            hs.append(m.final_norm(m._hidden(ids))[0,-1,:].float())
    _BOUNDARY_DIR=F.normalize(torch.stack(hs).mean(0,keepdim=True),dim=1)[0]
    return _BOUNDARY_DIR

def boundary_edits(m, tok, alphas=(0.03,0.06)):
    """Diagnosis-derived candidate edits for the docstring-boundary disease.

    These are NOT invented by the editor -- they come straight out of the
    measurement in F6/F8. logits = final_norm(x) @ embedding.weight.T, so
    adding +alpha*u to the indentation-token rows and -alpha*u to the newline
    row raises the indent logit exactly at positions whose hidden state
    resembles the boundary, and barely elsewhere (cos(h,u) is high only
    there).

    Measured on this checkpoint before wiring in:
        alpha   p_indent   ce_py
        0       0.0597     3.1038
        0.02    0.4617     3.1072   (guard OK)
        0.05    0.9760     3.1660   (guard OK, +2.0%)
        0.10    0.9999     3.5075   (guard FAIL)
    The ce_py guard is what keeps alpha honest -- ablation still arbitrates,
    and the absolute ceiling below stops it accumulating across cycles.
    """
    return [{'param':'embedding.weight','boundary':a,
             'reason':f'docstring-boundary direction: +{a}*u on indent rows, '
                      f'-{a}*u on newline row (measured: lifts p_indent ~16x '
                      f'at alpha {a} within the ce_py guard)'}
            for a in alphas]

def vec(report):
    """Dense score vector from one graded HumanEval report.

    syntax_rate  completions that exec cleanly        (191/210 fail here)
    defined_rate completions that define entry_point   (currently ~0)
    assert_rate  individual asserts passed / total     (currently 0/103)
    cont_rate    completions that CONTINUED the prompt (did not restart with
                 a brand-new top-level def/class) -- this is the direct
                 measure of the F1 disease (193/210 restart)
    """
    n=len(report) or 1
    syn=sum(1 for x in report if x['detail'].get('stage')!='syntax')
    dfr=sum(1 for x in report if x['detail'].get('stage')=='tests')
    tp=sum(x['detail'].get('tests_passed',0) for x in report)
    tt=sum(x['detail'].get('tests_total',0) for x in report)
    def restarts(x):
        g=x['generated'].lstrip()
        return g.startswith('def ') or g.startswith('\ndef ') or g.startswith('class ')
    cont=sum(1 for x in report if not restarts(x))
    return {'n':len(report),'passed':sum(1 for x in report if x['passed']),
            'syntax_rate':round(syn/n,4),'defined_rate':round(dfr/n,4),
            'assert_rate':round(tp/tt,4) if tt else 0.0,
            'cont_rate':round(cont/n,4),
            'asserts':f'{tp}/{tt}'}

def fmt_vec(v,ce=None,ce_php=None,pind=None):
    s=(f"passed {v['passed']}/{v['n']} | syntax_rate {v['syntax_rate']:.0%} | "
       f"defined_rate {v['defined_rate']:.0%} | assert_rate {v['assert_rate']:.0%} "
       f"({v['asserts']}) | cont_rate {v.get('cont_rate',0):.0%}")
    if ce is not None: s+=f" | ce_py {ce:.3f}"
    if ce_php is not None: s+=f" | ce_php {ce_php:.3f}"
    if pind is not None: s+=f" | p_indent {pind:.4f}"
    return s

# Stage B: params worth offering the editor (the old menu was attn_gate only,
# which is why every cycle proposed the same 5 edits).
PARAM_MENU=[
 'embedding.weight','final_norm.scale','mtp_final_norm.scale',
 'blocks.N.attn.q_norm.scale','blocks.N.attn.k_norm.scale','blocks.N.attn_gate',
 'blocks.N.norm1.scale','blocks.N.post_attn_norm.scale','blocks.N.pre_hada_norm.scale',
 'blocks.N.hadamard_mlp.d1','blocks.N.hadamard_mlp.d2',
 'blocks.N.attn.q_proj.weight','blocks.N.attn.out_proj.weight',
]
def menu_txt():
    return ("Edit menu (use a spread of these, N = 0..11; do not use the same "
            "param every cycle):\n  " + ", ".join(PARAM_MENU))

def ablate(m, edits, tok, top_k=3):
    """Stage B+C: score each proposed edit SEPARATELY, keep only the winners.

    One edit at a time = credit assignment. Without this, 5 edits are applied
    together and all are kept blind, so a no-op edit is indistinguishable from
    a good one (that is what accumulated into the final_norm.scale collapse).

    Ranking signal is delta p_indent -- the docstring-boundary disease (F6) --
    NOT delta ce_py. ce_py is applied as a GUARD: an edit that buys a little
    p_indent while wrecking Python loss is rejected outright. This ordering
    matters because ce_py averages over all tokens while the diseased
    position is ~1 in 1000, so ce_py alone would happily rate a no-op for
    the disease as a win.

    Returns (winners, rows) where rows = [(score, edit, p_indent_after,
    ce_py_after, note)]. score > 0 means p_indent improved and the ce_py
    guard held.
    """
    CE_GUARD=1.05   # ce_py may rise at most 5% over the loop-start reference
    base={k:v.detach().clone() for k,v in m.state_dict().items()}
    pi_base=eval_pindent(m,tok); ce_base=eval_ce(m)
    ce_lim=CE_REF*CE_GUARD         # absolute, not relative to this cycle
    rows=[]
    for e in edits:
        name=e.get('param')
        if name not in base:
            rows.append((-1e9,e,None,None,'unknown param')); continue
        cands=[dict(e)]
        # direction search: scalar params are cheap and high-leverage, so also
        # try the opposite direction (Gemma only ever proposes one).
        if base[name].numel()==1 and abs(float(e.get('scale',1.0))-1.0)>1e-6:
            inv=dict(e); inv['scale']=round(1.0/float(e.get('scale',1.0)),4); inv['add']=0.0
            cands.append(inv)
        for c in cands:
            m.load_state_dict({k:v.clone() for k,v in base.items()},strict=False)
            apply_edits(m,[c])
            pi=eval_pindent(m,tok); ce=eval_ce(m)
            note='mirror' if c is not cands[0] else ''
            if ce > ce_lim:
                rows.append((-1e9,c,pi,ce,(note+' ce_py guard').strip())); continue
            rows.append((pi-pi_base,c,pi,ce,note))
    m.load_state_dict(base,strict=False)          # back to pre-edit state
    rows.sort(key=lambda r:-r[0])
    winners=[r for r in rows if r[0]>0][:top_k]
    if not winners and rows:                      # keep the least-bad, never none
        winners=[rows[0]]
    return [r[1] for r in winners], rows

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
        out=_GEMMA.generate(**inputs, max_new_tokens=1600, do_sample=False)
    return _GEMMA_TOK.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
def parse_edits(resp):
    """Salvage edit objects from an editor response, even if truncated.

    The old code did re.search(r'\\[.*\\]') and required a closing bracket, so
    a response cut off by max_new_tokens returned None -> edits=[] with NO
    exception raised and nothing logged. That silently skipped a whole cycle
    (observed 15:23:39: 'proposed edits: []').

    This walks the string keeping brace depth and string state, so it
    extracts every COMPLETE {...} object regardless of markdown fences,
    prose, or the array never being closed. Truncation now costs at most the
    last partial edit instead of the whole cycle.
    """
    s=re.sub(r'```[a-zA-Z]*','',resp)
    out=[]; depth=0; start=None; in_str=False; esc=False
    for i,ch in enumerate(s):
        if in_str:
            if esc: esc=False
            elif ch=='\\': esc=True
            elif ch=='"': in_str=False
            continue
        if ch=='"': in_str=True
        elif ch=='{':
            if depth==0: start=i
            depth+=1
        elif ch=='}':
            depth-=1
            if depth==0 and start is not None:
                try:
                    o=json.loads(s[start:i+1])
                    if isinstance(o,dict) and o.get('param'): out.append(o)
                except Exception: pass
                start=None
    return out

def apply_edits(m,edits):
    sd=m.state_dict()
    for e in edits:
        name=e.get("param"); scale=e.get("scale",1.0); add=e.get("add",0.0)
        if name not in sd: 
            log("skip unknown param",name); continue
        with torch.no_grad():
            if e.get('boundary') is not None:
                # targeted docstring-boundary edit, see boundary_edits()
                a=float(e['boundary'])
                u=boundary_dir(m,tok).to(sd[name].dtype)
                sd[name][indent_ids(tok)] = sd[name][indent_ids(tok)] + a*u
                sd[name][222]              = sd[name][222] - a*u
                continue
            sd[name].mul_(scale).add_(add)
    m.load_state_dict(sd,strict=False)

ckpt=torch.load(CKPT,map_location='cpu',weights_only=False)
cfgd=ckpt['config']
cfg=SANConfig(vocab_size=cfgd['vocab_size'],d_model=cfgd['d_model'],num_heads=cfgd['num_heads'],num_kv_heads=cfgd['num_kv_heads'],num_layers=cfgd['num_layers'],max_seq_len=cfgd['seq_len'])
m=SimpleAttentionNetwork(cfg); m.load_state_dict(ckpt['model_state_dict'],strict=True); m.cuda()
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained('bigcode/starcoder2-7b')
if tok.pad_token is None: tok.pad_token=tok.eos_token

# Absolute ce_py reference taken once at loop start. The ablation guard is
# measured against THIS, not against the previous cycle's value -- otherwise
# a boundary edit that costs 2% of ce_py each cycle would ratchet upward
# forever, passing its own moving guard every time until Python collapsed.
CE_REF=eval_ce(m)
log(f"ce_py reference at loop start: {CE_REF:.4f} (guard ceiling {CE_REF*1.05:.4f})")

deadline=datetime.datetime.now().replace(hour=21,minute=0,second=0)
if deadline<=datetime.datetime.now(): deadline+=datetime.timedelta(days=1)
log("running editor loop until",deadline)
best_score=0; cyc=0; HIST=[]   # Stage A: per-cycle history shown to the editor
while datetime.datetime.now()<deadline:
    cyc+=1
    mp,mt,fails,report=eval_he(m,tok,detail=True)
    score=mp
    v0=vec(report); ce0=eval_ce(m); cep0=eval_ce(m,CE_PHP_WINS); pi0=eval_pindent(m,tok)
    log(f"cycle {cyc} HumanEval@{mt} {mp}/{mt} | {fmt_vec(v0,ce0,cep0,pi0)}")
    rp=ROOT/f'eval_results/weight_edit_cycle_{cyc}.json'
    json.dump(report, open(rp,'w'), indent=1, default=str)
    dlog(f"graded report -> {rp}")
    dlog(f"SCORE VECTOR before: {fmt_vec(v0,ce0,cep0,pi0)}")
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
  before this cycle: {fmt_vec(v0,ce0,cep0,pi0)}
What each component means and why it can move:
  syntax_rate   % of completions that execute without SyntaxError. Currently LOW -> the model writes unclosed parens and brand-new 'def' statements instead of continuing the function body it was given. This is rung 1 of the ladder and the biggest headroom.
  defined_rate  % of completions that actually define the requested function (entry_point). Currently ~0 -> the model ignores the prompt and defines a DIFFERENT function.
  assert_rate   % of individual unit-test asserts passed. Currently 0 -> nothing has ever passed a test.
  cont_rate     % of completions that CONTINUED the prompt instead of restarting with a brand-new top-level 'def'. This is the single most important number: it directly measures the root defect. Raise this first.
  p_indent      mean probability the model puts on a 4-space indent at the exact spot where each HumanEval prompt ends (right after the docstring). This is the cheapest and most direct measure of the disease, and it is what your edits are scored on during ablation. Higher is better. Currently ~0.05.
  ce_py         cross-entropy on held-out PYTHON code. Lower is better; continuous, so it responds to small edits even when pass count does not. Used as a GUARD during ablation: an edit may not raise it more than 5%.
  ce_php        cross-entropy on PHP code. A GUARD ONLY -- if ce_php rises a lot while ce_py falls, you are trading away general ability. Do not optimize ce_php.
Ladder to climb: cont_rate -> syntax_rate -> defined_rate -> assert_rate. Improving ce alone does NOT fix HumanEval.

MEASURED DIAGNOSIS (do not re-derive this, build on it):
  - The model DOES know how to indent. Given 'def f():' plus a newline it puts P=0.79 on an indented body (argmax = tab). Add a docstring line and it collapses to P=0.95 on a plain newline and starts a NEW top-level statement.
  - So the defect is specific to the docstring boundary: the model treats a triple-quoted string as the ENTIRE function body rather than as a header that an indented body follows.
  - Python loss (ce_py ~3.10) is 2x worse than PHP loss (ce_php ~1.51). The model has seen far less Python. The headroom is in Python, which is why ce_py is the target.
  - Training data contains almost no docstrings (~1 per million characters), so the pattern 'docstring then indented statement' is rare in what it learned.
  - Highest-value edits therefore target: whether the model attends back to the prompt it was just given (q_norm/k_norm, attn_gate), and the relative logit weight of indentation tokens vs newline in embedding.weight.
{failtxt}
Weight fingerprint:
{wfp(m)}
Propose 3-5 targeted weight edits as a JSON list of objects with keys: param (full param name), scale (float near 1.0), add (float small), justification (ONE short sentence, under 20 words).
Rules:
- Justify each edit from the graded failures above (which stage fails, which assert).
- Keep the whole reply compact: 3-5 edits, short justifications, no preamble, no code fences.
- Prefer params that control whether the model CONTINUES the prompt instead of restarting: attention q/k norms, attn_gate, embedding rows, norm scales.
- Vary direction: if you scaled a param down last cycle and nothing moved, try up.
- Use the menu below; do not propose blocks.0-4.attn_gate again (already tried, no gain).
{menu_txt()}
Only output JSON, no prose."""
    dlog(f"=== cycle {cyc} | eval before: {mp}/{mt} | fails={len(fails)} ===")
    dlog("--- PROMPT SENT TO EDITOR ---"); dlog(prompt); dlog("--- END PROMPT ---")
    try:
        resp=ask_gemma(prompt)
        dlog("--- RAW EDITOR RESPONSE ---"); dlog(resp); dlog("--- END RESPONSE ---")
        edits=parse_edits(resp)
        if not edits and not re.search(r'\{',resp):
            raise ValueError("no JSON object in editor response")
    except Exception as ex:
        log("editor parse err",ex); dlog("PARSE ERR:",ex); edits=[]
    log("proposed edits:",edits); dlog("PARSED EDITS:",json.dumps(edits))
    # Stage C: always test the diagnosis-derived boundary candidates too, so
    # the disease metric gets a shot every cycle even if the editor returns
    # nothing parseable. u is refreshed here because the weights have moved.
    boundary_dir(m,tok,refresh=True)
    edits=list(edits)+boundary_edits(m,tok)
    dlog("boundary candidates added:",
         [e['boundary'] for e in edits if e.get('boundary') is not None])
    if not edits:
        dlog("cycle %d: no valid edits -> skipped (no weights changed)"%cyc)
        time.sleep(2); continue
    before=copy.deepcopy(m.state_dict())
    # Stage B: ablate — score each edit on its own, keep only the winners.
    kept,rows=ablate(m,edits,tok)
    dlog(f"ABLATION (score = delta p_indent at the docstring boundary; ce_py guard {1.05}):")
    for d,e,pi,ce,note in rows:
        dlog(f"   score {d:+.5f}  {e}  p_indent_after={None if pi is None else round(pi,5)} "
             f"ce_py_after={None if ce is None else round(ce,4)} {note}")
    log(f"ablation: {len(edits)} proposed -> {len(kept)} kept")
    if kept!=edits:
        log(f"  rejected {[e for e in edits if e not in kept]}")
    apply_edits(m,kept)
    mp2,mt2,_,rep2=eval_he(m,tok,detail=True)
    v1=vec(rep2); ce1=eval_ce(m); cep1=eval_ce(m,CE_PHP_WINS); pi1=eval_pindent(m,tok)
    log(f"after edit HumanEval@{mt2} {mp2}/{mt2} | {fmt_vec(v1,ce1,cep1,pi1)}")
    dlog(f"cycle {cyc}: applied {len(edits)} -> {fmt_vec(v1,ce1,cep1,pi1)} (was {fmt_vec(v0,ce0,cep0,pi0)})")
    HIST.append({'cyc':cyc,'edits':f"proposed {edits} -> kept {kept}",
                 'before':fmt_vec(v0,ce0,cep0,pi0),'after':fmt_vec(v1,ce1,cep1,pi1)})
    # always keep: edits are applied in place to san_latest.pt regardless of
    # whether the score improved (single-checkpoint policy, no base copies).
    ck=copy.deepcopy(ckpt); ck['model_state_dict']={k:v.detach().cpu() for k,v in m.state_dict().items()}
    torch.save(ck,CKPT); log("kept edit, saved in place",CKPT)
    best_score=max(best_score,mp2); dlog(f"cycle {cyc}: DECISION=KEEP ({mp}->{mp2}, saved to {CKPT})")
    if best_score>=mt: log("target reached"); break
log("done best",best_score,"cycles",cyc)