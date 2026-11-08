"""Smoke test for the weight-edit flow (one cycle)."""
import subprocess, sys, time
sys.path.insert(0,'/home/kenpeter/work/coder')
import weight_edit_9b as w
# 1. load model + eval
ckpt,cfg,m=w.load_model(w.CKPT) if hasattr(w,'load_model') else None
print("module imports OK; eval_he signature:", w.eval_he.__doc__)
# dry-run eval on a single problem by calling eval_he with n=1
import torch
from san_model import SimpleAttentionNetwork, SANConfig
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained('bigcode/starcoder2-7b')
if tok.pad_token is None: tok.pad_token=tok.eos_token
ckptd=ckpt['config'] if isinstance(ckpt,dict) else w.torch.load(w.CKPT,map_location='cpu',weights_only=False)['config']
cfg=SANConfig(vocab_size=ckptd['vocab_size'],d_model=ckptd['d_model'],num_heads=ckptd['num_heads'],num_kv_heads=ckptd['num_kv_heads'],num_layers=ckptd['num_layers'],max_seq_len=ckptd['seq_len'])
m=SimpleAttentionNetwork(cfg); m.load_state_dict(w.torch.load(w.CKPT,map_location='cpu',weights_only=False)['model_state_dict'],strict=True); m.cuda()
p,t,fails=w.eval_he(m,tok,n=1)
print(f"eval_he n=1 -> pass={p}/{t} fails={len(fails)}")
assert t==1
# 2. 9B editor returns parseable JSON
prompt="Return JSON list: [{\"param\":\"final_norm.scale\",\"scale\":1.0,\"add\":0.0}]"
r=w.ask_9b(prompt)
print("9B response:", r[:200])
import re,json
mch=re.search(r'\[.*\]',r,re.S)
assert mch, "9B did not return JSON"
edits=json.loads(mch.group(0))
print("parsed edits:", edits)
assert isinstance(edits,list) and len(edits)>0
print("FLOW OK")