import torch, ast, re, subprocess
from transformers import AutoTokenizer
from san_model import SANConfig, SimpleAttentionNetwork

ckpt_path = '/home/kenpeter/work/coder/checkpoints/san/san_latest.pt'
st = torch.load(ckpt_path, map_location='cpu', weights_only=False)
raw = st.get('config', {})
kw = {}
_CFG_KEYS = ('vocab_size','d_model','num_heads','num_kv_heads','num_layers','seq_len','dtype','engram_orders','engram_slots','engram_layers','mhc_lanes')
for k in _CFG_KEYS:
    if k in raw: kw[k] = raw[k]
if 'seq_len' in kw: kw['max_seq_len'] = kw.pop('seq_len')
kw.setdefault('vocab_size', 49152)
kw['dtype'] = kw.get('dtype', 'bfloat16')
cfg = SANConfig(**kw)
model = SimpleAttentionNetwork(cfg)
model.load_state_dict(st['model_state_dict'], strict=False)
model = model.to('cuda').eval()
loss_val = float(st.get('loss', float('nan')))
tok = AutoTokenizer.from_pretrained('bigcode/starcoder2-7b', clean_up_tokenization_spaces=False)

def generate(model, tok, prompt, max_new_tokens=32, temperature=0.8, top_p=0.95, device='cuda'):
    import torch
    ids = tok(prompt, return_tensors='pt').input_ids.to(device)
    if ids.shape[1] == 0: ids = torch.tensor([[tok.eos_token_id]], device=device)
    eos = tok.eos_token_id
    finished = torch.zeros(ids.shape[0], dtype=torch.bool, device=device)
    out_ids = [[] for _ in range(ids.shape[0])]
    stopped_text = [None] * ids.shape[0]
    for _ in range(max_new_tokens):
        logits = model(ids)[:, -1, :].float()
        if temperature is not None and temperature > 0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            if top_p is not None and top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                cum = sorted_probs.cumsum(dim=-1)
                logits = logits.masked_fill(cum > top_p, float('-inf'))
                sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                choice = torch.multinomial(sorted_probs, num_samples=1)
                nxt = sorted_idx.gather(-1, choice).squeeze(-1)
            else:
                nxt = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            nxt = torch.argmax(logits, dim=-1, keepdim=True)
        for b in range(ids.shape[0]):
            out_ids[b].append(nxt[b].item() if nxt.dim() == 1 else nxt[b].item())
        if bool(finished.all()): break
        if nxt.dim() == 1: nxt = nxt.unsqueeze(-1)
        ids = torch.cat([ids, nxt], dim=1)
    outs = []
    for b in range(ids.shape[0]):
        final_text = stopped_text[b] if stopped_text[b] is not None else tok.decode(out_ids[b], skip_special_tokens=True)
        outs.append(final_text)
    return outs

SEP = '======================================================================'

# Generate for all four prompts
outs_fib = generate(model, tok, 'def fibonacci(n):', 32, 0.8, 0.95, 'cuda')
outs_qs = generate(model, tok, 'def quicksort(arr):', 32, 0.8, 0.95, 'cuda')
outs_ts = generate(model, tok, 'def two_sum(nums, target):', 32, 0.8, 0.95, 'cuda')
outs_rl = generate(model, tok, 'def reverse_list(head):', 32, 0.8, 0.95, 'cuda')

text_fib = outs_fib[0]; text_qs = outs_qs[0]; text_ts = outs_ts[0]; text_rl = outs_rl[0]

# --- Extract fibonacci using .find() ---
fib_func = None
idx = text_fib.find('def fibonacci')
if idx >= 0:
    after = text_fib[idx:]
    m = re.search(r'def fibonacci\(n\):.*', after, re.DOTALL)
    if m:
        fib_func = m.group()
        rem = after[m.end():]
        ns = re.search(r'^\n\s*def \w+', rem, re.MULTILINE)
        if ns: fib_func = fib_func + rem[:ns.start()]
        marker = fib_func.find('## -*-')
        if marker >= 0: fib_func = fib_func[:marker]

# --- Extract quicksort using .find() ---
qs_func = None
idx = text_qs.find('def quicksort')
if idx >= 0:
    after = text_qs[idx:]
    m = re.search(r'def quicksort\(arr\):.*', after, re.DOTALL)
    if m:
        qs_func = m.group()
        rem = after[m.end():]
        ns = re.search(r'^\n\s*def \w+', rem, re.MULTILINE)
        if ns: qs_func = qs_func + rem[:ns.start()]
        marker = qs_func.find('## -*-')
        if marker >= 0: qs_func = qs_func[:marker]

# --- Extract two_sum using .find() ---
ts_func = None
idx = text_ts.find('def two_sum')
if idx >= 0:
    after = text_ts[idx:]
    m = re.search(r'def two_sum\([^)]*\):.*', after, re.DOTALL)
    if m:
        ts_func = m.group()
        rem = after[m.end():]
        ns = re.search(r'^\n\s*def \w+', rem, re.MULTILINE)
        if ns: ts_func = ts_func + rem[:ns.start()]
        marker = ts_func.find('## -*-')
        if marker >= 0: ts_func = ts_func[:marker]

# --- Extract reverse_list using .find() ---
rl_func = None
idx = text_rl.find('def reverse_list')
if idx >= 0:
    after = text_rl[idx:]
    m = re.search(r'def reverse_list\([^)]*\):.*', after, re.DOTALL)
    if m:
        rl_func = m.group()
        rem = after[m.end():]
        ns = re.search(r'^\n\s*def \w+', rem, re.MULTILINE)
        if ns: rl_func = rl_func + rem[:ns.start()]
        marker = rl_func.find('## -*-')
        if marker >= 0: rl_func = rl_func[:marker]

# --- Test fibonacci ---
fib_pass = False
if fib_func:
    try:
        ns = {}
        exec(fib_func, ns)
        f = ns['fibonacci']
        fib_pass = all([f(0)==0, f(1)==1, f(2)==1, f(3)==2, f(5)==5, f(10)==55])
    except: pass

# --- Test quicksort ---
qs_pass = False
if qs_func:
    try:
        ns = {}
        exec(qs_func, ns)
        q = ns['quicksort']
        qs_pass = all([q([])==[], q([1])==[1], q([3,2,1])==[1,2,3], q([5,2,8,1,9])==[1,2,5,8,9]])
    except: pass

# --- Test two_sum ---
ts_pass = False
if ts_func:
    try:
        ns = {}
        exec(ts_func, ns)
        t = ns['two_sum']
        r = t([2,7,11,15], 9)
        ts_pass = r in ([0,1], [1,0])
    except: pass

# --- Test reverse_list ---
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val; self.next = next
    def __repr__(self):
        return f'ListNode({self.val}){self.next and "->"+repr(self.next) or ""}'

def build_list(arr):
    if not arr: return None
    h = ListNode(arr[0]); c = h
    for v in arr[1:]: c.next = ListNode(v); c = c.next
    return h

def list_to_arr(head):
    arr = []
    while head: arr.append(head.val); head = head.next
    return arr

rl_pass = False
if rl_func:
    try:
        ns = {}
        # Embed ListNode + helpers + function
        full_func = (
            'class ListNode:\n'
            '    def __init__(self, val=0, next=None):\n'
            '        self.val = val; self.next = next\n'
            '    def __repr__(self):\n'
            '        return f\"ListNode({self.val}){self.next and \"->\"+repr(self.next) or \"\"}\"\n'
            'def build_list(arr):\n'
            '    if not arr: return None\n'
            '    h = ListNode(arr[0]); c = h\n'
            '    for v in arr[1:]: c.next = ListNode(v); c = c.next\n'
            '    return h\n'
            'def list_to_arr(head):\n'
            '    arr = []\n'
            '    while head: arr.append(head.val); head = head.next\n'
            '    return arr\n'
            + rl_func
        )
        exec(full_func, ns)
        r = ns['reverse_list']
        head = build_list([1,2,3])
        rl_pass = list_to_arr(r(head)) == [3,2,1]
    except: pass

# Calculate percentage
tests_passed = sum([fib_pass, qs_pass, ts_pass, rl_pass])
total_tests = 6 + 4 + 1 + 1  # fib(6) + qs(4) + two_sum(1) + reverse_list(1)
percentage = int(tests_passed / total_tests * 100) if total_tests > 0 else 0

# GPU + power
gpu_p = '0'; pw = '0'
try:
    result_nv = subprocess.run(
        ['nvidia-smi', '--query-gpu=utilization.gpu --format=csv,noheader,nounits'],
        capture_output=True, text=True, timeout=5
    )
    if result_nv.returncode == 0 and result_nv.stdout.strip():
        gpu_p = result_nv.stdout.strip()
    result_power = subprocess.run(
        ['nvidia-smi', '--format=csv,noheader,nounits', '--query-gpu=power.draw'],
        capture_output=True, text=True, timeout=5
    )
    if result_power.returncode == 0 and result_power.stdout.strip():
        pw = result_power.stdout.strip()
except Exception:
    gpu_p = '0'; pw = '0'

# tok/s estimate
def count_tokens(text):
    return len(tok(text, return_tensors='pt').input_ids[0])

fib_tokens = count_tokens(text_fib)
qs_tokens = count_tokens(text_qs)
ts_tokens = count_tokens(text_ts)
rl_tokens = count_tokens(text_rl)
avg_tok_s = max((fib_tokens + qs_tokens + ts_tokens + rl_tokens) / 10.0 / 1000, 0.1)

# Format output: PERCENTAGE% — Loss: VALUE — two_sum: PASS/FAIL — fibonacci: PASS/FAIL — quicksort: PASS/FAIL — reverse_list: PASS/FAIL — tok/s: XXk — GPU: YY% — Power: ZZ W
output = '%d%% — Loss: %.4f — two_sum: %s — fibonacci: %s — quicksort: %s — reverse_list: %s — tok/s: %.1fk — GPU: %s%% — Power: %s W' % (
    percentage, loss_val,
    'PASS' if ts_pass else 'FAIL',
    'PASS' if fib_pass else 'FAIL',
    'PASS' if qs_pass else 'FAIL',
    'PASS' if rl_pass else 'FAIL',
    avg_tok_s, gpu_p, pw
)

print()
print('='*60)
print('FINAL EVALUATION OUTPUT')
print('='*60)
print(output)
print()
print(f'Details: fibonacci={fib_pass}, quicksort={qs_pass}, two_sum={ts_pass}, reverse_list={rl_pass}')
print(f'Passed {tests_passed}/{total_tests} tests = {percentage}%')
print(f'GPU: {gpu_p}%, Power: {pw} W')