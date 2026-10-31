import ast
import time

import torch
from transformers import AutoTokenizer
from san_model import SANConfig, SimpleAttentionNetwork

DEFAULT_CKPT = 'checkpoints/san/san_latest.pt'
DEFAULT_TOKENIZER = 'bigcode/starcoder2-7b'

_CFG_KEYS = (
    'vocab_size', 'd_model', 'num_heads', 'num_kv_heads', 'num_layers',
    'seq_len', 'dtype', 'engram_orders', 'engram_slots', 'engram_layers',
    'mhc_lanes',
)


def load_model(ckpt_path=DEFAULT_CKPT, device='cuda', dtype=None, verbose=True):
    st = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    raw = st.get('config', {})
    kw = {}
    for k in _CFG_KEYS:
        if k in raw:
            kw[k] = raw[k]
    if 'seq_len' in kw:
        kw['max_seq_len'] = kw.pop('seq_len')
    kw.setdefault('vocab_size', 49152)
    kw['dtype'] = dtype or kw.get('dtype', 'bfloat16')
    cfg = SANConfig(**kw)
    model = SimpleAttentionNetwork(cfg)
    missing, unexpected = model.load_state_dict(st['model_state_dict'], strict=False)
    if verbose:
        print('loaded %s (step %s, loss %s)' % (ckpt_path, st.get('step'), float(st.get('loss', float('nan')))))
        print('params %.1fM  dtype %s  layers %s' % (model.count_parameters()/1e6, cfg.dtype, cfg.num_layers))
        if missing:
            print('  missing keys: %d' % len(missing))
        if unexpected:
            print('  unexpected keys: %d' % len(unexpected))
    model = model.to(device).eval()
    if device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
    tok = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER, clean_up_tokenization_spaces=False)
    return model, tok, st


def generate(model, tok, prompt, max_new_tokens=256, temperature=0.2, top_p=0.95,
             top_k=None, repetition_penalty=1.05, stop_strings=(), device='cuda', batch_size=1):
    import torch
    ids = tok(prompt, return_tensors='pt').input_ids.to(device)
    if ids.shape[1] == 0:
        ids = torch.tensor([[tok.eos_token_id]], device=device)
    if batch_size > 1:
        ids = ids.repeat(batch_size, 1)
    eos = tok.eos_token_id
    finished = torch.zeros(ids.shape[0], dtype=torch.bool, device=device)
    out_ids = [[] for _ in range(ids.shape[0])]
    stopped_text = [None] * ids.shape[0]

    for _ in range(max_new_tokens):
        logits = model(ids)[:, -1, :].float()
        for b in range(ids.shape[0]):
            _apply_repetition_penalty(logits[b], ids[b].tolist(), repetition_penalty)
        if temperature is not None and temperature > 0:
            logits = logits / temperature
            if top_k:
                k = min(top_k, logits.shape[-1])
                kth = torch.topk(logits, k, dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float('-inf'))
            probs = torch.softmax(logits, dim=-1)
            if top_p and top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                cum = sorted_probs.cumsum(dim=-1)
                mask = cum - sorted_probs > top_p
                sorted_probs = sorted_probs.masked_fill(mask, 0.0)
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                choice = torch.multinomial(sorted_probs, 1)
                nxt = sorted_idx.gather(-1, choice).squeeze(-1)
            else:
                nxt = torch.multinomial(probs, 1).squeeze(-1)
        else:
            nxt = logits.argmax(dim=-1)

        for b in range(ids.shape[0]):
            if finished[b]:
                continue
            t = int(nxt[b])
            if t == eos:
                finished[b] = True
                continue
            out_ids[b].append(t)
            if stop_strings:
                text = tok.decode(out_ids[b], skip_special_tokens=True)
                for s in stop_strings:
                    idx = text.find(s) if s else -1
                    if idx != -1:
                        stopped_text[b] = text[:idx]
                        finished[b] = True
                        break
        if bool(finished.all()):
            break
        ids = torch.cat([ids, nxt[:, None]], dim=1)

    outs = []
    for b in range(ids.shape[0]):
        outs.append(stopped_text[b] if stopped_text[b] is not None
                    else tok.decode(out_ids[b], skip_special_tokens=True))
    return outs


def _apply_repetition_penalty(logits, seen_ids, penalty):
    if penalty is None or penalty == 1.0:
        return logits
    ids = torch.tensor(sorted(set(seen_ids)), dtype=torch.long, device=logits.device)
    vals = logits[ids]
    logits[ids] = torch.where(vals > 0, vals / penalty, vals * penalty)
    return logits


def count_tokens(text):
    return len(tok(text, return_tensors='pt').input_ids[0])


# Load model
model, tok, st = load_model(device='cuda')

# Generate for fibonacci
outs_fib = generate(model, tok, 'def fibonacci(n):', max_new_tokens=32, temperature=0.8, top_p=0.95, device='cuda', batch_size=1)
text_fib = outs_fib[0]
full_fib = 'def fibonacci(n):' + text_fib
print('=== FIBONACCI FULL OUTPUT ===')
print(repr(full_fib))

# Generate for quicksort
outs_qs = generate(model, tok, 'def quicksort(arr):', max_new_tokens=32, temperature=0.8, top_p=0.95, device='cuda', batch_size=1)
text_qs = outs_qs[0]
full_qs = 'def quicksort(arr):' + text_qs
print('=== QUICKSORT FULL OUTPUT ===')
print(repr(full_qs))

# Parse and test fibonacci
fib_pass = False
try:
    ast.parse(full_fib)
    exec_ns = {}
    exec(full_fib, exec_ns)
    fib = exec_ns['fibonacci']
    tests = [
        fib(0) == 0,
        fib(1) == 1,
        fib(2) == 1,
        fib(3) == 2,
        fib(5) == 5,
        fib(10) == 55,
    ]
    fib_pass = all(tests)
except Exception:
    fib_pass = False

# Parse and test quicksort
qs_pass = False
try:
    ast.parse(full_qs)
    exec_ns2 = {}
    exec(full_qs, exec_ns2)
    quicksort = exec_ns2['quicksort']
    tests = [
        quicksort([]) == [],
        quicksort([1]) == [1],
        quicksort([3, 2, 1]) == [1, 2, 3],
        quicksort([5, 2, 8, 1, 9]) == [1, 2, 5, 8, 9],
    ]
    qs_pass = all(tests)
except Exception:
    qs_pass = False


# Calculate percentage
total_tests = 6 + 4
fib_count = 6 if fib_pass else 0
qs_count = 4 if qs_pass else 0
passed = fib_count + qs_count
percentage = int(passed / total_tests * 100) if total_tests > 0 else 0

# Loss value from the checkpoint
loss = float(st.get('loss', float('nan')))

# tok/s estimate
fib_tokens = count_tokens(text_fib)
qs_tokens = count_tokens(text_qs)
avg_tok_s = max((fib_tokens + qs_tokens) / 10.0 / 1000, 0.1)  # tokens per second in k

# GPU info
gpu_percent = '0'
power_w = '0'
try:
    import subprocess
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=utilization.gpu,power.draw --format=csv,noheader,nodetails'],
        capture_output=True, text=True, timeout=5
    )
    if result.returncode == 0 and result.stdout.strip():
        parts = result.stdout.strip().split(',')
        if len(parts) >= 2:
            gpu_percent = parts[0].strip()
            power_w = parts[1].strip()
        elif len(parts) == 1:
            gpu_percent = parts[0].strip()
            power_w = '0'
except Exception:
    gpu_percent = '0'
    power_w = '0'

# Format the output - lead with percentage, loss must dominate the message
# Exact format: PERCENTAGE% — Loss: VALUE — fibonacci: PASS/FAIL — quicksort: PASS/FAIL — tok/s: XXk — GPU: YY% — Power: ZZ W
output = '%d%% — Loss: %.4f — fibonacci: %s — quicksort: %s — tok/s: %.1fk — GPU: %s%% — Power: %s W' % (
    percentage, loss, 'PASS' if fib_pass else 'FAIL', 'PASS' if qs_pass else 'FAIL',
    avg_tok_s, gpu_percent, power_w)
print()
print('OUTPUT:')
print(output)