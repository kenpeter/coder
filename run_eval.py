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

st = torch.load(DEFAULT_CKPT, map_location='cpu', weights_only=False)
raw = st.get('config', {})
kw = {}
for k in _CFG_KEYS:
    if k in raw:
        kw[k] = raw[k]
if 'seq_len' in kw:
    kw['max_seq_len'] = kw.pop('seq_len')
kw.setdefault('vocab_size', 49152)
kw['dtype'] = kw.get('dtype', 'bfloat16')
cfg = SANConfig(**kw)
model = SimpleAttentionNetwork(cfg)
missing, unexpected = model.load_state_dict(st['model_state_dict'], strict=False)
print('loaded %s (step %s, loss %s)' % (DEFAULT_CKPT, st.get('step'), float(st.get('loss', float('nan')))))

model = model.to('cuda').eval()
torch.backends.cuda.matmul.allow_tf32 = True
tok = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER, clean_up_tokenization_spaces=False)


def generate(model, tok, prompt, max_new_tokens=256, temperature=0.2, top_p=0.95,
             top_k=None, repetition_penalty=1.05, stop_strings=(), device='cuda', batch_size=1):
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
            ids_list = ids[b].tolist()
            if repetition_penalty and repetition_penalty != 1.0:
                seen = set(ids_list)
                vals = logits[b][list(seen)]
                logits[b][list(seen)] = torch.where(vals > 0, vals / repetition_penalty, vals * repetition_penalty)
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


# Generate for fibonacci
outs_fib = generate(model, tok, 'def fibonacci(n):', max_new_tokens=32, temperature=0.8, top_p=0.95, device='cuda', batch_size=1)
text_fib = outs_fib[0]
print('=== FIBONACCI OUTPUT ===')
print(repr(text_fib))

# Extract function definition
idx_fib = text_fib.find('def fibonacci')
if idx_fib >= 0:
    func_text_fib = text_fib[idx_fib:]
    print('=== EXTRACTED FIBONACCI ===')
    print(func_text_fib)
else:
    print('Could not find def fibonacci in output')

# Generate for quicksort
outs_qs = generate(model, tok, 'def quicksort(arr):', max_new_tokens=32, temperature=0.8, top_p=0.95, device='cuda', batch_size=1)
text_qs = outs_qs[0]
print('=== QUICKSORT OUTPUT ===')
print(repr(text_qs))

# Extract function definition
idx_qs = text_qs.find('def quicksort')
if idx_qs >= 0:
    func_text_qs = text_qs[idx_qs:]
    print('=== EXTRACTED QUICKSORT ===')
    print(func_text_qs)
else:
    print('Could not find def quicksort in output')