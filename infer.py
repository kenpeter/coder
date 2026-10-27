"""Inference for the trained Simple Attention Network (SAN) code model.

Loads checkpoints/san/san_latest.pt (written by train_san.py), rebuilds the
architecture from the checkpoint's own config, and generates code.

Usage:
    python infer.py --prompt "def add(a, b):\n    \"\"\"Return a + b.\"\"\"\n"
    python infer.py --instruction "write a python function that reverses a list"
    python infer.py --prompt-file prompt.py --max-new-tokens 256 --temperature 0.2
"""
import argparse

import torch
from transformers import AutoTokenizer

from san_model import SANConfig, SimpleAttentionNetwork

DEFAULT_CKPT = "checkpoints/san/san_latest.pt"
DEFAULT_TOKENIZER = "bigcode/starcoder2-7b"

# Architecture keys we copy from the training config.
_CFG_KEYS = (
    "vocab_size", "d_model", "num_heads", "num_kv_heads", "num_layers",
    "seq_len", "dtype", "engram_orders", "engram_slots", "engram_layers",
    "mhc_lanes",
)


def load_model(ckpt_path=DEFAULT_CKPT, device="cuda", dtype=None, verbose=True):
    """Return (model, tokenizer, checkpoint_meta)."""
    st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw = st.get("config", {})
    kw = {}
    for k in _CFG_KEYS:
        if k in raw:
            kw[k] = raw[k]
    if "seq_len" in kw:  # training uses seq_len; SANConfig calls it max_seq_len
        kw["max_seq_len"] = kw.pop("seq_len")
    kw.setdefault("vocab_size", 49152)
    kw["dtype"] = dtype or kw.get("dtype", "bfloat16")
    # Checkpoint does not store engram_heads; engram_geometry derives it from
    # d_model/orders exactly as at training time.
    cfg = SANConfig(**kw)

    model = SimpleAttentionNetwork(cfg)
    missing, unexpected = model.load_state_dict(st["model_state_dict"], strict=False)
    if verbose:
        print(f"loaded {ckpt_path} (step {st.get('step')}, loss {float(st.get('loss', float('nan'))):.4f})")
        print(f"params {model.count_parameters()/1e6:.1f}M  dtype {cfg.dtype}  layers {cfg.num_layers}")
        if missing:
            print(f"  missing keys: {len(missing)}")
        if unexpected:
            print(f"  unexpected keys: {len(unexpected)}")
    model = model.to(device).eval()
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    tok = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER, clean_up_tokenization_spaces=False)
    return model, tok, st


def _apply_repetition_penalty(logits, seen_ids, penalty):
    if penalty is None or penalty == 1.0:
        return logits
    ids = torch.tensor(sorted(set(seen_ids)), dtype=torch.long, device=logits.device)
    vals = logits[ids]
    logits[ids] = torch.where(vals > 0, vals / penalty, vals * penalty)
    return logits


@torch.no_grad()
def generate(model, tok, prompt, max_new_tokens=256, temperature=0.2, top_p=0.95,
             top_k=None, repetition_penalty=1.05, stop_strings=(), device="cuda",
             batch_size=1):
    """Autoregressive completion. batch_size>1 replicates the prompt (sampling only)."""
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    if ids.shape[1] == 0:
        ids = torch.tensor([[tok.eos_token_id]], device=device)
    if batch_size > 1:
        ids = ids.repeat(batch_size, 1)

    eos = tok.eos_token_id
    finished = torch.zeros(ids.shape[0], dtype=torch.bool, device=device)
    out_ids = [[] for _ in range(ids.shape[0])]
    stopped_text = [None] * ids.shape[0]

    for _ in range(max_new_tokens):
        logits = model(ids)[:, -1, :].float()  # (B,V)
        for b in range(ids.shape[0]):
            _apply_repetition_penalty(logits[b], ids[b].tolist(), repetition_penalty)
        if temperature is not None and temperature > 0:
            logits = logits / temperature
            if top_k:
                k = min(top_k, logits.shape[-1])
                kth = torch.topk(logits, k, dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--prompt", default=None, help="raw code prompt to continue")
    ap.add_argument("--prompt-file", default=None)
    ap.add_argument("--instruction", default=None, help="natural-language instruction (wrapped as a comment)")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.2, help="0 = greedy")
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--repetition-penalty", type=float, default=1.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if args.prompt_file:
        prompt = open(args.prompt_file).read()
    elif args.instruction:
        prompt = f"# {args.instruction.strip()}\n"
    else:
        prompt = args.prompt or ""
    if not prompt:
        raise SystemExit("pass --prompt, --prompt-file or --instruction")

    model, tok, _ = load_model(args.checkpoint, device=args.device)
    out = generate(
        model, tok, prompt,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        top_p=args.top_p, top_k=args.top_k,
        repetition_penalty=args.repetition_penalty, device=args.device,
    )[0]
    print("=" * 70)
    print(prompt + out)


if __name__ == "__main__":
    main()
