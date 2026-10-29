"""Throughput sweep for the coder SAN, mirroring x-small/bench_san_tput.py.

Measures the real training compute path (san_loss = main + 0.1*MTP) for:
  --compile whole   : torch.compile(model.compute_loss)  (x-small bench style)
  --compile block   : per-block forwards (our old train_san._compile_model)
  --compile none    : eager
with or without gradient checkpointing, over a small (B,S) grid.
"""
import time, argparse, torch
from san_model import SimpleAttentionNetwork, SANConfig
from train_san import san_loss, _compile_model

device = "cuda"
torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def build(compile_mode, checkpoint, ckpt_every, num_layers, autotune):
    torch.manual_seed(42)
    cfg = SANConfig(vocab_size=49152, d_model=384, num_heads=8, num_kv_heads=4,
                    num_layers=num_layers, max_seq_len=4096,
                    use_checkpoint=checkpoint, ckpt_every=ckpt_every,
                    dtype="bfloat16")
    model = SimpleAttentionNetwork(cfg).to(device).train()
    if compile_mode == "whole":
        m = "max-autotune-no-cudagraphs" if autotune else "default"
        model.compute_loss = torch.compile(model.compute_loss, dynamic=True, mode=m)
    elif compile_mode == "block":
        model = _compile_model(model)
    return model, cfg


def run(model, B, S, warm=3, steps=8):
    opt = torch.optim.AdamW(model.parameters(), lr=4e-4, fused=True)
    torch.cuda.reset_peak_memory_stats()
    t0 = None
    for i in range(warm + steps):
        x = torch.randint(0, 49152, (B, S), device=device)
        y = torch.randint(0, 49152, (B, S), device=device)
        loss, _, _ = san_loss(model, x, y, 0.1)
        loss.backward()
        opt.step(); opt.zero_grad(set_to_none=True)
        if i == warm - 1:
            torch.cuda.synchronize(); t0 = time.time()
    torch.cuda.synchronize()
    dt = time.time() - t0
    return (B * S * steps) / dt, torch.cuda.max_memory_allocated() / 1e9


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--compile", default="whole", choices=["whole", "block", "none"])
    ap.add_argument("--no-checkpoint", action="store_true")
    ap.add_argument("--ckpt-every", type=int, default=3)
    ap.add_argument("--layers", type=int, default=27)
    ap.add_argument("--autotune", action="store_true")
    ap.add_argument("--grid", default="8x2048,6x2048,4x2048,8x1024,16x1024")
    a = ap.parse_args()

    print(f"cfg compile={a.compile} checkpoint={not a.no_checkpoint} "
          f"ckpt_every={a.ckpt_every} layers={a.layers} autotune={a.autotune}", flush=True)
    model, cfg = build(a.compile, not a.no_checkpoint, a.ckpt_every, a.layers, a.autotune)
    print(f"PARAMS: {model.count_parameters()/1e6:.2f}M", flush=True)
    for pair in a.grid.split(","):
        B, S = map(int, pair.split("x"))
        try:
            tps, mem = run(model, B, S)
            print(f"{B:>4} x {S:<5} {tps:>9,.0f} tok/s   peak {mem:.1f} GB", flush=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"{B:>4} x {S:<5} OOM", flush=True)
        except Exception as e:
            print(f"{B:>4} x {S:<5} ERR {type(e).__name__}: {e}", flush=True)
            break
