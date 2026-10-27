"""Evaluate the trained SAN code model on HumanEval (execution-based pass@k).

Zero-shot: the model continues the HumanEval function prompt; the generated
completion is truncated at the next top-level definition, appended to the
official tests, and executed in a subprocess sandbox with a timeout.

Usage:
    python eval_code.py                          # greedy, all 164 tasks
    python eval_code.py --limit 20 --samples 5   # 20 tasks, 5 samples, pass@1/@5
    python eval_code.py --no-ppl                 # skip held-out perplexity
"""
import argparse
import ast
import gzip
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from infer import DEFAULT_CKPT, load_model, generate

DATA_GZ = Path("eval_data/HumanEval.jsonl.gz")
DATA_URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
VAL_DIR = Path("/home/kenpeter/work/data/_shards_code_easy")

STOPS = ("\nclass ", "\ndef ", "\nif __name__")


def ensure_humaneval():
    if not DATA_GZ.exists():
        DATA_GZ.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {DATA_URL}")
        subprocess.run(["curl", "-sSL", DATA_URL, "-o", str(DATA_GZ)], check=True)
    with gzip.open(DATA_GZ, "rt") as f:
        return [json.loads(line) for line in f]


def truncate_completion(text):
    """Keep only the completion of the current function body."""
    for stop in STOPS:
        i = text.find(stop)
        if i != -1:
            text = text[:i]
    # An unindented line ends the function body.
    lines = text.splitlines()
    out = []
    for line in lines:
        if out and line and not line[0].isspace():
            break
        out.append(line)
    return "\n".join(out).rstrip() + "\n"


def build_program(problem, completion):
    return (problem["prompt"] + completion + "\n\n" +
            problem["test"] + f"\n\ncheck({problem['entry_point']})\n")


def run_program(program, timeout=10):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "solution.py")
        with open(path, "w") as f:
            f.write(program)
        try:
            p = subprocess.run([sys.executable, path], cwd=d, capture_output=True,
                               text=True, timeout=timeout)
            return p.returncode == 0, (p.stdout + p.stderr)[-800:]
        except subprocess.TimeoutExpired:
            return False, f"TIMEOUT after {timeout}s"
        except Exception as e:
            return False, f"EXEC ERROR {e}"


def pass_at_k(n, c, k):
    """Unbiased pass@k estimator (Chen et al. 2021)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def shard_loss(model, path, seq_len=512, n_batches=4, batch_size=2, device="cuda"):
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)
    tokens = np.memmap(str(path), dtype=np.uint16, mode="r")
    need = seq_len + 1
    rng = np.random.default_rng(0)
    losses = []
    with torch.no_grad():
        for _ in range(n_batches):
            starts = [int(s) for s in rng.integers(0, len(tokens) - need, size=batch_size)]
            x = torch.stack([torch.from_numpy(tokens[s:s + seq_len].astype(np.int64)) for s in starts])
            y = torch.stack([torch.from_numpy(tokens[s + 1:s + seq_len + 1].astype(np.int64)) for s in starts])
            x, y = x.to(device), y.to(device)
            h = model._hidden(x, None).to(model.dtype)
            W = model.embedding.weight.t()
            tot, cnt = 0.0, 0
            for s in range(0, seq_len, 64):
                lg = h[:, s:s + 64] @ W
                ce = torch.nn.functional.cross_entropy(
                    lg.reshape(-1, lg.shape[-1]).float(), y[:, s:s + 64].reshape(-1),
                    reduction="sum")
                tot += float(ce)
                cnt += y[:, s:s + 64].numel()
            losses.append(tot / cnt)
    return sum(losses) / len(losses)


def val_shard_paths(max_per_group=6):
    """Validation shards actually held out from the training run.

    Preferred source: checkpoints/san/run_meta.json, written by train_san.py when
    the curriculum loader is built (holdout happens before the shard cap, so
    these shards were never trained on). Falls back to the md5-sorted 1% split
    of the flat easy directory for pre-curriculum runs.
    """
    meta = Path("checkpoints/san/run_meta.json")
    if meta.exists():
        try:
            with open(meta) as f:
                m = json.load(f)
            paths = []
            for t in m.get("tiers", {}).values():
                paths.extend(Path(p) for p in t.get("val_shards", []))
            paths = [p for p in paths if p.exists()]
            if paths:
                return paths
        except Exception:
            pass
    shards = sorted(VAL_DIR.glob("*.bin"), key=lambda p: hashlib.md5(str(p).encode()).hexdigest())
    n_val = max(1, int(len(shards) * 0.01))
    return shards[-n_val:]


def val_loss(model, device="cuda", n_batches=4):
    """Held-out loss over the run's validation shards (see val_shard_paths)."""
    paths = val_shard_paths()
    # Cap work: score at most 4 Python shards plus a spread of other languages.
    py = [p for p in paths if "-py-" in p.name][:4]
    other = [p for p in paths if "-py-" not in p.name]
    step = max(1, len(other) // 6)
    sample = py + other[::step][:6]
    per = {p.name: shard_loss(model, p, n_batches=n_batches, device=device) for p in sample}
    mean = sum(per.values()) / len(per)
    py_scores = {k: v for k, v in per.items() if "-py-" in k}
    py_mean = sum(py_scores.values()) / len(py_scores) if py_scores else None
    return mean, math.exp(mean), py_mean, (math.exp(py_mean) if py_mean is not None else None), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--limit", type=int, default=0, help="0 = all problems")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.2, help="0 = greedy")
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--timeout", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-ppl", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(0)
    problems = ensure_humaneval()
    if args.limit:
        problems = problems[:args.limit]

    model, tok, st = load_model(args.checkpoint, device=args.device)
    step = st.get("step")

    out_path = args.out or f"eval_results/humaneval_step{step}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    results = []
    t0 = time.time()
    n_total = len(problems) * args.samples
    done = 0

    print(f"HumanEval: {len(problems)} tasks x {args.samples} samples "
          f"(T={args.temperature}, device={args.device})")
    for pi, prob in enumerate(problems):
        outs = generate(
            model, tok, prob["prompt"],
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            top_p=args.top_p, repetition_penalty=1.05, device=args.device,
            batch_size=args.samples, stop_strings=STOPS,
        )
        rec = {"task_id": prob["task_id"], "samples": []}
        for out in outs:
            completion = truncate_completion(out)
            program = build_program(prob, completion)
            try:
                ast.parse(prob["prompt"] + completion)
                syntax_ok = True
            except SyntaxError:
                syntax_ok = False
            passed, log = run_program(program, timeout=args.timeout)
            rec["samples"].append({
                "completion": completion, "syntax_ok": syntax_ok,
                "passed": passed, "log": None if passed else log,
            })
            done += 1
        c = sum(s["passed"] for s in rec["samples"])
        rec["passed_count"] = c
        results.append(rec)
        status = "PASS" if c else ("syntax-only" if any(s["syntax_ok"] for s in rec["samples"]) else "fail")
        print(f"[{pi+1}/{len(problems)}] {prob['task_id']:<18} {c}/{args.samples} {status} "
              f"({(time.time()-t0)/done:.1f}s/item)")

    n = args.samples
    pass1 = sum(r["passed_count"] for r in results) / (len(results) * n)
    syntax_rate = sum(s["syntax_ok"] for r in results for s in r["samples"]) / (len(results) * n)
    summary = {
        "checkpoint": args.checkpoint,
        "step": step,
        "num_tasks": len(results),
        "samples_per_task": n,
        "temperature": args.temperature,
        "pass@1": pass1,
        "syntax_valid_rate": syntax_rate,
        "num_passed": sum(r["passed_count"] for r in results),
        "num_generations": len(results) * n,
        "elapsed_sec": time.time() - t0,
    }
    for k in (5, 10):
        if n >= k:
            summary[f"pass@{k}"] = sum(pass_at_k(n, r["passed_count"], k) for r in results) / len(results)

    if not args.no_ppl:
        try:
            loss, ppl, py_loss, py_ppl, per = val_loss(model, device=args.device)
            summary.update({"val_loss": loss, "val_ppl": ppl,
                            "val_loss_py": py_loss, "val_ppl_py": py_ppl,
                            "val_per_shard": per})
        except Exception as e:
            summary["val_error"] = str(e)

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
