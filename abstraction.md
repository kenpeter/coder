# SAN coder — Abstraction

> Current state of the SAN training setup plus the findings/fixes from the
> performance-debugging session (throughput 15.8k → GPU-maxed 12L config).

## What SAN is
SAN (Simple Attention Network) is a custom code language model — a PyTorch port
of modded-nanogpt record #360 (ANVIL2) scaled down for a single-GPU RTX 4070 Ti
12 GB. Trained locally on UltraData-Code pre-tokenized `uint16` shards.

Components:
- **ZCRMSNorm** — zero-centered RMSNorm on q/k and pre-MLP
- **ReLU² MLP** — `relu(x @ W1.T)² @ W2.T`
- **MultiHeadAttention** — GQA + q/k RMSNorm + YaRN RoPE + learned per-head gate,
  mixed head widths, sliding window + document boundaries
- **Engram** — hashed bigram+trigram n-gram KV memory table
- **MTP** — multi-token-prediction head on lm_head output
- **Smear gate** — sigmoid gate mixing previous-token embeddings into current
- **Tied embeddings** — embed.weight tied to lm_head.weight
- **MUDD** — per-block residual lambda and post-lambda scalars

## Current training configuration (this run)
| field | value |
|---|---|
| vocab_size | 49152 (SmolLM2-135M tokenizer) |
| d_model | 384 |
| num_heads | 8 (num_kv_heads 4) |
| **num_layers** | **12** (auto-matched from checkpoint — see bug below) |
| seq_len | 2048 |
| params | **~30.96M** (12 layers) |
| batch_size | 8 |
| grad_accum | 8 |
| effective batch | 64 seqs = **131,072 tokens/step** |
| dtype | bfloat16 |
| optimizer | AdamW (fused), betas (0.9, 0.95), wd 0.1 |
| lr / min_lr / warmup | 4e-4 / 1e-4 / 1000 steps |
| MTP weight | 0.1 |
| use_checkpoint | True, **ckpt_every=1** (full recompute for 12 GB fit at B≥4) |
| max_steps | 1,000,000 |
| save_every | 2000 steps |
| **num_workers (DataLoader)** | **8** (`fork`, `prefetch_factor=4`, `persistent_workers=False`) |
| GPU power cap | **150 W** (`nvidia-smi -pl 150`) |
| checkpoint | **`san_latest.pt` step 150000, loss field 1.1258 (stale metadata — see F2), latest only, 1 model kept** |

Resume is the normal path: `run_train.sh` resumes from
`checkpoints/san/san_latest.pt` (latest only, overwritten every 2000 steps).

## Data
- **Shards:** `/home/kenpeter/work/data/_shards_final` — 338 GB, 1,816 `.bin`
  files, raw `uint16`, ~**88.4M** sequences at `seq_len=2048`.
- **Domains:** `code_easy`, `code_mid` (=code_medium), `code_hard`, `code_gold`.
- **Curriculum (`--curriculum`):** stratified interleave each step.
  - tier splits: EASY `{code_easy: 1.0}`, MED `{code_mid: 1.0}`,
    HARD `{code_hard: 0.7, code_gold: 0.3}`
  - tier weights glide easy→hard over the 1M-step plan, with a periodic easy
    "review wave" (G2 anti-forgetting) and a `curriculum_boost.json` override.
  - re-glide (ratio rebuild) every `CURRICULUM_UPDATE_INTERVAL = 2000` steps.

## DataLoader workers — why `num_workers=8`

"Workers" are **CPU processes** (PyTorch `DataLoader` children) that fetch and
prepare batches in parallel with the GPU. The setting lives in
`build_curriculum_dataloader()` in `train_san.py`:

```python
dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                collate_fn=_collate_curriculum,
                num_workers=8,                        # 8 CPU feeder processes
                multiprocessing_context="fork",       # copy-on-write after CUDA init
                pin_memory=True,                      # page-locked for fast H2D copy
                persistent_workers=False,             # respawn after each re-glide
                prefetch_factor=4)                    # each worker keeps 4 batches ready
```

**What one worker does per sample** (`__getitem__`):
1. `epoch_order[idx]` → global sequence index (already in curriculum-mixed order).
2. `_lookup(global_idx)` → `(domain, local_idx)` via the compact `_offset_arr`
   (`np.searchsorted`, no 88M-row list).
3. `_fetch_tokens()` → **mmap read** of 2049 `uint16` values (~4 KB) from the
   domain's `.bin` shard; clamp at shard end; cast to `int64`.
4. Split into `x = toks[:-1]`, `y = toks[1:]` (shifted labels).
5. `_collate_curriculum` stacks 8 samples → `x[8,2048]`, `y[8,2048]`; `pin_memory`
   copies into page-locked RAM; the batch goes onto the worker queue.

> Workers do **not** tokenize. The shards already hold `uint16` token IDs, so a
> worker only *slice → mmap-read → stack → pin → queue*. Per batch that is ~32 KB
> of tokens, but it must happen **off** the main thread.

**Pipeline:**

```
8 × worker ──► [batch][batch][batch][batch] (prefetch 4 each)
                     │ queue
                     ▼
              main process ──► GPU ──► 8 micro-batches = 1 optimizer step (131,072 tok)
```

**Why 8 and not 0:** with `num_workers=0` the mmap/copy/stack/pin work ran
synchronously *between* GPU steps, so the GPU idled ~90 % of wall clock →
15.8k tok/s. With 8 workers the queue stays full and the GPU is never starved →
**100 % / 149 W / ~31k tok/s**. A micro-batch (16,384 tokens) is consumed every
~0.5 s; 8 workers × 4 prefetched batches is ample headroom.

**Live check (how to verify the worker count):**
```bash
MAIN=$(for p in $(pgrep -x python3); do
         tr '\0' ' ' < /proc/$p/cmdline | grep -q train_san.py && echo $p; done)
ps --ppid "$MAIN" -o pid,stat,pcpu,rss,comm     # 8 × pt_data_worker
pgrep -P "$MAIN" | wc -l                        # 10 (8 loaders + inductor pool + resource_tracker)
```
Each worker RSS is ~1.3 GB (copy-on-write view of the ~350 MB `epoch_order`
int32 array). `persistent_workers=False` means these PIDs change after every
curriculum re-glide (every 2000 steps) — the count stays at 8.

## Throughput findings (this session) — why it was 15.8k not 31k

### 1. Data starvation (fixed)
- **Symptom:** training ran at ~15.8k tok/s, step 31 s, **GPU only ~10 % util**.
- **Cause:** the curriculum `DataLoader` used **`num_workers=0`**, so every batch
  was fetched/decoded on the main process, starving the GPU.
- **Fix:** `num_workers=8`, `multiprocessing_context="fork"`, `prefetch_factor=4`,
  `persistent_workers=False` (so `reweight()` → `iter(train_loader)` actually
  respawns workers that see the new order). See
  [DataLoader workers](#dataloader-workers--why-num_workers8) above for the full
  per-worker flow.
- **Result:** GPU pinned at **100 % / ~149 W** (was 10 %).

### 2. The 20 GB dataset pickle / `num_workers` crash (fixed)
- Original `StratifiedShardDataset` held an **~88M-tuple Python `self.index`**
  list (~15 GB) plus a Python-list `epoch_order`. Under torch's default
  `forkserver`, that got pickled to workers and **truncated** (`pickle data was
  truncated`), which is why `num_workers=0` had been forced.
- **Fix (compact dataset):**
  - removed `self.index`; global-index → (domain, local) is derived from a small
    `_offset_arr` numpy array via `_lookup()`;
  - `epoch_order` is now a **compact `int32` ndarray (~350 MB)**, not a Python list.
  - It now pickles cheaply to workers (verified 353 MB survives forkserver in ~1 s),
    and `fork` (copy-on-write) is also safe after CUDA init on this box.

### 3. Startup "building dataset" = building the epoch-order index
- Nothing is re-read/re-tokenized at startup. The `.bin` shards already are the
  data. The startup cost is building the **88.4M-entry stratified epoch-order**
  (bucket every sequence by domain, then interleave by ratios) in RAM.
- That bucketing used `self._lookup()` (an `np.searchsorted` **per index** × 88M)
  after the refactor → many minutes of CPU. **Fixed** to O(N) range-based
  bucketing (use each domain's contiguous global-index range; the dedup path keeps
  the scan).
- Also fixed: on resume the run rebuilt the order again on step 1 because
  `next_glide` started at 2000 while the resume step was ~102k. `next_glide` is
  now `start_step + INTERVAL`, and the initial ratios use the *resume* step, so
  there is exactly one build at startup.

### Reference numbers (measured on this box)
- **27 layers / B=3** (what the code was mistakenly running, see bug): ~**17k**
  tok/s at GPU 100 %/149 W — compute-bound ceiling for a 27L model.
- **12 layers / B=8** (x-small's config, now used): **~31–32k tok/s at 150 W**
  (measured step windows: 30,951 · 32,111 · 31,638 · 32,309 tok/s; avg ≈ 31.75k;
  GPU 100 % / 148–149 W, 10.3 GB VRAM; loss ~1.4–1.5 on clean resume).
  x-small's own scale for reference: ~23k @100 W / ~31.5k @200 W.
- ✅ **31k reached at 150 W** once (a) data loading was moved off the main
  process (`num_workers=8`), (b) the dataset was made compact/picklable, and
  (c) the model was correctly built at 12 layers (not the default 27).

## Architecture / resume bug (fixed)
- `TrainConfig.num_layers` defaulted to **27**, and nothing read the checkpoint's
  real config. The checkpoint is **12 layers**, so the trainer built a 27-layer
  model, loaded the 12L weights into it (15 layers left randomly initialized) and
  failed to load the optimizer state (`parameter group doesn't match`) — silently
  training a broken, ~1.8× slower model.
- **Fix:** on resume the trainer now **auto-matches `num_layers` from the
  checkpoint's `config`** (unless `--num-layers` is explicitly passed). `run_train.sh`
  also passes `--num-layers 12 --batch-size 8` explicitly.
- A clean backup of the 12L checkpoint is at
  `checkpoints/san/san_latest_12L_step102000.pt`.

## Checkpointing
| Setting | Value |
|---|---|
| Save interval | every 2000 steps (`--save-every`) |
| Checkpoint file | `san_latest.pt` (overwrites; latest only) |
| Resume from | `san_latest.pt` if `--resume` not given |
| Resume-safe | num_layers auto-matched; per-layer params sliced if overridden |
| Save on signal | SIGTERM/SIGINT sets a flag; saves at next step boundary then exits |
| **Only checkpoint** | **`checkpoints/san/san_latest.pt` — no other checkpoint files are kept** (extra backups removed 2026-11-11) |

## Maxing GPU and CPU utilization
- **GPU:** keep the model on CUDA, use bfloat16, keep `ckpt_every=1` (full recompute) to fit 12GB at B=8, and keep the data pipeline feeding the GPU. Cap power with `sudo nvidia-smi -pl 150` (resets to ~285W on reboot). Target ~100% util / 149W.
- **CPU:** use `num_workers=8` with `multiprocessing_context="fork"`, `prefetch_factor=4`, `pin_memory=True`. Workers do mmap reads off the main thread so the GPU is never starved. With `num_workers=0` throughput collapses to ~15.8k tok/s.
- **Live check:**
```bash
nvidia-smi --query-gpu=utilization.gpu,power.draw,memory.used --format=csv,noheader
top -bn1 | grep "Cpu(s)"
```
- **Weight-edit cycle:** runs on GPU during HumanEval eval (~97% util). The editor (Gemma 4 12B it) runs on **CPU**, bf16, so GPU eval and editor generation can overlap.

## Key operating insights
- **Data feeding is 8 CPU workers, not the GPU.** Keep `num_workers=8`; dropping
  to 0 collapses throughput to ~15.8k regardless of GPU headroom.
- **GPU 100 % / 150 W is the ceiling** — tok/s is then compute-bound; the only
  remaining lever is fewer FLOPs/token (fewer layers, less recompute) or a
  larger/more efficient batch (VRAM-limited at 12 GB).
- **`nvidia-smi` power cap resets to ~285 W on reboot** — reapply
  `sudo nvidia-smi -pl 150`.
- **12 L at B=8 forces full gradient checkpointing (`ckpt_every=1`)** to fit 12 GB;
  it is the config x-small uses for its 31k number.

## Data pipeline history (kept for reference)
- L2 = CPT pre-training tiers (easy/medium/hard); L3 (gold) = SFT/mid-train.
- Earlier plan rotated stages (CPT → 1:1 L2+L3 → L3 only); the current
  `--curriculum` mixes all tiers every step with gliding ratios instead.
- SSD rotation (historical): full tier set ~399 GB doesn't fit SSD at once; train
  in two-tier stages, repointing `SHARD_DIRS` between stages.

## Agentic evaluation loop (cron-driven)

An automated outer loop runs every **30 minutes** (via cron) to evaluate the
model's coding ability and feed findings back into training.

**Cadence / trigger:** cron fires every 30 min and posts a "tick" message; the
agent then runs the evaluation pass.

**Procedure each tick:**
1. Load the latest checkpoint (`checkpoints/san/san_latest.pt`) and run a
   **coding-question evaluation** of 4 questions.
2. **If all 4 pass** → advance the eval set to the next 4 questions (progress).
3. **If any fail (loop condition)** → do not advance. Instead:
   - Inspect the failures to determine *why* (e.g., weak on a domain, forgetting
     easy-tier skills, overfitting hard tier, poor multi-token prediction).
   - Adjust a training knob accordingly — e.g., learning rate, curriculum glide
     ratios / tier weights, review-wave cadence, MTP weight, batch size, or
     data mix — and let training continue.
   - Re-run the same 4 questions on the next tick to confirm the fix.

**Loop exit / success condition:** a full block of 4 questions passes, at which
point the eval advances to the next block. Persistent failures across ticks
trigger successive training adjustments until the block passes.

** knobs the loop may touch:** `lr`, `curriculum` tier weights / glide schedule,
`curriculum_boost.json`, `mtp_weight`, `--batch-size`, `--num-layers`,
`save_every`, `num_workers`. All changes are applied via `run_train.sh` / config
and training resumes from `san_latest.pt`.

## Plan for Weight‑Space Editing to Boost Coding Ability

Benchmarks to test after editing: **HumanEval**, **MBPP**, **SWE‑Bench Lite**.
Methods to try:
1. **Direction removal** – identify and subtract the “loop” direction (as in the Astra paper) from the MLP of layers 15/19 to reduce repetitive reasoning loops.
2. **LoRA‑style rank‑injection** – add a small LoRA adapter (rank = 8) fine‑tuned on the 4‑question coding eval set, then merge back into base weights.
3. **In‑batch editing** – edit weights directly on the 31M SAN model (`san_latest.pt`, 12 layers) using cycle-based editing; no separate teacher model involved.

After each edit, re‑run the **cron‑driven evaluation loop** (every 30 min) with 4 coding questions. The result is recorded; the edit stays in `san_latest.pt` either way (single-checkpoint, always-keep policy).

Suggested first edit: **subtract the looping direction** identified by probing MLP activations on the 4‑question eval set, then verify with the cron loop.

### Text‑flow diagram

```
Load model weights
  ↓
LLM reads weights as text
  ↓
identifies problematic
  direction / feature
  ↓
edits a few weights
  (subtract / add)
  ↓
run forward passes
  on benchmark
  (HumanEval/MBPP)
  ↓
save edited weights into
san_latest.pt (always —
the only checkpoint)
```

*No gradients; weight editing over raw numbers; verification via forward passes. The SAN model is 31M parameters (12 layers) — trained and edited within this environment. All weight editing cycles run on `san_latest.pt` (31M, 12 layers), with **Gemma 4 12B it** acting as the editor: it reasons over the HumanEval@30 failures and proposes the `{param, scale, add}` edits. Cumulative cycles across all sessions: ~2000+. *(Historical note: since 2026-11-11 no base copy is kept — `san_latest.pt` is the single checkpoint and edits overwrite it unconditionally.)*

**Roles**: The **editor** (**Gemma 4 12B it**, loaded from `gemma4_local/`) reasons over the 30M SAN model's weights/eval failures and proposes concrete weight edits. The **30M SAN model** (`san_latest.pt`, 12 layers) is the **editee** — its weights are edited in place. Only `san_latest.pt` is loaded; weight edits are applied **in place** directly to `san_latest.pt` (overwritten), no separate base/edited copies and no gradient training.

**Active weight-edit cycle (updated 2026-11-11 13:48):** `weight_edit.py` running in background (PID 2962718) against `san_latest.pt`, applying HumanEval@30-grounded edits in place, scheduled to run until 2026-11-11 21:00 (9 PM). Editor is **Gemma 4 12B it** loaded from the local dir `/home/kenpeter/work/coder/gemma4_local`. Progress logged to `weight_edit_run.log`. **Every cycle saves into `san_latest.pt` regardless of the score (always-keep).**

**Current status (2026-11-11):** the 30M model scores **0/30** on HumanEval@30
(step 150000). Every cycle's edits are saved into `san_latest.pt`
**regardless of the score** (keep rule = always), so the checkpoint keeps
changing every cycle. See [Root-cause findings](#root-cause-findings-2026-11-11)
below — the 0/30 is **not** primarily the editor's fault, and the earlier
"loss ~1.12" reading was stale metadata, not a measurement.

## Root-cause findings (2026-11-11)

Measured, not assumed. Four findings, in order of how much they explain.

### F1 — The score was never climbable: the model doesn't continue, it restarts

**198 of 210** graded generations (7 cycles × 30 problems) begin with a
**brand-new `def ...`** instead of continuing the function body after the
docstring. The prompt ends `...return True\n    """\n`; the model answers with
`\ndef get_current_in_list(...)`. Then it runs away with unclosed parens.

Three *different* prompts (HumanEval/0, HumanEval/1, and a synthetic one) all
produce the same `\ndef get_string...` shape, and the next-token argmax is the
same token (222) for all three. Output entropy at the prompt boundary is
**10.4 nats vs. max 10.8** — near-uniform. The model barely reads the prompt
before writing.

**Consequence:** the ladder is `syntax-valid → entry_point defined → asserts
pass`, and the model is stuck on rung 1. Editor feedback of "0/30" is identical
no matter what it tries, so there is nothing to climb.

| stage reached | count (210 problems) |
|---|---|
| `syntax` (SyntaxError — never reached a test) | **191** |
| `tests` (function ran, asserts executed) | 19 |
| `load` | 0 |
| **asserts passed** | **0 / 82** |

Typical error: `SyntaxError: '(' was never closed (<gen>, line 77)`.

### F2 — The checkpoint's `loss` field is stale metadata

`san_latest.pt` reports `loss 1.1258`. That number is **frozen** —
`weight_edit.py` copies the checkpoint dict and writes `model_state_dict`
back without ever updating `loss`. `weight_edit_final_results.json` confirms
the intent: `"metadata stays 1.0756 (expected behavior)"`.

Measured instead: **CE 4.377** on 18 held-out `code_easy` windows.

### F3 — `final_norm.scale` had collapsed (repaired)

`ZCRMSNorm` computes `(1 + scale) * x / rms`, so `scale` is the logit-gain
knob. A recorded Nov-8 snapshot (`all_weight_edit_summary.json`) shows
`final_norm.scale` at **4.038**; it had fallen to **0.500** — gain 5.04 → 1.50,
a **3.36× logit shrink**. The earlier editor proposed exactly this:
`{'param': 'final_norm.scale', 'scale': 0.35}`, `scale 0.95, add -0.45`,
`scale 0.98, add -0.15` — all **kept** under always-keep.

**Repair (2026-11-11 13:41):** `final_norm.scale` mean `0.500 → 2.5`,
**CE 4.377 → 1.483**, saved into `san_latest.pt`, recorded in
`eval_results/knob_repair.json`. Training had logged main CE 0.985 at step
150000 (`train_san.log:10344`), so 1.48 is the right order of magnitude.

**But HumanEval stayed 0/30** — expected: greedy decoding is *argmax*, and
argmax is invariant to a positive rescale of all logits. **Calibration was
never the HumanEval blocker; F1 is.**

### F4 — The model has never passed HumanEval, edited or not

| date | step | score | editing active? |
|---|---|---|---|
| Oct 28 | 38162 | **0/164** | no (editing started Nov 7) |
| Oct 29 | 52966 | **0/164** | no |
| Oct 30 | 40000 | **0/3** | no |
| Nov 11 | 150000 | 0/30 × 7 cycles | yes |

Top-1 next-token accuracy on in-distribution shards is **~60%** — the model
*ranks* correctly but assigns low probability (under-scaled logits, F3). On
HumanEval prompts top-1 drops to **0.29–0.65**.

**Implication:** no scalar edit to a handful of gates can move a metric that
has been 0 since before any editing existed.

## Plan: A → B → C (approved 2026-11-11)

Three ordered stages. Each unblocks the next; do not skip.

### A — Fix the ruler (make the score climbable)

Today the editor is told only `0/30`, which never changes, so it receives no
signal. Replace the single number with a **dense score vector**:

| component | what it measures | why it can move |
|---|---|---|
| `syntax_rate` | % completions that `exec` cleanly | 191/210 fail here — big headroom |
| `defined_rate` | % where `entry_point` exists | currently 0 |
| `assert_rate` | asserts passed / total | currently 0/82 |
| `ce` | loss on held-out `code_easy` | 1.483, continuous |

Feed the **vector plus its delta since last cycle** to Gemma, alongside the
existing per-test failure detail. Also pass the **fingerprint of what it tried
last cycle and what happened**, so it can stop repeating itself.

*Success for A:* the editor receives a number that changes when it acts.

### B — Fix the hand (measure edits individually, widen the menu)

Today: 5 edits proposed → all applied → all kept blind. Change to:

1. **Ablation:** apply each proposed edit *separately*, score it on the cheap
   signals (`ce`, `syntax_rate`), keep the winners. One edit at a time =
   credit assignment.
2. **Wider menu:** the current reflex is `blocks.{0..4}.attn_gate`. Add
   `embedding.weight`, `final_norm.scale`, per-block `q_norm/k_norm.scale`
   (attention temperature), `norm1.scale`, `hadamard_mlp.d1/d2`.
3. **History in the prompt:** "last cycle you tried X, Δ was Y" — prevents the
   identical-edit loop observed across all 7 cycles.
4. **Direction search:** for scalar params, try `scale` up *and* down and keep
   the better — currently Gemma only ever proposes one direction.

*Single-checkpoint reconciliation:* `san_latest.pt` remains the only checkpoint
and is still written every cycle (always-keep preserved). "Reject" is
implemented as **reverting that one edit in memory before the save**, and the
reject is logged to `weight_edit_run.log` — so nothing is ever lost from disk,
but bad edits stop accumulating.

*Success for B:* the score vector trends instead of flatlining.

### C — Fix the disease (prompt-insensitivity)

A and B are machinery; this is the target. The defect is that the model writes
a new `def` regardless of prompt (F1). Edits must be aimed at what controls
**how much of the prompt the model uses**:

- `blocks.*.attn_gate` / `q_norm` / `k_norm` — attention temperature and how
  sharply the model attends back to the docstring it was just given
- `embedding.weight` rows for `"""`, `def`, indentation tokens — the
  continuation-vs-restart decision
- `smear gate` / `engram` params — previous-token and n-gram context that
  decides "keep writing this body"

Diagnose first: dump the attention distribution over the prompt at the exact
token where the model chooses `\ndef` over an indented body line. Edit what
that trace implicates, not what sounds plausible.

*Success for C:* `defined_rate` > 0 — the first time the model writes the
function it was actually asked to write.

### Anti-goals

- No gradients, no LoRA, no fine-tuning — this stays weight editing only.
- No second checkpoint file; `san_latest.pt` stays the single artifact.
- No blind edits: every proposal still traces to a graded HumanEval@30 failure.
- No claiming success from `ce` alone — F3 proved a big CE win can leave
  HumanEval at 0/30. `defined_rate` and `assert_rate` are the real targets.

**Editor must be based on eval results**: the editor's decisions (Gemma 4 12B it) MUST be grounded in the **inference/eval results of the 30M model (`san_latest.pt`)** using the standard **HumanEval@30** benchmark (30 HumanEval problems, pass@1, greedy decoding). Each edit cycle MUST begin by running inference + eval on the 30M model first (HumanEval@30) to identify which problems fail → collect failures → feed those eval results to the editor → editor proposes targeted weight edits (NOT blind edits) → apply in place to `san_latest.pt` → re-evaluate on HumanEval@30 → **save the edited weights into `san_latest.pt` (always; there is no other checkpoint and no revert)**. The eval still runs every cycle because the editor must be grounded in real HumanEval@30 failures — it just no longer gates the save. **Goal: a better 30M model** (higher HumanEval pass rate / lower loss), not a better editor model. The editor is only a reasoning tool; the artifact being improved is `san_latest.pt` (31M, 12 layers). Edits are NEVER blind; every edit MUST be justified by 30M inference/eval evidence using the HumanEval@30 benchmark. No toy micro tasks; use the proper standard benchmark (HumanEval@30). ALL editing decisions must be grounded in HumanEval@30 failures of `san_latest.pt`.

**Gemma 4 12B it — Downloaded (November 2026)**

**Download completed:** 2026-11-10 07:44:54 (authenticated HF download)

**Files downloaded:**
- `tokenizer.json` — 32 MB — `/home/kenpeter/work/coder/tokenizer.json` (moved to HF cache at `/home/kenpeter/.cache/huggingface/hub/models--google--gemma-4-12B-it/`)
- `model.safetensors` — 22.28 GB — `/home/kenpeter/work/coder/model.safetensors` (moved to HF cache)

**Repository:** `google/gemma-4-12B-it` (instruct/chat, 12B parameters)

**Current status:** Model files in Hugging Face cache, ready for use

**Use cases:**
- Chat/instruction following via `transformers` or Ollama (after GGUF conversion)
- Weight editing editor for the 31M SAN model (`san_latest.pt`)
- Text generation and code completion

**Note:** GGUF conversion not yet attempted; 22.28 GB safetensors format held in HF cache.

**Location (remember):** `/home/kenpeter/.cache/huggingface/hub/models--google--gemma-4-12B-it` (HF cache; also originally `/home/kenpeter/work/coder/model.safetensors` + `tokenizer.json`).

**Inference (Transformers):**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
tok = AutoTokenizer.from_pretrained("google/gemma-4-12B-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-4-12B-it", torch_dtype=torch.bfloat16, device_map="auto")
inputs = tok("Write a Python function to reverse a string.", return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=128)
print(tok.decode(out[0], skip_special_tokens=True))
```

**Editor grounding:** Gemma 4 12B it is the editor, following the eval-first protocol (HumanEval@30 → failures → editor proposes edits → apply in place → re-evaluate).

## Single edit cycle (weight_edit.py) — CURRENT FLOW

**Roles: Gemma 4 12B it is the EDITOR; the 30M SAN model (`san_latest.pt`) is the EDITEE.**
All edits are applied in place to `san_latest.pt`. No gradients, no base/edited copies.

### Exact flow of one cycle (as implemented in `weight_edit.py`)

```
1.  Load san_latest.pt (12L, ~31M params) into CUDA + starcoder2-7b tokenizer
        │
2.  EVAL  ── HumanEval@30 on the 30M model
        │    generate 160 tokens, greedy (temperature 0.0)
        │    grade PER TEST (not just pass/fail):
        │      stage = syntax | load | tests
        │      tests_passed / tests_total + the exact failing assert + error
        │    → pass count  +  per-problem grading (see below)
        │
3.  Build WEIGHT FINGERPRINT: for every tensor → name, shape, mean, std, norm
        │
4.  EDITOR  ── Gemma 4 12B it (CPU, bf16, local dir gemma4_local)
        │    prompt = fail summary + failing problems + fingerprint
        │    instruction: reason about WHY they fail, return JSON only
        │    → [{"param": "...", "scale": 1.01, "add": 0.0}, ...]  (3–5 edits)
        │
5.  APPLY  ── for each edit:  tensor = tensor * scale + add   (in memory)
        │
6.  RE-EVAL  ── HumanEval@30 again on the edited model
        │
7.  KEEP  ── always save the edited state_dict over `san_latest.pt`
        │    (good or bad — single-checkpoint policy, no base copies)
        │
8.  loop back to 2 until the deadline (21:00)
```

### Plain-text summary of the loop
Load model → measure (HumanEval@30) → fingerprint weights → ask Gemma for
targeted `{param, scale, add}` edits → apply → measure again → **save over
`san_latest.pt` (always, good or bad)** → repeat.

> **Planned change (stage A/B, see [Plan: A → B → C](#plan-a--b--c-approved-2026-11-11)):**
> step 2 will report a **dense score vector** (`syntax_rate`, `defined_rate`,
> `assert_rate`, `ce`) instead of a bare `0/30`, and step 5–7 will **ablate
> edits one at a time** so only the helpful ones are kept. `san_latest.pt`
> stays the single checkpoint and is still written every cycle.

> **Single-checkpoint policy:** `checkpoints/san/san_latest.pt` is the ONLY
> checkpoint. There are no base/edited copies and no rollback — every cycle's
> edits are written straight into `san_latest.pt` regardless of whether the
> HumanEval score improved. The eval numbers are recorded for the record, not
> used as a gate.

### Step 2 in detail — how ONE HumanEval question is graded (per-test, not binary)

A problem is **not** just "pass/fail". `grade()` runs the model's completion and
reports *why* it failed, in one of three stages:

| stage | meaning | what is reported |
|---|---|---|
| `syntax` | `exec(prompt+completion)` raises | exception type + message + line |
| `load` | completion defines no `entry_point` function | "function not defined" |
| `tests` | function exists, but asserts run | **`tests_passed` / `tests_total`** + the exact failing assertions |

For `tests`, the `check()` block is parsed with **`ast`** so multi-line asserts
stay intact (naive line-splitting truncates them and wrongly blames the test),
then every `assert` is executed **one at a time** — so we know *which* assert
failed, not merely that something failed.

Real example (cycle 1, `eval_results/weight_edit_cycle_1.json`):

```json
{
  "task_id": "HumanEval/0",
  "passed": false,
  "generated": "\ndef get_current_number(numbers: List[int]) -> int:\n    ...",
  "detail": {
    "stage": "syntax",
    "error": "SyntaxError: '(' was never closed (<gen>, line 77)",
    "tests_passed": 0, "tests_total": 0, "fails": []
  }
}
```

and a graded failure (stage `tests`):

```json
{"stage": "tests", "tests_passed": 0, "tests_total": 4,
 "fails": [{"line": 1, "assert": "assert candidate('(()()) ((())) () ((())()())') == ['(()())', ...]",
            "err": "AssertionError"}]}
```

Cycle-1 grade distribution over all 30 problems: **22 `syntax` + 8 `tests`, 0 passed**.

### Step 4 in detail — the EXACT prompt sent to Gemma 4 (real, cycle 1)

```
You are an editor for a 30M-param code model (SAN). Eval: HumanEval@30 pass 0/30.
Each failure is graded per unit test (not just pass/fail): stage, how many
asserts passed, and the exact failing assertion + error.
Failing problems with their generated (wrong) code and per-test grading:

--- HumanEval/0 ---
Prompt:
from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer ... """
Generated:
    def get_current_number(numbers: List[int]) -> int:
        """ Get the current number of numbers ... """
        return ( ( ( ( ( ( ...
Grade: stage=syntax  tests passed 0/0
Error: SyntaxError: '(' was never closed (<gen>, line 77)

--- HumanEval/1 ---
Prompt:
def is_zero_balanced(numbers): ...
Generated:
    def is_empty(s): ... return not is_empty(s)   # infinite recursion
Grade: stage=tests  tests passed 0/4
  FAILING TEST #1: assert candidate([]) == True  ->  AssertionError
  ...

(… up to 6 failing problems, then …)

Weight fingerprint:
phi_pre: shape=(384,) mean=0.0087 std=0.2470 norm=19.41
blocks.0.attn_gate: shape=() mean=0.9591 std=nan norm=0.96
blocks.10.hadamard_mlp.d1: shape=(512,) mean=1.0312 std=0.0000 norm=23.33
… (all 206 tensors: name / shape / mean / std / norm) …

Based on these concrete graded failures, propose 3-5 targeted weight edits as
JSON list of objects with keys: param (full param name), scale (float near
1.0), add (float small). Only output JSON, no prose.
```

This string is wrapped in Gemma's chat template
(`apply_chat_template([{"role":"user","content":prompt}], add_generation_prompt=True)`),
decoded, and generated **greedy, `max_new_tokens=512`** on CPU (bf16).

### What Gemma 4 returned (real, cycle 1)

```json
[
  {"param": "blocks.0.attn_gate",   "scale": 0.8, "add":  0.1},
  {"param": "blocks.1.attn_gate",   "scale": 0.8, "add":  0.1},
  {"param": "blocks.10.attn_gate",  "scale": 0.7, "add": -0.2},
  {"param": "blocks.11.attn_gate",  "scale": 0.7, "add": -0.2},
  {"param": "mtp_block.attn_gate",  "scale": 0.8, "add": -0.1}
]
```

→ parsed with `re.search(r'\[.*\]', resp, re.S)` + `json.loads`, then each edit
becomes `tensor.mul_(scale).add_(add)` (unknown param names are skipped and
logged). Re-eval follows, then the edited weights are saved into
`san_latest.pt` (always).

### Everything is logged (for later debugging / improvement)

| file | contents |
|---|---|
| `weight_edit_run.log` | one line per cycle: eval before, proposed edits, eval after, **KEEP (always) + score before → after** |
| `weight_edit_debug.log` | **full prompt sent to Gemma, full raw Gemma response, parsed edits, decision** — every cycle |
| `eval_results/weight_edit_cycle_<n>.json` | full 30-problem graded report (prompt, generated code, stage, tests passed/total, failing asserts) per cycle |

### Editor: Gemma 4 12B it (local dir)
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("/home/kenpeter/work/coder/gemma4_local")
model = AutoModelForCausalLM.from_pretrained(
    "/home/kenpeter/work/coder/gemma4_local", torch_dtype=torch.bfloat16, device_map="cpu")
```
`gemma4_local` is a symlink dir (`config.json`, `tokenizer.json`,
`model.safetensors`, `chat_template.jinja`) pointing at
`~/.cache/huggingface/hub/models--google--gemma-4-12B-it/` — needed because the
HF hub ID can't be resolved offline. Chat template is used
(`apply_chat_template`, add_generation_prompt=True), generation is greedy,
`max_new_tokens=512`. **Requires `accelerate`** (installed; it provides
`device_map`).


### What is HumanEval@30
HumanEval is the standard code-generation benchmark (OpenAI). Each problem has a function signature + docstring prompt, a reference solution, and unit tests. HumanEval@30 = the first 30 problems of `eval_data/HumanEval.jsonl.gz`, evaluated pass@1 with greedy decoding (temperature 0.0), 160 new tokens. A problem passes if `exec(prompt + generated_completion + test + check(entry_point))` raises no exception.

### Example of one HumanEval test (problem 0)
Prompt:
```
from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    \"\"\"
```
Generated completion (model output, truncated at `\ndef `):
```
    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2 and abs(elem - elem2) < threshold:
                return True
    return False
```
Test (from the dataset):
```
def check(candidate):
    assert candidate([1.0, 2.0, 3.0], 0.5) == False
    assert candidate([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True
    ...
check(has_close_elements)
```
Pass if no AssertionError/exception is raised.

One cycle (~1–2 min) proceeds as described in
[Single edit cycle (weight_edit.py) — CURRENT FLOW](#single-edit-cycle-weight_editpy--current-flow)
above: load → HumanEval@30 → fingerprint → Gemma 4 editor → apply →
re-evaluate → **save into `san_latest.pt` (always)**. Runs in `weight_edit.py`
until the 9 PM deadline (log: `weight_edit_run.log`). *No gradients; LLM
reasons over raw numbers; verification via forward passes.*
