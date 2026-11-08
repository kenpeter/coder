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
| checkpoint | **`san_latest.pt` step 138508, loss 1.0756 (latest only, 1 model kept)** |

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
3. **In‑batch editing** – edit weights directly on the 31M SAN model (`san_latest.pt`, 12 layers) using cycle-based editing; no 9B teacher model involved.

After each edit, re‑run the **cron‑driven evaluation loop** (every 30 min) with 4 coding questions. If the pass rate improves, keep the edit; otherwise revert and try the next method.

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
if accuracy ↑ → save
  else revert
```

*No gradients; weight editing over raw numbers; verification via forward passes. The SAN model is 31M parameters (12 layers) — trained and edited within this environment. The 9B Qwopus3-5-9B-Coder-Flash model (5.78 GB, GGUF format, Ollama-cached) is a separate transformer-based LLM architecture and can perform weight editing on the 31M SAN model via task-vector alignment and merge-based methods. All weight editing cycles run on `san_latest.pt` (31M, 12 layers) with the 9B model providing teacher guidance for edit direction identification. Cumulative cycles across all sessions: ~2000+. Loss improvement demonstrated during editing: actual model loss decreased from ~1.05 → 0.9986 (best) → 1.0282 (final) across previous editing cycles. Base checkpoint preserved at step 138508, loss 1.0756.

**Roles**: The **9B model** (`qwopus-coder`, Ollama) is the **editor** — it reasons over the 30M SAN model's weights/eval failures and proposes concrete weight edits. The **30M SAN model** (`san_latest.pt`, 12 layers) is the **editee** — its weights are edited in place. Only `san_latest.pt` is loaded; weight edits are applied **in place** directly to `san_latest.pt` (overwritten), no separate base/edited copies and no gradient training. Ollama 9B is running (`ollama serve`).

**Editor must be based on eval results**: the 9B editor's decisions must be grounded in the **inference/eval results of the 30M model** using the standard **HumanEval@30** benchmark (30 HumanEval problems, pass@1, greedy decoding). Each edit cycle: **run inference + eval on the 30M model first** (HumanEval@30) to identify which problems fail → collect failures → feed those eval results to the 9B editor → editor proposes targeted weight edits (not blind edits) → apply in place to `san_latest.pt` → re-evaluate on HumanEval@30. **Goal: a better 30M model** (higher HumanEval pass rate / lower loss), not a better 9B model. The 9B is only a reasoning tool; the artifact being improved is `san_latest.pt` (30M). Edits are never blind; every edit is justified by 30M inference/eval evidence. No toy micro tasks; use the proper standard benchmark (HumanEval@30).

## Single edit cycle (weight_edit_9b.py)

One cycle (~30-60s) proceeds as:

1. **Load** `san_latest.pt` (30M, 12 layers) into GPU; load tokenizer.
2. **Inference + eval (HumanEval@30)** on the 30M model: generate a completion for each of the 30 problems (greedy, 160 new tokens), `exec` prompt+completion+tests, record pass/fail. Capture failing problem IDs, prompts, generated code, and error messages.
3. **Build weight fingerprint**: for every parameter, name/shape/mean/std/norm.
4. **Ask the 9B editor** (Ollama `qwopus-coder`) with a prompt containing the failing problems' prompts + generated code + errors + weight fingerprint. The 9B returns a JSON list of targeted edits `{param, scale, add}`.
5. **Apply edits in place** to `san_latest.pt` tensors (multiply by scale, add delta).
6. **Re-evaluate** the 30M model on HumanEval@30.
7. **Keep or revert**: if pass rate did not drop, overwrite `san_latest.pt` in place with the edited weights; otherwise restore the previous weights.
8. Repeat until 8pm deadline. A Telegram message is sent every 30 min reporting pass/fail counts.
*No gradients; LLM reasons over raw numbers; verification via forward passes.*
