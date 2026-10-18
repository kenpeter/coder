# SAN x-small — Abstraction

> Abstraction of the project and the agent session that produced this release.

## What SAN is
SAN (Simple Attention Network) is a ~31M-parameter custom language model — a faithful
PyTorch port of the `needle` architecture (arXiv:2607.18363) from JAX/Flax. It is trained
locally on a single **RTX 4070 Ti 12 GB**.

Components (1:1 with the JAX source):
- **ZCRMSNorm** — zero-centered RMSNorm
- **HadamardMLP** — fixed Walsh–Hadamard transform MLP (no dense FFN weights)
- **MultiHeadAttention** — GQA + q/k RMSNorm + RoPE + learned gate
- **Engram** — hashed n-gram KV memory tables with conv taps, injected at sites
- **MHC** — Multi-Lane Hyper-Connections with Sinkhorn routing across 4 residual lanes
- **MTP** — multi-token-prediction head
- Tied embeddings + causal/packing mask

## Trained configuration (this release)
| field | value |
|---|---|
| vocab_size | 49152 (SmolLM2-135M tokenizer) |
| d_model | 384 |
| num_heads / num_kv_heads | 8 / 4 (GQA) |
| num_layers | 12 |
| max_seq_len | 2048 |
| dtype | bfloat16 |
| engram_orders / slots / layers | (2,3) / 8192 / (2,15) → active (2,) |
| mhc_lanes | 4 |
| mtp_weight | 0.1 |

Training: curriculum over **16 domains**, target **100w (1,000,000) steps**,
`--batch-size 8 --grad-accum 8` (eff batch 64). GPU power cap **100 W** (cool) /
**200 W** (sweet spot; ~31.5k tok/s @200W, ~23k @100W).

## Session log (this agent run)
1. **Resume @100W** — `nvidia-smi -pl 100`, resumed from `san_latest.pt` (step 132,706),
   target `--steps 1000000`. Stepping confirmed (~21–23k tok/s).
2. **"Plateau?"** — analyzed eval perplexity (the real signal, not the flat training-loss):
   - step 63,663 → mean ppl **53.69** (code 13.76 / math 36.41 / prose 110.89)
   - step 110,938 → mean ppl **25.38** (code 11.75 / math 30.32 / prose 34.07)
   - step 123,141 → mean ppl **23.74** (code 10.25 / math 29.62 / prose 31.34)
   - Conclusion: **not a plateau** — eval ppl still improving. The flat *training* loss
     (~2.0 since ~105k) is curriculum-mix masking, not model stall. The ~2.0 floor is:
     (a) model-capacity + data-entropy floor for 31M on a diverse 16-domain mix,
     (b) the harder MTP aux head (~2.35), (c) curriculum re-glide sawtooth.
     LR is still near peak (cosine over 1M steps, only 13.5% in) — LR is *not* the cause.
3. **Save + stop + release GPU** — SIGTERM saved checkpoint at **step 137,260**
   (`san_latest.pt`, `✅ done`), released GPU (idle).
4. **Release decision** — HF token provided; decided: **code + this abstraction → GitHub**
   (`kenpeter/x-small`), **model weights + latest `.pt` → HuggingFace** (`kenpeter123/san-xsmall`).

## Key operating insights
- **Eval perplexity is the usable signal** at this scale; generated text loops/decays
  regardless at 31M — do not read sample text as a quality verdict.
- **Training-loss flat ≈ curriculum masking**, not stall — trust eval ppl.
- **Power cap resets to ~285 W on reboot** — reapply after restart.
- **Resume is the normal path**; `--num-layers` is resume-safe (per-layer params sliced).
- **Checkpoint save requires SIGTERM** (saves at next step boundary then exits) — there is
  no save-and-keep-running signal.

## Release layout
- **GitHub `kenpeter/x-small`**: training/inference code (`train_san.py`, `san_model.py`,
  `san_triton.py`, eval scripts), this `abstraction.md`, README, LICENSE.
- **HuggingFace `kenpeter123/san-xsmall`**: `pytorch_model.bin` (model weights),
  `san_latest.pt` (raw checkpoint, step 137,260), `config.json`, `tokenizer/`
  (SmolLM2-135M), `san_model.py` + `load_example.py`, model card.

## Status at release
- Last training step: **137,260** (of 1,000,000 target).
- Last eval (step 123,141): mean ppl **23.74** (code 10.25 / math 29.62 / prose 31.34);
  training loss @ckpt 2.04.
- No eval run between 123,141 and 137,260 — recommend an eval to confirm the trend if
  training resumes.

## Data Pipeline (planned) — code training on UltraData

Goal: convert `/mnt/file_drive/data/ultradata` (1.2 TB, 22 languages, L2/L3 tiers) into
`.bin` uint16 shards consumable by `train_san.py`, and train a code-focused SAN.

### Tokenizer choice (replace SmolLM2)
- **Primary candidate:** `bigcode/starcoder2-7b` tokenizer — vocab **49,152** (drop-in for
  `SANConfig.vocab_size`, no model change), byte-level BPE, trained on The Stack v2
  (600+ languages), FIM-ready. Best code compression in the 49k class.
- **Alternative (smaller):** `TerminatorPower/ezellm-lite-tokenizer` — 24,600 vocab, ~5%
  less compression than StarCoder2 at half the vocab (embedding-table savings for a 31M
  model), FIM + repo-metadata specials pre-allocated. Requires `vocab_size=24600`.
- **Alternative (max compression):** `HusseinEid/cute-tokenizer` — byte-perfect roundtrip on
  Python, but slower encode and custom Rust path; keep as research option.
- Decision rule: stay on 49,152 (StarCoder2) unless embedding memory becomes the bottleneck;
  only then drop to ezellm-lite and resize the model.

### Source data
- 1,121 parquet files, columns: `uuid, repo_name, relative_path, content, category,
  algo_rel_score, quality_score`.
- ~1.2 TB on disk; estimated ~250–400B raw tokens before filtering (sample: 50 files ≈
  36B tokens).

### L2 vs L3 (and how to train on them)
- **L2 — raw algorithmic code** (~400B tokens, 11 langs): selected high-quality
  algorithmically-relevant source files. Use for **continued pre-training (CPT)** of the
  base model. Filter on `algo_rel_score`/`quality_score`/`category`.
- **L3 — task-oriented synthesis** (~150B tokens, 11 langs): each row is a structured
  programming exercise (`task`, `analysis`, `solution`, `test`, `full_content`) generated
  from an L2 seed. Use for **mid-training / SFT** (instruction-style), not as raw code.
- **Published recipe (UltraData):** on a 1B model with 10B CPT tokens, L2 alone beats L1
  by +7.80 EvalPlus / +5.13 MultiPL-E. Replacing **half** of L2 tokens with L3 gives a
  further +8.42 / +8.07, and the 1:1 L2+L3 mix scales to 100B tokens (+10.11 / +12.39).
  A separate SFT recipe (LFM2.5) uses L2 for CPT then L3 for SFT with completion-only loss.
- **For SAN x-small (31M):** plan a two-stage run — (1) CPT on L2-heavy mix,
  (2) mid-train/SFT on L3 (task→solution) — or a single 1:1 L2+L3 mix per the
  published recipe. L3 should be serialized into a prompt format
  (task + analysis + solution + tests) before tokenization.

### Conversion plan
1. **Filter L2** by `quality_score` / `algo_rel_score` / `category` thresholds; dedup by
   `uuid` + content hash. Tier into `code_easy/medium/hard` (+ `code_gold`) by score
   quantiles + language.
2. **Prepare L3** by serializing `task`/`analysis`/`solution`/`test` into a consistent
   instruction template; tokenize as a separate `code_gold`/`code_sft` tier.
3. **Tokenize** both with the chosen tokenizer (StarCoder2 49,152), write `uint16` `.bin`
   shards (~100M tokens) into `/home/kenpeter/work/data/_shards_code_*`.
4. **Register** the new dirs in `SHARD_DIRS` and tune `curriculum_boost.json` for a
   code-dominant mix (L2 heavy early, L3 mixed in later via curriculum glide).
5. **Curriculum rotation (yes, x-small already rotates tiers):** Yes — x-small's
   curriculum already rotates easy → medium → hard (+ gold) tiers: `_smooth_tier_weights(t)`
   glides easy-heavy at the start to hard-heavy at the end, a cyclic review wave
   periodically boosts easy data, and re-glide happens every 2,000 steps. Gold domains
   (`gold_hard`, `web_gold`, `code_gold`) sit in the hard tier. So the new code tiers
   slot into the existing rotation without a new scheduler.
6. **Smoke-test** the loader before a full run.

### Budget / throughput notes
- Disk: 316 GB free on `/home/kenpeter/work`; 12.5 TB on `/mnt/file_drive`. Write shards
  to `/mnt/file_drive` (or a dedicated `_shards_*` tree there) and point `--data-dir` at
  it; keep the project dir for checkpoints only.
- Tokenization: parallel workers per language family; follow `tokenize_domain_parallel.py`
  pattern (parquet → batches → classify → tokenize → flush).
- Target: reuse the existing 100w-step / 131B-token budget; UltraData alone can supply it
  several times over, so quality filtering matters more than volume.

### Scoring provenance (who scored them)
- Scores are **model-generated by the UltraData-Code-L2-Classifier** (openbmb), not human labels:
  - Embeddings: Qwen3-Embedding-0.6B (1024-dim), per file.
  - Role classifier → `category` (ALGO/WEB/TOOL/DATA/TEST/CONFIG/EXCLUDE) + `cls_confidence`.
  - Relevance model → `algo_rel_score` in [0,1].
  - Quality model → `quality_score` in [0,10].
  - Language-specific calibrated thresholds are applied; the released L2 already only
    contains rows that pass them (observed floors: quality_score ≥ ~3.0 everywhere;
    algo_rel_score ≥ 0.50–0.85 per language, except Python where the floor is 0.0).
- Practical consequence: we do **not** need to reimplement `classify_document`. Filtering can
  be done purely on the shipped columns (`quality_score`, `algo_rel_score`, `category`,
  plus dedup on `uuid`/content hash). Tiering (easy/medium/hard/gold) can be derived from
  score quantiles and `category`.

### Target model architecture — modded-nanogpt record #360 (ANVIL2), scaled to 12 GB

```
                        UltraData code token stream
                                  |
        +-------------------------+-------------------------+
        |                         |                         |
   [token embed]          hashed n-gram table          prefix table
        |                 (bigram + trigram halves;      (prefix-token
        |                  b1=0, sparse row-RMS Adam)     prediction)
        |                         |
     x0 = norm(embed)            |
        |                         |
        v                         |
  +============= 11 transformer blocks =============+
  |  each block:                                     |  injection sites:
  |    x += resid_lambda * Attn(norm(x))             |    x0      -> layers 0,1,2,4,5,7
  |         (attn on layers 0,1,2,3,5,8,10;          |    n-gram  -> layers 0,1,4,5,9
  |          layers 4,6,7,9 = no attention)          |    layer 6 -> gated skip from layer 3
  |         GQA + RoPE + QK-RMSNorm,                 |
  |         mixed head widths (QK 128 wide / 64 narrow, V 64 half),
  |         paired heads, YaRN, sliding-window + partial key offset,
  |         value embeddings, gated XSA, per-head MUDD out-gate
  |    x += post_lambda * ReLU^2(MLP(norm(x)))       |
  |         (all layers except 7; layer 8 runs a parallel MLP)
  |    MUDD gates (pre gate < layer 4, post gate >= layer 4) + residual mix
  +==================================================+
        |
     final norm
        |
     lm_head (fp8) --> logits
        |
     loss: sampled softmax (train)  |  full softmax (val)
     aux : MTP head, prefix-token prediction
```

Decision: the model is **not** the SAN x-small architecture. It is the architecture of
`KellerJordan/modded-nanogpt` record #360 (ANVIL2), used verbatim (every feature below),
with only the *size* knobs reduced so the whole run fits a single 4070 Ti (12 GB, Ada
sm_89). Source of truth: `/home/kenpeter/work/modded-nanogpt` (`track_1_short/`).

**Architecture inventory (all kept):**
- 11 transformer blocks (`NUM_LAYERS = 11`), fixed topology baked into the code:
  `NO_ATTN_LAYERS = (4, 6, 7, 9)`, `NO_MLP_LAYERS = (7,)`, layer 6 adds a gated skip from
  layer 3, `CACHE_LAYERS = (3, 7)`.
- `model_dim = num_heads * head_dim`; `assert hdim == model_dim`.
- Mixed head widths: wide QK `head_dim = 128` on `WIDE_QK_LAYERS = (3, 10)`, narrow QK
  `NARROW_HEAD_DIM = 64` elsewhere in attention; `HALF_V_LAYERS = (1, 8)` halve V/O to 64.
- Paired-head rotary (`PAIRED_HEAD_LAYERS = (0, 2, 5)`), YaRN, partial key offset,
  long/short sliding-window attention in 128-token blocks.
- ReLU² MLP (`relu(x @ W1.T)² @ W2.T`), `MLP_HIDDEN_DIM`, zero-init `c_proj`, uniform
  `c_fc`; one MLP bank of `NUM_MLP_SLOTS = 12` `(c_fc, c_proj)` pairs, plus a parallel MLP
  on layer 8 (slot 11).
- Value embeddings (`VALUE_EMBED_LAYERS = (1, 2, 8, 10)`), value-embed gates, gated XSA
  (`XSA_LAYERS = (1, 3)`), MUDD residual skips and 2-stage MUDD gates (`POST_GATE_LAYER = 4`,
  `MUDD_GROUPS = 12`), x0 / n-gram injection sites.
- Hashed bigram+trigram table (`ngram_table.py`): one `[NGRAM_VOCAB_SIZE, NGRAM_DIM]` table,
  rows `[0, V/2)` = bigram, `[V/2, V)` = trigram, ±1 sign pool, sparse row-RMS Adam
  (β1 = 0), per-cycle row cache/pull.
- Multi-token prediction (`mtp_weights`), prefix-token-prediction auxiliary loss, smear gate,
  per-sublayer `resid_lambdas` / `post_lambdas`.
- Training: ANVIL optimizer (`optim/anvil.py`), sampled softmax for the early-stages training
  loss, FP8 QKV/MLP/lm_head GEMMs, five-stage batch/lr/window schedule.

**Parameter counts (bf16, StarCoder2 vocab 49,152 rounded to 49,152; d = 768):**

| Component | Params | bf16 |
|---|---|---|
| embed | 37.7M | 75 MB |
| lm_head (untied at stage 4) | 37.7M | 75 MB |
| value_embeds (4 planes) | 151.0M | 302 MB |
| mlp_bank (12 slots) | 51.9M | 104 MB |
| qk_bank | 9.4M | 19 MB |
| vo_bank | 1.6M | 3 MB |
| **transformer core** | **~289M** | **~578 MB** |
| ngram table (84.6M × 768, PR #360) | **65.0B** | **130 GB** |

The transformer is small; the **n-gram table is the only part that cannot fit** (it was
sharded 1/8 across 8×H100 = 16 GB/rank). Everything else at d=768 is ~0.6 GB.

**12 GB-fit configuration (primary):** keep every dimension of record #360 exactly
(`d = 768`, `num_heads = 6`, `head_dim = 128`, `NARROW_HEAD_DIM = 64`, `MLP_HIDDEN_DIM = 2816`,
`ngram_dim = 768`, `vocab = 49152`), and shrink only:
- **`NGRAM_VOCAB_SIZE`: 84,602,880 → ~1–2M rows** (1M → 0.77 GB, 2M → 1.54 GB bf16). This is
  the single knob that makes it fit. Keep both channels (bigram/trigram) and the sign trick;
  row-RMS Adam state is only 4 bytes/row, i.e. negligible.
- **Batch / sequence schedule** against the 12 GB activation budget (drop the 131k-token
  batches; use a few thousand tokens/step with grad checkpointing).
- **H100-only systems tricks are replaced, not the model:** the patched FlashAttention-3
  (Hopper-only) → PyTorch `scaled_dot_product_attention`; whole-step CUDA-graph capture →
  per-layer `torch.compile` (or partial graphs); port/keep the FP8 `torch._scaled_mm` paths
  (Ada has FP8 tensor cores) only where the Triton kernels build for sm_89.

**VRAM budget (primary, 2M-row table, grad checkpointing):**

| Item | Approx |
|---|---|
| transformer weights (bf16) | 0.6 GB |
| FP8 weight caches (MLP/lm_head/QKV) | ~0.4 GB |
| ANVIL + Adam states (banks, embed, head, value_embeds) | ~2.3 GB |
| n-gram table (2M × 768 bf16) + sign pool | ~1.6 GB |
| activations / workspace / compile + CUDA context | ~7 GB |
| **total** | **~12 GB** |

**Smaller fallbacks if the primary is too tight:** `d = 512` (`num_heads = 4`,
`MLP_HIDDEN_DIM = 2048`, ~184M core) or `d = 384` (`num_heads = 3`, `MLP = 1408`, ~132M
core). Both keep the 11-layer topology and every feature; only `num_heads`/`model_dim`/
`head_dim`-derived sizes change. Note `head_dim` should stay 128 so the mixed-width
(64,128)/(128,64) structure is preserved.

**Trade-off to accept:** with a 1–2M-row table the hashed n-gram signal is far weaker than
record #360's 84.6M rows, so the large early-convergence boost that table provided will be
much smaller. Everything else (ANVIL, ReLU², MUDD, XSA, mixed-width attention, sampled
softmax, MTP, prefix loss) is retained.

### Plan for the code model (PR #360 architecture, 4070 Ti 12 GB)

Goal: fast convergence to very low loss on code, within 12 GB VRAM.

1. **Tokenizer:** StarCoder2 (49,152 vocab) drop-in; validate roundtrip on all 22 langs.
2. **Data:** convert UltraData L2 (+ L3 for mid-train/SFT) into `uint16` `.bin` shards
   under `/mnt/file_drive/data/_shards_code_*`, tiered easy/medium/hard/gold, reuse the
   existing curriculum rotation. Target the 100w-step / 131B-token budget; L2 alone
   supplies ~400B tokens, L3 ~150B.
3. **Model:** port the record #360 architecture (section above) at the 12 GB-fit
   configuration; shrink only `NGRAM_VOCAB_SIZE` and the batch/sequence schedule.
4. **Optimizer:** ANVIL (as in #360); keep grad checkpointing + per-layer compile for 12 GB.
5. **Training stages:** (a) CPT on L2-heavy mix, (b) mid-train on 1:1 L2+L3 mix,
   (c) SFT on L3 (task→solution, completion-only loss). Eval with code-domain perplexity.
6. **Throughput:** SDPA instead of FA3, bf16, FP8 `_scaled_mm` where it builds on sm_89,
   avoid whole-step CUDA graphs on 12 GB.

### Open items
- Validate the chosen tokenizer roundtrips on all 22 languages before committing shards.
- Decide whether to keep the 16-domain curriculum or collapse to a code-only tier set.
- Optionally add `category`-aware mixing (e.g. boost ALGO, down-weight TOOL/CONFIG).
