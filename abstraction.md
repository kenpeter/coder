# SAN x-small — Abstraction

> Abstraction of the project and the agent session that produced this release.

## What SAN is
SAN (Simple Attention Network) is a custom code language model — a PyTorch port of modded-nanogpt record #360 (ANVIL2) scaled down to x-small for single-GPU RTX 4070 Ti 12 GB. Trained locally on UltraData-Code `uint16` shards.

Components:
- **ZCRMSNorm** — zero-centered RMSNorm on q/k and pre-MLP
- **ReLU² MLP** — `relu(x @ W1.T)² @ W2.T` MLP (no dense FFN weights beyond c_fc/c_proj)
- **MultiHeadAttention** — GQA + q/k RMSNorm + YaRN RoPE + learned per-head gate, mixed head widths (QK 128 wide / 64 narrow, V 64 half-V layers), sliding window + document boundaries
- **Engram** — hashed bigram+trigram n-gram KV memory table (NGRAM_VOCAB_SIZE rows)
- **MTP** — multi-token-prediction head on lm_head output
- **Smear gate** — sigmoid gate mixing previous-token embeddings into current
- **Tied embeddings** — embed.weight tied to lm_head.weight
- **MUDD** — per-block residual lambda and post-lambda scalars

## Trained configuration (this release)
| field | value |
|---|---|
| vocab_size | 49152 (StarCoder2) |
| d_model | 512 (x-small) |
| num_heads | 4 |
| head_dim | 128 |
| num_layers | 6 |
| ngram_vocab | 256,000 (bigram+trigram halves) |
| ngram_dim | 512 |
| max_seq_len | 8192 |
| dtype | bfloat16 |
| no_attn_layers | (1, 3) |
| wide_qk_layers | (0, 5) |
| half_v_layers | (1, 4) |
| long_window_layers | (0, 5) |
| value_embed_layers | (0, 1, 4, 5) |
| batch_tokens | 8192 |
| grad_accum | 4 (effective 32,768 tokens/step) |

Training: target **1,000,000 steps**, GPU power cap **150 W** (max token/s on RTX 4070 Ti 12 GB).

## Session log (this agent run)
1. **x-small + Muon + torch.compile** — `coder_final.py` launched at step 0, fresh start (old `trainer_state.pt` was AdamW-incompatible with Muon, cleared). ~60k tok/s at step 200, VRAM 6.1 GB (~50%), GPU 100%, power ~145 W.
2. **Architecture changes**: x-small (D=512, 6 layers) from D=768, 11 layers; Muon optimizer with Newton-Schulz orthogonalization on square 2D params; AdamW for embeddings/scalars/gates; torch.compile `mode="default"` on all layers (blocks, attention, MLP, embed, lm_head).
3. **N-gram table shrunk** 2M → 256K rows, freeing ~2.6 GB VRAM for bigger batch (8192 tokens/step).
4. **torch.compile** works with mode="default" (no CUDAGraphs, which conflicted with grad accumulation).
5. **Flash Attention 2 blocked** — host CUDA 12.4 vs torch cu130 mismatch, can't build.

## Key operating insights
- **~60k tok/s** achieved with x-small + Muon + compile — up from ~26k tok/s on the 286M model with AdamW + no compile.
- **VRAM headroom** at ~50% — batch could go higher but 8192 × 4 is the throughput sweet spot (larger batches hit quadratic attention cost).
- **Power cap resets to ~285 W on reboot** — reapply `sudo nvidia-smi -pl 150` after restart.
- **Resume is the normal path** — checkpoints every 30 min to `san_latest.pt`, resume from `trainer_state/trainer_state.pt`.
- **Checkpoint save requires SIGTERM** (saves at next step boundary then exits) — no save-and-keep-running signal.

## Checkpointing — save every 30 minutes, overwrite old
| Setting | Value |
|---|---|
| Save interval | Every 30 minutes (wall-clock) |
| Checkpoint file | `san_latest.pt` (overwrites previous) |
| Trainer state | `trainer_state/trainer_state.pt` (saved every 60s for resume) |
| Resume from | `trainer_state.pt` — restores optimizer, RNG, step count |
| Fresh start | No checkpoint → starts from initialization |

## Compilation & throughput tuning (max tokens/sec on RTX 4070 Ti)
| Optimization | Setting | Why |
|---|---|-----|
| Attention | `F.scaled_dot_product_attention` (SDPA) | Works on Ada sm_89; FlashAttention-3 needs Hopper |
| Precision | bf16 weights + activations | Native on Ada |
| Compile | Per-layer `torch.compile(mode="default")` | No CUDAGraphs (conflicts with grad accumulation) |
| Grad accumulation | 4 steps | Effective batch 32,768 tokens/step |
| N-gram table | 256K rows | Frees ~2.6 GB VRAM for batch headroom |

**Measured:** ~60k tok/s at 8192 tokens/step × 4 grad accum, 100% GPU util, ~145 W, ~6.1 GB VRAM (50% of 12 GB).

**Dead-ends:**
- `torch.compile(mode="reduce-overhead")` — conflicts with grad accumulation + CUDAGraphs
- FP8 `_scaled_mm` — no autograd backward on torch 2.14.1+cu130
- Flash Attention 2 — CUDA 12.4 host vs cu130 torch mismatch, can't build
- Flash Attention 3 — Hopper-only, 4070 Ti is Ada

## Optimizer — Muon + AdamW
| Param type | Optimizer | LR | Notes |
|---|---|---|---|
| 2D weight matrices (square) | Muon (Newton-Schulz) | 0.02 | Orthogonalization via 5 NS steps |
| Embeddings, lm_head, scalars, gates, n-gram | AdamW | 1e-3 | Standard betas (0.9, 0.95) |
| LR schedule | Cosine warmup | 100-step linear warmup, cosine decay to 0.1x |

## Data Pipeline — code training on UltraData
- **Shards:** UltraData L2 shards (easy/medium/hard) at `/mnt/file_drive/data/_shards_code_*` (~499.6B tokens, 5,015 shards, raw uint16, vocab 49152, no BOS, no header)
- **L2 = CPT pre-training**, L3 (gold) = SFT/mid-train (task→solution, completion-only loss on solution portion)
- **Rotation:** Stage 1 CPT on L2; Stage 2 fixed 1:1 L2+L3 mix; Stage 3 L3 gold only
- **Data loader:** Simple sequential chunking (no BOS markers in shards), restarts at document boundary for varlen packing

## SSD Rotation Plan (UltraData L2 code)
SSD free ~291 GB; tiers ~150 GB (easy) + 114 GB (medium) + 135 GB (hard) = 399 GB,
so not all fit at once. Train in two-tier stages on SSD:
- Stage 1: easy + medium (~264 GB) on SSD
- Stage 2: medium + hard (~249 GB) on SSD
- Stage 3: easy + hard (~285 GB) on SSD
Per stage: stop training, rsync tiers HDD→SSD, repoint SHARD_DIRS, train with
--resume san_latest.pt, then delete swapped tier and copy next.

