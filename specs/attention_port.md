# Spec — Attention for the 12 GB build (decisions)

## Decision log
- **Sliding-window causal attention: KEEP / USE.** Confirmed by user ("we use sliding window
  attention in our coder"). x-small does *not* have it today (it runs full causal SDPA), so this
  is an addition, not a preservation.
- **Kernel on Ada (RTX 4070 Ti, sm_89, 12 GB):** do **not** use the patched FlashAttention-3
  (Hopper-only). Primary path = **`torch.nn.attention.flex_attention`** (block mask =
  `causal AND (t - s < window)`; supports per-layer windows and `score_mod`). Fallback =
  **`torch.nn.functional.scaled_dot_product_attention`** with an explicit additive/bool mask
  (fine at seq ≤ 2048–3072).
- **Selections still open:** (1) bolt the window + optional YaRN onto SAN's existing GQA
  attention, or (2) port #360's full attention (MHA 6/6, mixed QK 64/128, XSA, value-embeds,
  paired heads, partial key offset). Recommendation: (1) first, (2) after the data pipeline is
  proven.

## #360 attention feature set (target)
causal; GQA or MHA; QK-RMSNorm; RoPE + YaRN; long/short per-layer sliding window (blocks of 128),
warm-up `(1,3)→(3,7)→(5,11)→(6,13)`, long window on layers `(3,10)`; partial key offset;
mixed head widths (QK 64 narrow / 128 wide, V/O 64 on `HALF_V_LAYERS=(1,8)`); value-embedding
planes with per-head gate; gated learnable XSA on `(1,3)`; sparse attention gate + per-head
output gates on `(3,10)`; MUDD value skips.

## x-small attention today (`san_model.py` → `MultiHeadAttention`)
GQA (`num_heads=8`, `num_kv_heads=4`), `head_dim = attn_dim/num_heads = 48`, per-head
`ZCRMSNorm` on q/k, plain `RoPE` (`theta=100000`, no YaRN), `sigmoid(gate_proj(x))` output gate,
`out_proj`, **full causal `F.scaled_dot_product_attention(..., is_causal=True)`** (no window).
`make_causal_packing_mask(..., window=)` exists but is unused in training.
