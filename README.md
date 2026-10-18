# coder

Planning artifacts for training a code model on the **UltraData** corpus
(`/mnt/file_drive/data/ultradata`), targeting a single 12 GB GPU (RTX 4070 Ti, Ada sm_89).

## Contents
- `abstraction.md` — the plan: data pipeline (tokenizer choice, L2 vs L3, scoring
  provenance, shard conversion) and the target model architecture
  (modded-nanogpt record #360 / ANVIL2, scaled to fit 12 GB).
- `ultradata_stats.py` — summarise the UltraData parquet corpus.
- `inspect_parquet.py` — inspect L2 parquet schema/columns.
- `inspect_l3.py`, `inspect_l3b.py` — inspect L3 parquet schema/columns.

## Summary of the plan
- **Tokenizer:** `bigcode/starcoder2-7b` (vocab 49,152) — not SmolLM2.
- **Data:** UltraData L2 (raw algorithmic code) → continued pre-training; L3
  (task/analysis/solution/test) → mid-training / SFT.
- **Model:** modded-nanogpt record #360 (ANVIL2) architecture used verbatim
  (11 layers, 6 heads, head_dim 128, model_dim 768, ReLU² MLP, MUDD skips, XSA,
  mixed-width attention, value embeddings, hashed bigram/trigram table, ANVIL optimizer,
  sampled-softmax loss), with only the n-gram table (~84.6M → 1–2M rows) and the
  batch/sequence schedule shrunk to fit 12 GB.
