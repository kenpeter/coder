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
- `tokenize_ultradata.py` — **the data pipeline**: UltraData L2/L3 parquet → StarCoder2
  `uint16` `.bin` shards (tiered easy/medium/hard/gold), EOS-separated, streaming.
- `run_tokenize_all.sh` — launcher for the full conversion.
- `verify_shards.py` — check shard dtype/ids/EOF boundaries and decode a sample.

## Data pipeline (ready to run)

Converts `/mnt/file_drive/data/ultradata` (1129 parquet files, 1.2 TB) into `uint16`
token shards consumable by `train_san.py` unchanged.

```bash
# smoke test (one parquet file per group) — done, verified
python tokenize_ultradata.py --smoke --langs py --levels L2 --out-root ./_smoke

# full build → /mnt/file_drive/data/_shards_code_{easy,medium,hard,gold}
WORKERS=11 ./run_tokenize_all.sh            # L2 + L3, all languages
./run_tokenize_all.sh L2                    # L2 only (CPT)
./run_tokenize_all.sh L3                    # L3 only (gold / mid-train)
```

Runtime deps: use `/home/kenpeter/work/fly_brain/.venv/bin/python` (has `pyarrow`,
`transformers`, `tokenizers`; installed with `uv pip install`).

**Format contract:** `uint16` little-endian, continuous token stream, EOS (`<|endoftext|>`,
id `0`) between documents, `100_000_000` tokens/shard (matches `abstraction.md`; override with
`--tokens-per-shard`). Tiers: L2 → `code_easy/medium/hard` by per-language `quality_score`
quantiles (q33/q66); L3 → `code_gold` with `task`/`analysis`/`solution`/`test` serialized.
Filters: `quality_score >= 3.0`, per-language `algo_rel_score` floor (Python = 0.0), drop
`EXCLUDE` category, dedup on `uuid`.

**Wiring:** shards live on `/mnt/file_drive/data/_shards_code_*` (12 TB free) and are
symlinked into `/home/kenpeter/work/data/_shards_code_*`, so `train_san.py`'s `SHARD_DIRS`
work with no edit.

**Smoke result (validated):** L2 `py` part-00001 → 612,143 docs / 734M tokens / 2.59 GB
text (3.53 tokens/byte, ~3.2M tok/s). L3 `py` part-00001 → 141,159 docs / 237M tokens
(3.26 tokens/byte). All ids `< 49152`, dtype `uint16`, EOS boundaries present, decode
round-trips to real Python.

## Summary of the plan
- **Tokenizer:** `bigcode/starcoder2-7b` (vocab 49,152) — not SmolLM2.
- **Data:** UltraData L2 (raw algorithmic code) → continued pre-training; L3
  (task/analysis/solution/test) → mid-training / SFT.
- **Model:** modded-nanogpt record #360 (ANVIL2) architecture used verbatim
  (11 layers, 6 heads, head_dim 128, model_dim 768, ReLU² MLP, MUDD skips, XSA,
  mixed-width attention, value embeddings, hashed bigram/trigram table, ANVIL optimizer,
  sampled-softmax loss), with only the n-gram table (~84.6M → 1–2M rows) and the
  batch/sequence schedule shrunk to fit 12 GB.
