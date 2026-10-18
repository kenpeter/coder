#!/usr/bin/env python3
"""
tokenize_ultradata.py
=====================
Convert UltraData-Code (L2 / L3) parquet into StarCoder2 `uint16` `.bin` shards
consumable by `train_san.py` (BinShardDataset / StratifiedShardDataset).

Tokenizer : bigcode/starcoder2-7b  (vocab 49152, EOS id 0)
Format    : uint16 little-endian, continuous token stream, EOS between documents
Tiers/L2  : code_easy / code_medium / code_hard  (per-language score quantiles)
Tier/L3   : code_gold  (task/analysis/solution/test serialized as an exercise)

Output dirs (under --out-root):
    _shards_code_easy/    train-<lang>-<NNNNN>.bin
    _shards_code_medium/  train-<lang>-<NNNNN>.bin
    _shards_code_hard/    train-<lang>-<NNNNN>.bin
    _shards_code_gold/    codegold-<lang>-<NNNNN>.bin

Examples
--------
# Smoke test: tokenize one parquet file end to end
python tokenize_ultradata.py --smoke --langs py --levels L2 \
    --out-root /home/kenpeter/work/coder/_smoke

# Full run (22 = 11 langs x 2 levels groups, parallel workers)
python tokenize_ultradata.py --levels L2,L3 --langs all \
    --out-root /mnt/file_drive/data --workers 11
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# ---------------------------------------------------------------- constants
TOKENIZER_NAME = "bigcode/starcoder2-7b"
VOCAB_SIZE = 49152
EOS_ID = 0

DEFAULT_TOKENS_PER_SHARD = 100_000_000          # abstraction.md: ~100M tokens/shard
DEFAULT_QUALITY_FLOOR = 3.0                     # observed UltraData floor
DEFAULT_ALGO_FLOOR = 0.5                        # per-language in [0.5, 0.85]
PY_ALGO_FLOOR = 0.0                             # Python ships with a 0.0 floor
DEFAULT_DROP_CATEGORIES = ("EXCLUDE",)

INPUT_ROOT = Path("/mnt/file_drive/data/ultradata")

FILE_RE = re.compile(
    r"^UltraData-Code-(?P<level>L[0-9]+)-(?P<lang>[A-Za-z0-9]+)"
    r"-part-(?P<part>[0-9]+)-of-(?P<total>[0-9]+)\.parquet$"
)

L2_COLUMNS = ["uuid", "content", "category", "algo_rel_score", "quality_score"]
L3_COLUMNS = [
    "uuid", "content", "content_format", "raw_content",
    "task", "analysis", "solution", "test",
    "full_content", "full_content_format",
]

# L2 -> CPT tiers; L3 -> gold / mid-train tier
TIER_PREFIX = {"easy": "train", "medium": "train", "hard": "train", "gold": "codegold"}


# ---------------------------------------------------------------- helpers
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def algo_floor_for(lang: str, overrides: dict) -> float:
    if lang in overrides:
        return float(overrides[lang])
    if lang in ("py", "python"):
        return PY_ALGO_FLOOR
    return DEFAULT_ALGO_FLOOR


def discover(input_root: Path) -> dict:
    """Return {(level, lang): [sorted parquet paths]}."""
    groups: dict[tuple[str, str], list[Path]] = {}
    for p in sorted(input_root.glob("*.parquet")):
        m = FILE_RE.match(p.name)
        if not m:
            continue
        key = (m.group("level"), m.group("lang"))
        groups.setdefault(key, []).append(p)
    for k in groups:
        groups[k].sort(key=lambda x: int(FILE_RE.match(x.name).group("part")))
    return groups


def serialize_l3(row: dict) -> str:
    """Serialize an L3 exercise row into a consistent instruction-style document."""
    parts = []
    for key in ("task", "analysis", "solution", "test"):
        v = row.get(key)
        if v:
            parts.append(f"### {key}\n{v}")
    if not parts:
        for key in ("full_content", "content", "raw_content"):
            v = row.get(key)
            if v:
                parts.append(str(v))
                break
    return "\n\n".join(parts)


# ---------------------------------------------------------------- thresholds
def compute_thresholds(files, lang, quality_floor, sample_cap=500_000):
    """Per-language q33/q66 of quality_score from a bounded sample (single column read)."""
    qs = []
    for fp in files:
        try:
            for batch in pq.ParquetFile(fp).iter_batches(batch_size=8192,
                                                         columns=["quality_score"]):
                col = batch.column("quality_score").to_pylist()
                qs.extend(v for v in col if v is not None)
                if len(qs) >= sample_cap:
                    break
        except Exception:
            continue
        if len(qs) >= sample_cap:
            break
    if len(qs) < 100:
        return 0.0, 0.0  # degenerate -> everything lands in "hard"
    arr = np.asarray(qs, dtype=np.float64)
    return float(np.quantile(arr, 0.33)), float(np.quantile(arr, 0.66))


def tier_for(level: str, quality, q33: float, q66: float) -> str:
    if level != "L2":
        return "gold"
    if quality is None:
        return "medium"
    if quality < q33:
        return "easy"
    if quality < q66:
        return "medium"
    return "hard"


# ---------------------------------------------------------------- shard writer
class ShardWriter:
    """Accumulates uint16 arrays and flushes fixed-size shards.

    Buffers numpy arrays (2 bytes/token) rather than Python ints, so a
    100M-token shard costs ~200 MB, not multiple GB.
    """

    def __init__(self, out_dir: Path, prefix: str, lang: str, tokens_per_shard: int):
        self.out_dir = out_dir
        self.prefix = prefix
        self.lang = lang
        self.tokens_per_shard = tokens_per_shard
        self.parts: list[np.ndarray] = []
        self.count = 0
        self.shard_idx = 0
        self.total_tokens = 0
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def add(self, ids) -> None:
        arr = np.asarray(ids, dtype=np.uint16)
        if arr.size == 0:
            return
        self.parts.append(arr)
        self.count += int(arr.size)
        while self.count >= self.tokens_per_shard:
            self._flush(self.tokens_per_shard)

    def _take(self, n: int) -> np.ndarray:
        out = []
        need = n
        while need > 0:
            p = self.parts[0]
            if p.size <= need:
                out.append(p)
                need -= int(p.size)
                self.parts.pop(0)
            else:
                out.append(p[:need])
                self.parts[0] = p[need:]
                need = 0
        arr = out[0] if len(out) == 1 else np.concatenate(out)
        self.count -= int(arr.size)
        return arr

    def _flush(self, n: int) -> None:
        arr = self._take(n)
        if arr.size and int(arr.max()) >= VOCAB_SIZE:
            raise ValueError(f"token id >= {VOCAB_SIZE} (max {int(arr.max())})")
        path = self.out_dir / f"{self.prefix}-{self.lang}-{self.shard_idx:05d}.bin"
        arr.tofile(path)
        self.total_tokens += int(arr.size)
        self.shard_idx += 1

    def close(self) -> None:
        if self.count:
            self._flush(self.count)


# ---------------------------------------------------------------- worker
def process_group(args_tuple):
    (level, lang, files, out_root, tokens_per_shard, quality_floor,
     algo_floor, drop_categories, dedup_cap, tokenizer_name, limit_files,
     do_smoke, max_docs) = args_tuple

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_name, clean_up_tokenization_spaces=False)
    if len(tok) != VOCAB_SIZE:
        log(f"WARN vocab {len(tok)} != {VOCAB_SIZE} for {tokenizer_name}")
    eos = tok.eos_token_id if tok.eos_token_id is not None else EOS_ID

    files = files[:limit_files] if limit_files else files

    # thresholds only matter for L2 tiering
    if level == "L2":
        q33, q66 = compute_thresholds(files, lang, quality_floor)
    else:
        q33 = q66 = 0.0

    columns = L2_COLUMNS if level == "L2" else L3_COLUMNS

    writers: dict[str, ShardWriter] = {}
    seen: set = set()
    dedup_disabled = False
    stats = {"level": level, "lang": lang, "files": len(files),
             "docs_in": 0, "docs_kept": 0, "tokens": 0, "bytes": 0,
             "q33": q33, "q66": q66, "dedup_disabled": False, "shards": {}}
    t0 = time.time()
    sample_text = None
    done = False

    for fp in files:
        if done:
            break
        pf = pq.ParquetFile(fp)
        try:
            it = pf.iter_batches(batch_size=2048, columns=columns)
        except Exception as e:
            log(f"SKIP {fp.name}: {e}")
            continue
        for batch in it:
            if done:
                break
            d = batch.to_pydict()
            n = len(d.get("uuid", d.get("content", [])))
            contents, metas = [], []
            for i in range(n):
                stats["docs_in"] += 1
                uuid = d["uuid"][i] if "uuid" in d else f"{fp.name}:{i}"
                # dedup (best-effort, per worker): key by a cheap hash to bound memory
                if not dedup_disabled:
                    key = hash(uuid)
                    if key in seen:
                        continue
                    if len(seen) < dedup_cap:
                        seen.add(key)
                    else:
                        dedup_disabled = True
                # score filters (L2)
                if level == "L2":
                    q = d["quality_score"][i]
                    a = d["algo_rel_score"][i]
                    cat = d.get("category", [None] * n)[i]
                    if q is None or q < quality_floor:
                        continue
                    if a is None or a < algo_floor:
                        continue
                    if cat in drop_categories:
                        continue
                    tier = tier_for(level, q, q33, q66)
                    text = d["content"][i]
                else:
                    tier = "gold"
                    row = {k: d[k][i] for k in columns}
                    text = serialize_l3(row)
                if not text:
                    continue
                contents.append(text)
                metas.append(tier)

            if not contents:
                continue
            try:
                enc = tok(contents, add_special_tokens=False)["input_ids"]
            except Exception:
                enc = [tok.encode(c, add_special_tokens=False) for c in contents]

            for text, tier, ids in zip(contents, metas, enc):
                stats["docs_kept"] += 1
                stats["bytes"] += len(text.encode("utf-8", "ignore"))
                if sample_text is None:
                    sample_text = (tier, text[:200], ids[:16])
                w = writers.get(tier)
                if w is None:
                    w = writers[tier] = ShardWriter(
                        out_root / f"_shards_code_{tier}" if not do_smoke
                        else out_root / "_smoke" / f"_shards_code_{tier}",
                        TIER_PREFIX[tier], lang, tokens_per_shard)
                w.add(ids)
                w.add([eos])
            if max_docs and stats["docs_kept"] >= max_docs:
                done = True
                break

    for tier, w in writers.items():
        w.close()
        stats["shards"][tier] = {"shard_idx": w.shard_idx, "tokens": w.total_tokens}
        stats["tokens"] += w.total_tokens

    stats["dedup_disabled"] = dedup_disabled
    stats["seconds"] = round(time.time() - t0, 2)
    stats["sample"] = sample_text
    stats["sample_decoded"] = (
        tok.decode(sample_text[2]) if sample_text else None)
    return stats


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    ap.add_argument("--out-root", type=Path, default=Path("/mnt/file_drive/data"))
    ap.add_argument("--levels", default="L2,L3", help="comma list, e.g. L2,L3")
    ap.add_argument("--langs", default="all", help="comma list or 'all'")
    ap.add_argument("--workers", type=int, default=0, help="0 = min(groups, cpu)")
    ap.add_argument("--tokenizer", default=TOKENIZER_NAME)
    ap.add_argument("--tokens-per-shard", type=int, default=DEFAULT_TOKENS_PER_SHARD)
    ap.add_argument("--quality-floor", type=float, default=DEFAULT_QUALITY_FLOOR)
    ap.add_argument("--algo-floor", type=float, default=DEFAULT_ALGO_FLOOR)
    ap.add_argument("--algo-floor-json", default=None,
                    help="path to JSON {lang: floor} overrides")
    ap.add_argument("--drop-categories", default=",".join(DEFAULT_DROP_CATEGORIES))
    ap.add_argument("--dedup-cap", type=int, default=5_000_000,
                    help="max uuid hashes remembered per worker before dedup is "
                         "disabled (best-effort, bounded memory)")
    ap.add_argument("--limit-files", type=int, default=0, help="debug: cap files/group")
    ap.add_argument("--max-docs", type=int, default=0,
                    help="debug: cap documents kept per group (0 = unlimited)")
    ap.add_argument("--smoke", action="store_true",
                    help="process exactly ONE parquet file per selected group")
    args = ap.parse_args()

    overrides = {}
    if args.algo_floor_json:
        overrides = json.loads(Path(args.algo_floor_json).read_text())

    groups = discover(args.input_root)
    if not groups:
        log(f"ERROR: no parquet matching pattern under {args.input_root}")
        sys.exit(1)

    levels = {s.strip() for s in args.levels.split(",") if s.strip()}
    langs = None if args.langs.strip() == "all" else {s.strip() for s in args.langs.split(",") if s.strip()}

    selected = []
    for (level, lang), files in sorted(groups.items()):
        if levels and level not in levels:
            continue
        if langs and lang not in langs:
            continue
        selected.append((level, lang, files))

    if not selected:
        log("ERROR: no groups selected")
        sys.exit(1)

    log(f"selected {len(selected)} group(s): "
        + ", ".join(f"{lvl}/{lg}({len(f)})" for lvl, lg, f in selected))

    limit = 1 if args.smoke else args.limit_files
    drop_cats = tuple(s.strip() for s in args.drop_categories.split(",") if s.strip())

    tasks = []
    for level, lang, files in selected:
        tasks.append((
            level, lang, files, args.out_root, args.tokens_per_shard,
            args.quality_floor, algo_floor_for(lang, overrides), drop_cats,
            args.dedup_cap, args.tokenizer, limit, args.smoke, args.max_docs,
        ))

    workers = args.workers or min(len(tasks), 8)
    results = []
    if args.smoke or workers == 1:
        for t in tasks:
            results.append(process_group(t))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(process_group, t): t[:2] for t in tasks}
            for fut in as_completed(futs):
                lvl, lg = futs[fut]
                try:
                    r = fut.result()
                except Exception as e:
                    log(f"FAILED {lvl}/{lg}: {e}")
                    r = {"level": lvl, "lang": lg, "error": str(e)}
                results.append(r)

    results.sort(key=lambda r: (r.get("level", ""), r.get("lang", "")))

    total_tokens = sum(r.get("tokens", 0) for r in results)
    total_docs = sum(r.get("docs_kept", 0) for r in results)
    total_bytes = sum(r.get("bytes", 0) for r in results)
    total_sec = sum(r.get("seconds", 0) for r in results)

    log("=" * 68)
    for r in results:
        if "error" in r:
            log(f"  {r['level']}/{r['lang']}: ERROR {r['error']}")
            continue
        log(f"  {r['level']}/{r['lang']}: kept {r['docs_kept']:>9,}/{r['docs_in']:<10,} "
            f"tokens {r['tokens']:>12,}  shards {r['shards']}  "
            f"q33/q66={r['q33']:.1f}/{r['q66']:.1f}  {r['seconds']}s")
        if r.get("sample_decoded"):
            log(f"      sample[{r['sample'][0]}]: {r['sample_decoded']!r}")
    log("=" * 68)
    tpb = (total_bytes / total_tokens) if total_tokens else 0.0
    log(f"TOTAL kept docs {total_docs:,}  tokens {total_tokens:,}  "
        f"bytes {total_bytes:,}  tokens/byte {tpb:.4f}")
    log(f"TOTAL wall (sum of workers) {total_sec:.1f}s; "
        f"throughput ~{total_tokens / total_sec if total_sec else 0:.0f} tok/s/worker")

    # metadata
    meta = {
        "tokenizer": args.tokenizer, "vocab_size": VOCAB_SIZE, "eos_id": EOS_ID,
        "tokens_per_shard": args.tokens_per_shard,
        "quality_floor": args.quality_floor, "algo_floor_default": args.algo_floor,
        "drop_categories": list(drop_cats), "smoke": args.smoke,
        "groups": results,
    }
    meta_dir = (args.out_root / "_smoke") if args.smoke else args.out_root
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "_shards_code_meta.json").write_text(json.dumps(meta, indent=2))
    log(f"wrote {meta_dir / '_shards_code_meta.json'}")


if __name__ == "__main__":
    main()
