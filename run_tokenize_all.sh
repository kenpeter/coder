#!/usr/bin/env bash
# Full UltraData-Code -> StarCoder2 uint16 .bin shard build.
#
# Writes to /mnt/file_drive/data/_shards_code_{easy,medium,hard,gold}
# (12.5 TB free there). Runtime is dominated by tokenization; ~22 groups run
# in parallel (11 languages x 2 levels).
#
# Usage:
#   ./run_tokenize_all.sh              # L2 + L3, all languages, 11 workers
#   ./run_tokenize_all.sh L2           # L2 only (CPT shards)
#   ./run_tokenize_all.sh L3           # L3 only (gold / mid-train)
#   WORKERS=8 ./run_tokenize_all.sh
set -euo pipefail

cd "$(dirname "$0")"

PY=/home/kenpeter/work/fly_brain/.venv/bin/python
LEVELS="${1:-L2,L3}"
LANGS="${LANGS:-all}"
WORKERS="${WORKERS:-11}"
OUT_ROOT="${OUT_ROOT:-/mnt/file_drive/data}"

echo "== UltraData tokenization =="
echo "  levels   : $LEVELS"
echo "  langs    : $LANGS"
echo "  workers  : $WORKERS"
echo "  out-root : $OUT_ROOT"
echo

exec "$PY" tokenize_ultradata.py \
  --levels "$LEVELS" \
  --langs "$LANGS" \
  --workers "$WORKERS" \
  --out-root "$OUT_ROOT" \
  "$@"
