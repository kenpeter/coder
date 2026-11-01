#!/usr/bin/env bash
# Resume the SAN "glide" (curriculum) training on the restored hard shards.
# Runs from ~/work/coder; env = user-site torch (CUDA) installed earlier.
set -euo pipefail
cd /home/kenpeter/work/coder

export PYTHONHASHSEED=0
export CUDA_VISIBLE_DEVICES=0

LOG=/home/kenpeter/work/coder/train_san.log
echo "=== training launch $(date) ===" | tee -a "$LOG"

# Resume from the real checkpoint (NOT the x-small default dir).
# --curriculum enables the DoReMi re-glide "glide" pipeline.
# Data reads the LOCAL _shards_final copy (easy/medium/hard/gold).
# 12L + B=8 = x-small's ~31k tok/s config. The checkpoint is 12 layers; train_san
# now auto-matches num_layers from the checkpoint, and --batch-size 8 gives the
# same effective batch (8*8=64) x-small uses for full GPU utilization.
nohup python3 -u train_san.py \
  --resume /home/kenpeter/work/coder/checkpoints/san/san_latest.pt \
  --checkpoint-dir /home/kenpeter/work/coder/checkpoints/san \
  --curriculum \
  --data-dir /home/kenpeter/work/data/_shards_final \
  --num-layers 12 \
  --batch-size 8 \
  --log-every 10 \
  >> "$LOG" 2>&1 &

echo "training pid $!" | tee -a "$LOG"
