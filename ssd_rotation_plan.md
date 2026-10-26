# SSD Rotation Plan for UltraData Code Training

Goal: train `san_model` (via `train_san.py`) on UltraData L2 code shards
at max throughput by keeping the active training shards on the NVMe SSD
(`/`, 291 GB free) instead of the HDD (`/mnt/file_drive`, 14.6 TB).

## Constraints
- SSD free: ~291 GB
- Tier sizes (HDD): code_easy ~150 GB, code_medium ~114 GB, code_hard ~135 GB
- Total: ~399 GB (does not fit on SSD all at once)
- GPU: 4070 Ti 12 GB, currently ~13k tok/s with MTP off, B=3

## Strategy: two-tier stages
Keep two tiers on SSD at a time (~249-264 GB), train, then swap one tier.
This preserves a two-tier curriculum mix while fitting in SSD.

### Stage 1 (easy + medium)
- SSD dirs: `_shards_code_easy`, `_shards_code_medium`  (~264 GB)
- Train with SHARD_DIRS = {code_easy, code_medium}
- EASY_SPLIT/MED_SPLIT/HARD_SPLIT adjusted: easy 0.5, medium 0.5, hard 0
- Run for e.g. 200k steps or until loss plateau

### Stage 2 (medium + hard)
- Delete easy from SSD; copy hard to SSD  (~249 GB)
- SHARD_DIRS = {code_medium, code_hard}
- medium 0.5, hard 0.5, easy 0

### Stage 3 (easy + hard)  (optional full coverage)
- Delete medium; copy easy back
- SHARD_DIRS = {code_easy, code_hard}

## Per-stage commands (example Stage 1)
```bash
# 1. stop training
pkill -f train_san.py

# 2. prepare SSD dirs
mkdir -p /home/kenpeter/work/data_ssd
rm -rf /home/kenpeter/work/data_ssd/_shards_code_easy /home/kenpeter/work/data_ssd/_shards_code_medium
mkdir -p /home/kenpeter/work/data_ssd/_shards_code_easy /home/kenpeter/work/data_ssd/_shards_code_medium

# 3. copy (rsync, HDD -> SSD)
rsync -a --progress /home/kenpeter/work/data/_shards_code_easy/ /home/kenpeter/work/data_ssd/_shards_code_easy/
rsync -a --progress /home/kenpeter/work/data/_shards_code_medium/ /home/kenpeter/work/data_ssd/_shards_code_medium/

# 4. point train_san.py at SSD (edit SHARD_DIRS) then train
#    (use --mtp-weight 0 for max throughput)
cd /home/kenpeter/work/coder
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES=0 /home/kenpeter/.hermes/hermes-agent.bak/venv/bin/python -u train_san.py \
  --curriculum --steps 200000 --batch-size 3 --grad-accum 8 --mtp-weight 0 \
  --checkpoint-dir /home/kenpeter/work/coder/checkpoints/san
```

## Swap stage (example Stage1 -> Stage2)
```bash
pkill -f train_san.py
rm -rf /home/kenpeter/work/data_ssd/_shards_code_easy
mkdir -p /home/kenpeter/work/data_ssd/_shards_code_hard
rsync -a --progress /home/kenpeter/work/data/_shards_code_hard/ /home/kenpeter/work/data_ssd/_shards_code_hard/
# edit SHARD_DIRS/splits, then relaunch with --resume checkpoints/san/san_latest.pt
```

## Notes
- Resume with `--resume checkpoints/san/san_latest.pt` for seamless continuation.
- SSD free space must be checked before each copy (`df -h /home/kenpeter/work`).
- HDD utilization is only ~23% today, so SSD copy gives modest gain; do it only if
  I/O becomes the bottleneck (watch `iostat` for sda %util near 100).
- Clean SSD staging dirs after all stages to free space.
