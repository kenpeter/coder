#!/bin/bash
# Check if the SAN training process is running
# Runs every 30 minutes via cron
# Also sends Telegram notification about training status

TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Check if training process is running
if pgrep -f "python3.*train_san.py" > /dev/null 2>&1; then
    STATUS="RUNNING"
    # Optionally update PID file
    pgrep -f "train_san.py" > /home/kenpeter/work/coder/training.pid 2>/dev/null
else
    STATUS="STOPPED"
fi

# Log status
echo "[$TIMESTAMP] Coder training status: $STATUS" >> /home/kenpeter/work/coder/training.log

# Log current model loss value
CKPT_LOSS=$(/home/kenpeter/.hermes/tools/python-3.14.9+202****1021-linux-x64/bin/python3 -c "
import torch
st = torch.load('/home/kenpeter/work/coder/checkpoints/san/san_latest.pt', map_location='cpu', weights_only=False)
print(st.get('loss', 'N/A'))
" 2>/dev/null)

if [ -n "$CKPT_LOSS" ] && [ "$CKPT_LOSS" != "N/A" ]; then
    echo "[$TIMESTAMP] Model loss: $CKPT_LOSS" >> /home/kenpeter/work/coder/cron_loss.log
fi

# TELEGRAM NOTIFICATION via Hermes origin destination
TIMESTAMP2=$(date '+%Y-%m-%d %H:%M:%S')
echo "||TELEMETRY||origin||$TIMESTAMP2|Training status: $STATUS" >> /home/kenpeter/work/coder/telemetry.log
