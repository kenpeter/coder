#!/bin/bash
# Check if the SAN training process is running
# Runs every 30 minutes via cron

PID_FILE="/home/kenpeter/work/coder/training.pid"
LOG_FILE="/home/kenpeter/work/coder/training.log"

# Check if training process is running
if pgrep -f "train_san.py" > /dev/null 2>&1; then
    STATUS="RUNNING"
    # Optionally update PID file
    pgrep -f "train_san.py" > "$PID_FILE" 2>/dev/null
else
    STATUS="STOPPED"
    # Optional: try to restart
    # /home/kenpeter/work/coder/run_train.sh &
fi

# Log status
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
echo "[$TIMESTAMP] Coder training status: $STATUS" >> "$LOG_FILE"

# Optional: append to hermes ticker files for Hermes-aware tracking
echo "$(date +%s.%N)" > /home/kenpeter/.hermes/profiles/agent2/cron/ticker_last_success