#!/bin/bash
# Per-minute SAN training report -> Telegram (agent 2).
# Runs every minute via cron: * * * * *
# Writes ||TELEMETRY||agent2||<ts>|<report>|| to telemetry.log so the Hermes
# system delivers it to the Telegram agent-2 chat.
set -u

CODER=/home/kenpeter/work/coder
LOG=$CODER/train_san.log
TELEMETRY=$CODER/telemetry.log
PIDFILE=$CODER/training.pid
TICKER_HB=$CODER/ticker_heartbeat
TICKER_OK=$CODER/ticker_last_success
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# ---- training process status ----
if pgrep -f "python3.*train_san.py" > /dev/null 2>&1; then
    STATUS="RUNNING"
    pgrep -f "train_san.py" > "$PIDFILE" 2>/dev/null
else
    STATUS="STOPPED"
fi

# ---- parse latest step line from train_san.log ----
STEP="N/A"; LOSS="N/A"; MAIN="N/A"; TOKS="N/A"; COVERAGE="N/A"; EPOCH="N/A"
if [ -f "$LOG" ]; then
    STEP_LINE=$(grep "step " "$LOG" | tail -1)
    if [ -n "$STEP_LINE" ]; then
        STEP=$(echo "$STEP_LINE" | grep -oP 'step \K[0-9]+' | head -1)
        LOSS=$(echo "$STEP_LINE" | grep -oP 'loss \K[0-9.]+' | head -1)
        MAIN=$(echo "$STEP_LINE" | grep -oP 'main [0-9.]+' | grep -oP '[0-9.]+' | head -1)
        TOKS=$(echo "$STEP_LINE" | grep -oP '[0-9,]+(?= tok/s)' | head -1 | tr -d ',')
        COVERAGE=$(echo "$STEP_LINE" | grep -oP 'data \K[0-9.]+' | head -1)
        EPOCH=$(echo "$STEP_LINE" | grep -oP '\([0-9.]+ epochs?\)' | tr -d '()epochs ' | head -1)
    fi
fi

# ---- GPU stats ----
GPU_UTIL="N/A"; POWER="N/A"; VRAM="N/A"
if command -v nvidia-smi > /dev/null 2>&1; then
    GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    POWER=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    VRAM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
fi

# ---- build report ----
REPORT="SAN training $STATUS | step ${STEP} loss ${LOSS} (main ${MAIN}) | ${TOKS} tok/s | data ${COVERAGE}% (${EPOCH}) | GPU ${GPU_UTIL}% | P ${POWER}W | VRAM ${VRAM}MiB"

# ---- telemetry to agent 2 ----
echo "||TELEMETRY||agent2||$TIMESTAMP|$REPORT||" >> "$TELEMETRY"

# ---- tickers ----
echo "$TIMESTAMP $STATUS step=$STEP loss=$LOSS toks=$TOKS" > "$TICKER_HB"
if [ "$STATUS" = "RUNNING" ]; then
    echo "$TIMESTAMP step=$STEP loss=$LOSS toks=$TOKS gpu=$GPU_UTIL power=$POWER" > "$TICKER_OK"
fi
