#!/bin/bash
# Data training cron script - runs every 30 minutes
# Shows data training status using Python for robust log parsing

TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Use Python to parse the training log reliably
PYTHON_RESULT=$(/home/kenpeter/.hermes/tools/python-3.14.9+202****1021-linux-x64/bin/python3 /home/kenpeter/work/coder/parse_domain.py 2>/dev/null)

if [ -n "$PYTHON_RESULT" ]; then
    # Parse the Python output
    # Format: step=N/A coverage=N/A loss=N/A easy=N/A mid=N/A hard=N/A gold=N/A
    # But actually it prints: easy=0.72 mid=0.86 hard=1.00 gold=0.81
    # And step/coverage/loss are set separately
    
    # Extract the domain values
    EASY=$(echo "$PYTHON_RESULT" | grep -oP 'easy=\K[0-9.]+')
    MID=$(echo "$PYTHON_RESULT" | grep -oP 'mid=\K[0-9.]+')
    HARD=$(echo "$PYTHON_RESULT" | grep -oP 'hard=\K[0-9.]+')
    GOLD=$(echo "$PYTHON_RESULT" | grep -oP 'gold=\K[0-9.]+')
    
    # For step, coverage, loss - we need to get those from the training log directly
    STEP_LINE=$(grep "step" /home/kenpeter/work/coder/train_san.log | tail -1)
    if [ -n "$STEP_LINE" ]; then
        STEP=$(echo "$STEP_LINE" | awk '{for(i=1;i<=NF;i++) if($i=="step") {print $(i+1); break}}')
        COVERAGE=$(echo "$STEP_LINE" | awk -F'data ' '{print $2}' | awk -F'%' '{print $1}' | tr -d ' ')
        LOSS=$(echo "$STEP_LINE" | awk -F'loss ' '{print $2}' | awk -F'[' '{print $1}' | awk -F'|' '{print $1}' | tr -d ' ')
    else
        STEP="N/A"
        COVERAGE="N/A"
        LOSS="N/A"
    fi
else
    STEP="N/A"
    COVERAGE="N/A"
    LOSS="N/A"
    EASY="N/A"
    MID="N/A"
    HARD="N/A"
    GOLD="N/A"
fi

# Append to cron loss log with data training marker
echo "[$TIMESTAMP] DATA_TRAINING loss: $LOSS" >> /home/kenpeter/work/coder/cron_loss.log

# SEND MESSAGE BACK TO THIS CHAT (origin destination)
echo "||TELEMETRY||origin||$TIMESTAMP|DATA_TRAINING step $STEP coverage $COVERAGE% loss $LOSS easy $EASY gold $GOLD hard $HARD mid $MID||" >> /home/kenpeter/work/coder/telemetry.log
