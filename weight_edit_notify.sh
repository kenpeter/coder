#!/bin/bash
# Data training cron script - runs every 30 minutes
# Shows data training status using basic bash parsing

TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Get the last line with step info from training log
STEP_LINE=$(grep "step" /home/kenpeter/work/coder/train_san.log | tail -1)

if [ -n "$STEP_LINE" ]; then
    # Extract step number (word after "step" until "|")
    STEP=$(echo "$STEP_LINE" | awk '{for(i=1;i<=NF;i++) if($i=="step") {print $(i+1); break}}')
    
    # Extract coverage percentage (after "data " until "%")
    COVERAGE=$(echo "$STEP_LINE" | awk -F'data ' '{print $2}' | awk -F'%' '{print $1}' | tr -d ' ')
    
    # Extract loss value (after "loss " until "[" or "|")
    LOSS=$(echo "$STEP_LINE" | awk -F'loss ' '{print $2}' | awk -F'[' '{print $1}' | awk -F'|' '{print $1}' | tr -d ' ')
else
    STEP="N/A"
    COVERAGE="N/A"
    LOSS="N/A"
fi

# Get per-domain values from the per-domain line
PER_LINE=$(grep "per-domain" /home/kenpeter/work/coder/train_san.log | tail -1)

if [ -n "$PER_LINE" ]; then
    # Remove "per-domain: " prefix, then extract 4 numeric values
    # Format: "code_easy 0.76 M-BM-7 code_gold 1.10 M-BM-7 code_hard 0.92 M-BM-7 code_mid 1.02"
    # Strip the prefix and M-BM-7 suffixes, get just the numbers
    DOMAIN_CONTENT=$(echo "$PER_LINE" | sed 's/.*per-domain: //' | tr -d '·' | tr -d ' ')
    
    # Extract all numbers from the content
    NUMBERS=$(echo "$DOMAIN_CONTENT" | grep -o '[0-9]*\.[0-9]*')
    
    # Assign to easy, gold, hard, mid based on order in the log
    # Typical order: code_easy, code_gold, code_hard, code_mid
    COUNT=0
    for n in $NUMBERS; do
        COUNT=$((COUNT+1))
        case $COUNT in
            1) EASY="$n" ;;
            2) GOLD="$n" ;;
            3) HARD="$n" ;;
            4) MID="$n" ;;
        esac
    done
    
    # If we don't have 4 numbers, set defaults
    EASY="${EASY:-N/A}"
    GOLD="${GOLD:-N/A}" 
    HARD="${HARD:-N/A}"
    MID="${MID:-N/A}"
else
    EASY="N/A"
    GOLD="N/A"
    HARD="N/A"
    MID="N/A"
fi

# Append to cron loss log with data training marker
echo "[$TIMESTAMP] DATA_TRAINING loss: $LOSS" >> /home/kenpeter/work/coder/cron_loss.log

# SEND MESSAGE BACK TO THIS CHAT (origin destination)
echo "||TELEMETRY||origin||$TIMESTAMP|DATA_TRAINING step $STEP coverage $COVERAGE% loss $LOSS easy $EASY gold $GOLD hard $HARD mid $MID||" >> /home/kenpeter/work/coder/telemetry.log
