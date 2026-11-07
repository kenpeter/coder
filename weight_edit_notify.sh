#!/bin/bash
TOKEN="8346330809:AAFVV0rRH7E1_Lop8M_TPuO8oT5uop6YdWY"
CHAT="715681785"
LOG=/home/kenpeter/work/coder/weight_edit_9b.log
LAST=/tmp/opencode/last_he.txt
CUR=$(grep -o "HumanEval@[0-9]* [0-9]*/[0-9]*" "$LOG" | tail -1)
PREV=""
[ -f "$LAST" ] && PREV=$(cat "$LAST")
if [ "$CUR" != "$PREV" ] && [ -n "$CUR" ]; then
  curl -s -X POST "https://api.telegram.org/bot$TOKEN/sendMessage" -d "chat_id=$CHAT" -d "text=Weight edit update: $CUR (prev: ${PREV:-none})" >/dev/null
  echo "$CUR" > "$LAST"
fi