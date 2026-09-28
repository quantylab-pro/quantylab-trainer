#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/quantylab/quantylab-trainer
PYTHON=/home/quantylab/miniconda3/bin/python
FEATURES="$ROOT/output/stock_single_intraday/stock-single-intraday-v1/eod-current-tax-exp01/eod_features.pkl"
OUTPUT="$ROOT/output/stock_single_intraday/stock-single-intraday-v1/eod-exit-exp02"
LOG="$OUTPUT/run.log"
UNIT=stock-single-intraday-v1-eod-exit-exp02
SESSION=stock-single-intraday-v1-eod-exit-exp02

mkdir -p "$OUTPUT"
if [[ ! -f "$FEATURES" ]]; then
  echo "Missing exp01 feature cache: $FEATURES" >&2
  exit 1
fi

if ! systemctl --user is-active --quiet "$UNIT.service"; then
  systemd-run --user --unit="$UNIT" --collect --no-block \
    --working-directory="$ROOT" \
    --property=MemoryMax=3G --property=CPUQuota=150% --property=Restart=no \
    /bin/bash -c "exec /bin/bash $ROOT/bin/run_stock_single_intraday_exit_folds.sh >> $LOG 2>&1"
fi

if ! tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux new-session -d -s "$SESSION" -c "$ROOT" "tail -F '$LOG'"
  tmux split-window -v -t "$SESSION" "watch -n 3 'systemctl --user show $UNIT.service -p ActiveState -p Result -p NRestarts -p MemoryCurrent -p MemoryPeak'"
  tmux select-layout -t "$SESSION" main-horizontal
fi

echo "Research service: $UNIT.service"
echo "Attached progress console: tmux session $SESSION"
export TERM=xterm-256color
tmux attach-session -t "$SESSION"
