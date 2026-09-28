#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/quantylab/quantylab-trainer
RUN_DIR="$ROOT/output/stock_single_intraday/20260926_eod_close_v6"
UNIT=stock-single-intraday-cost-stress-v1
SESSION=stock-single-intraday-cost-stress
LOG="$RUN_DIR/cost_stress.log"
FEATURES="$RUN_DIR/eod_features.pkl"

if ! systemctl --user is-active --quiet "$UNIT.service"; then
  systemd-run --user --unit="$UNIT" --collect --no-block \
    --working-directory="$ROOT" \
    --property=MemoryMax=3G --property=CPUQuota=100% --property=Restart=no \
    /bin/bash -c "exec /usr/bin/env PYTHONPATH=$ROOT/src /home/quantylab/miniconda3/bin/python -m stock_single_intraday.audit_cost_stress --features $FEATURES --output $RUN_DIR/cost_stress.json >> $LOG 2>&1"
fi

if ! tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux new-session -d -s "$SESSION" -c "$ROOT" "tail -F '$LOG'"
  tmux split-window -v -t "$SESSION" "watch -n 3 'systemctl --user show $UNIT.service -p ActiveState -p Result -p NRestarts -p MemoryCurrent -p MemoryPeak'"
  tmux select-layout -t "$SESSION" main-horizontal
fi

echo "Cost stress service: $UNIT.service"
echo "Attached progress console: tmux session $SESSION"
export TERM=xterm-256color
tmux attach-session -t "$SESSION"
