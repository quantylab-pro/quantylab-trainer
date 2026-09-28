#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/quantylab/quantylab-trainer
RUN_ID=${RUN_ID:-stock-single-intraday-v1/eod-current-tax-exp01}
RUN_DIR="$ROOT/output/stock_single_intraday/$RUN_ID"
CACHE="$ROOT/output/stock_single_intraday/20260925_nminute_research_v2/features.pkl"
LOG="$RUN_DIR/run.log"
UNIT=${UNIT:-stock-single-intraday-v1-eod-current-tax-exp01}
SESSION=${SESSION:-stock-single-intraday-v1-eod-current-tax-exp01}

mkdir -p "$RUN_DIR"
if [[ ! -f "$CACHE" ]]; then
  echo "Missing prepared 5-minute feature cache: $CACHE" >&2
  exit 1
fi

if ! systemctl --user is-active --quiet "$UNIT.service"; then
  systemd-run --user --unit="$UNIT" --collect --no-block \
    --working-directory="$ROOT" \
    --property=MemoryMax=3G --property=CPUQuota=150% \
    --property=Restart=no \
    /bin/bash -c "exec /usr/bin/env EOD_RUN_ID=$RUN_ID /bin/bash $ROOT/bin/run_stock_single_intraday_eod_folds.sh >> $LOG 2>&1"
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
