#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/quantylab/quantylab-trainer
PYTHON=/home/quantylab/miniconda3/bin/python
FEATURES="$ROOT/output/stock_single_intraday/stock-single-intraday-v1/eod-current-tax-exp01/eod_features.pkl"
OUTPUT="$ROOT/output/stock_single_intraday/stock-single-intraday-v1/eod-exit-exp02"
export PYTHONPATH="$ROOT/src"

for fold in {1..8}; do
  checkpoint=$(printf '%s/fold_%02d.json' "$OUTPUT" "$fold")
  if [[ ! -f "$checkpoint" ]]; then
    "$PYTHON" -m stock_single_intraday.research_exit \
      --output "$OUTPUT" --features-cache "$FEATURES" --fold "$fold"
  fi
done

"$PYTHON" -m stock_single_intraday.research_exit \
  --output "$OUTPUT" --features-cache "$FEATURES" --finalize
