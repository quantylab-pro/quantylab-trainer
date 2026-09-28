#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/quantylab/quantylab-trainer
PYTHON=/home/quantylab/miniconda3/bin/python
RUN_ID=${EOD_RUN_ID:-stock-single-intraday-v1/eod-current-tax-exp01}
OUTPUT="$ROOT/output/stock_single_intraday/$RUN_ID"
FEATURES_CACHE="$ROOT/output/stock_single_intraday/20260925_nminute_research_v2/features.pkl"
export PYTHONPATH="$ROOT/src"

for fold in {1..8}; do
  "$PYTHON" -m stock_single_intraday.research_eod \
    --output "$OUTPUT" \
    --features-cache "$FEATURES_CACHE" \
    --resume \
    --tax-policy current \
    --experiment-id eod-current-tax-exp01 \
    --fold-start "$fold" \
    --fold-end "$fold"
done
