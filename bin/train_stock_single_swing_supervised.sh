#!/usr/bin/env bash
set -euo pipefail

BASE_PATH="${1:-/home/quantylab/quantylab-trainer}"
DATASET="${2:-stock_20260917}"
OUTPUT_DIR="${3:-${BASE_PATH}/output/stock_single_swing/$(date +%Y%m%d_%H%M%S)_supervised}"

cd "${BASE_PATH}"
PYTHONPATH=src python -m quantylab.trainer.stock_single_swing.train_supervised \
  --base-path "${BASE_PATH}" \
  --dataset "${DATASET}" \
  --output-dir "${OUTPUT_DIR}" \
  --model-name stock-single-swing-v2 \
  --clean-run
