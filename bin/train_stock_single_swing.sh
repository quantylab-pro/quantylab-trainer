#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
DATASET="${1:-stock_20260917}"
EPISODES="${2:-500}"

exec python -m quantylab.trainer.stock_single_swing.train \
  --base-path /home/quantylab/quantylab-trainer \
  --dataset "$DATASET" \
  --model-name stock-single-swing-v1 \
  --episodes "$EPISODES" \
  --device auto \
  --no-visualize
