#!/bin/bash
# 1분봉 단일 ETF PPO 학습 실행기
cd "$(dirname "$0")/.."

set -euo pipefail

DATA_SET="${1:-intraday_20260907}"
EPISODES="${2:-30}"
LR_POLICY="${3:-0.0001}"
LR_VALUE="${4:-0.0003}"
CHUNK_YEARS="${5:-1}"
OUTPUT_DIR="${6:-output/intraday_train}"
MAX_CHUNKS="${7:-0}"

TRAINING_FILE="data/$DATA_SET/training_scaled.csv"
if [ ! -f "$TRAINING_FILE" ]; then
    echo "학습 데이터가 없습니다: $TRAINING_FILE"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

exec python -m quantylab.trainer.etf_single_intraday.train \
    --dataset "$DATA_SET" \
    --trading-method swing \
    --episodes "$EPISODES" \
    --lr-policy "$LR_POLICY" \
    --lr-value "$LR_VALUE" \
    --chunk-years "$CHUNK_YEARS" \
    --max-chunks "$MAX_CHUNKS" \
    --log-dir "$OUTPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --gamma 0.995 \
    --hold-threshold 0.10 \
    --drawdown-penalty-scale 5.0 \
    --drawdown-penalty-threshold 0.12 \
    --rolling-sharpe-window 20 \
    --rolling-sharpe-scale 0.0 \
    --loss-aversion 1.0 \
    --policy-dropout 0.0 \
    --value-dropout 0.0 \
    --policy-weight-decay 0.0001 \
    --value-weight-decay 0.0003 \
    --validation-interval 5 \
    --early-stop-patience 0 \
    --network-type mamba \
    --device cuda \
    --no-visualize
