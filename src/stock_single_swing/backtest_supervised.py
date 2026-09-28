"""Backtest the stock-single-swing supervised forecast model."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

from .dataset import load_stock_sequences
from .supervised import evaluate_model, load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stock single swing 지도학습 모델 백테스트")
    parser.add_argument("--base-path", default="/home/quantylab/quantylab-trainer")
    parser.add_argument("--dataset", default="stock_20260917")
    parser.add_argument("--model", required=True, help="모델 디렉터리 이름 또는 절대경로")
    parser.add_argument("--start-date", default="20260101")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--checkpoint", choices=["best", "final"], default="best")
    parser.add_argument("--min-sequence-rows", type=int, default=30)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = args.model if os.path.isabs(args.model) else os.path.join(args.base_path, "models", args.model)
    config_path = os.path.join(model_dir, "train_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"모델 설정이 없습니다: {config_path}")
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)

    model_path = os.path.join(model_dir, f"model_{args.checkpoint}.joblib")
    model = load_model(model_path)
    sequences = load_stock_sequences(
        args.base_path,
        args.dataset,
        start_date=args.start_date,
        end_date=args.end_date,
        min_rows=args.min_sequence_rows,
    )
    env_kwargs = {
        "initial_balance": config.get("cost_assumptions", {}).get("initial_balance", 10_000_000.0),
        "trading_fee": config.get("cost_assumptions", {}).get("fee_one_way", 0.00015),
        "trading_tax": config.get("cost_assumptions", {}).get("stock_transaction_tax", 0.002),
        "slippage": config.get("cost_assumptions", {}).get("slippage_one_way", 0.0003),
        "hold_threshold": config.get("cost_assumptions", {}).get("hold_threshold", 0.10),
    }
    # The balance is a trading assumption rather than a cost; retain the
    # training default because older configs did not nest it in cost_assumptions.
    env_kwargs["initial_balance"] = config.get("initial_balance", env_kwargs["initial_balance"])
    aggregate, stats = evaluate_model(
        model,
        sequences,
        position_scale=config.get("position_scale", 0.01),
        environment_kwargs=env_kwargs,
    )
    output_dir = args.output_dir or os.path.join(
        args.base_path,
        "output",
        "stock_single_swing",
        datetime.now().strftime("%Y%m%d_%H%M%S") + "_backtest",
    )
    os.makedirs(output_dir, exist_ok=True)
    result = {
        "model": os.path.abspath(model_dir),
        "dataset": args.dataset,
        "period": {"start_date": args.start_date, "end_date": args.end_date},
        "checkpoint": args.checkpoint,
        "cost_assumptions": env_kwargs,
        "aggregate": aggregate,
        "by_stock": stats,
    }
    result_path = os.path.join(output_dir, "backtest_result.json")
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)

    manifest_path = os.path.join(model_dir, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest["locked_evaluation"] = {
            "start_date": args.start_date,
            "end_date": args.end_date,
            "checkpoint": args.checkpoint,
            "result_path": os.path.abspath(result_path),
            **aggregate,
        }
        manifest["deployment_approved"] = False
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)

    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    print(f"결과 저장: {output_dir}")


if __name__ == "__main__":
    main()
