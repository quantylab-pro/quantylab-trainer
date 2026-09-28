"""Train the stock-single-swing-v2 supervised forecast model."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from datetime import datetime, timezone

import numpy as np
import torch

from .dataset import load_stock_sequences
from .supervised import fit_forward_model, load_training_frame, make_forward_target, save_model, evaluate_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stock single swing 지도학습")
    parser.add_argument("--base-path", default="/home/quantylab/quantylab-trainer")
    parser.add_argument("--dataset", default="stock_20260917")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model-name", default="stock-single-swing-v2")
    parser.add_argument("--train-end-date", default="20241231")
    parser.add_argument("--validation-start-date", default="20250101")
    parser.add_argument("--validation-end-date", default="20251231")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument(
        "--position-scale",
        type=float,
        default=0.10,
        help="Sigmoid calibration temperature for the bounded position score",
    )
    parser.add_argument("--max-iter", type=int, default=180)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--min-samples-leaf", type=int, default=100)
    parser.add_argument("--l2-regularization", type=float, default=1.0)
    parser.add_argument("--min-sequence-rows", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial-balance", type=float, default=10_000_000.0)
    parser.add_argument("--trading-fee", type=float, default=0.00015)
    parser.add_argument("--trading-tax", type=float, default=0.002)
    parser.add_argument("--slippage", type=float, default=0.0003)
    parser.add_argument("--hold-threshold", type=float, default=0.10)
    parser.add_argument("--clean-run", action="store_true")
    return parser.parse_args()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _environment_kwargs(args: argparse.Namespace) -> dict:
    return {
        "initial_balance": args.initial_balance,
        "trading_fee": args.trading_fee,
        "trading_tax": args.trading_tax,
        "slippage": args.slippage,
        "hold_threshold": args.hold_threshold,
    }


def main() -> None:
    args = parse_args()
    _seed_everything(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join(
        args.base_path, "output", "stock_single_swing", timestamp + "_supervised"
    )
    if args.clean_run and os.path.isdir(output_dir):
        for name in os.listdir(output_dir):
            path = os.path.join(output_dir, name)
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.unlink(path)
    os.makedirs(output_dir, exist_ok=True)

    env, features, feature_names = load_training_frame(args.base_path, args.dataset)
    target, future_date = make_forward_target(env, args.horizon)
    train_end = int(args.train_end_date)
    train_mask = (
        (env["date"].to_numpy() <= train_end)
        & (future_date <= train_end)
        & np.isfinite(target)
    )
    print(f"제품: {args.model_name}")
    print(f"데이터셋: {args.dataset}, rows={len(env):,}, features={features.shape[1]}")
    print(f"Forward target: {args.horizon} trading days, train labels={int(train_mask.sum()):,}")

    model = fit_forward_model(
        features,
        target,
        train_mask,
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        max_leaf_nodes=args.max_leaf_nodes,
        min_samples_leaf=args.min_samples_leaf,
        l2_regularization=args.l2_regularization,
        random_state=args.seed,
    )
    save_model(model, os.path.join(output_dir, "model_best.joblib"))
    save_model(model, os.path.join(output_dir, "model_final.joblib"))

    validation_sequences = load_stock_sequences(
        args.base_path,
        args.dataset,
        start_date=args.validation_start_date,
        end_date=args.validation_end_date,
        min_rows=args.min_sequence_rows,
    )
    validation_aggregate, _ = evaluate_model(
        model,
        validation_sequences,
        position_scale=args.position_scale,
        environment_kwargs=_environment_kwargs(args),
    )
    print(json.dumps(validation_aggregate, ensure_ascii=False, indent=2))

    config = {
        "model_name": args.model_name,
        "model_type": "stock-single-swing-supervised",
        "dataset": args.dataset,
        "train_end_date": args.train_end_date,
        "validation_start_date": args.validation_start_date,
        "validation_end_date": args.validation_end_date,
        "locked_evaluation_start_date": "20260101",
        "feature_names": feature_names,
        "feature_dim": len(feature_names),
        "target": {
            "type": "forward_log_close_return",
            "horizon_trading_days": args.horizon,
            "label_cutoff": args.train_end_date,
            "execution": "feature at t-1 -> next trading-day open -> position held and rebalanced daily",
        },
        "position_scale": args.position_scale,
        "model_params": {
            "estimator": "HistGradientBoostingRegressor",
            "max_iter": args.max_iter,
            "learning_rate": args.learning_rate,
            "max_leaf_nodes": args.max_leaf_nodes,
            "min_samples_leaf": args.min_samples_leaf,
            "l2_regularization": args.l2_regularization,
            "seed": args.seed,
        },
        "cost_assumptions": {
            "fee_one_way": args.trading_fee,
            "stock_transaction_tax": args.trading_tax,
            "slippage_one_way": args.slippage,
            "hold_threshold": args.hold_threshold,
        },
        "train_rows": int(train_mask.sum()),
        "validation_stocks": len(validation_sequences),
        "validation": validation_aggregate,
        "selection_metric": "validation_mean_excess_bnh",
        "deployment_approved": False,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(output_dir, "train_config.json"), "w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)

    manifest = {
        **config,
        "candidate_output_dir": os.path.abspath(output_dir),
        "checkpoints": ["model_best.joblib", "model_final.joblib"],
        "validation_artifact": os.path.abspath(os.path.join(output_dir, "train_config.json")),
        "deployment_approved": False,
    }
    with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(f"학습 완료: {output_dir}")


if __name__ == "__main__":
    main()
