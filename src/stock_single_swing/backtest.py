"""Locked-period stock single-swing evaluation."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch

from ..etf_single_swing.agent import TradingAgent
from ..etf_single_swing.train import create_networks
from .dataset import load_stock_sequences
from .environment import StockSwingTradingEnvironment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stock single swing 잠금구간 백테스트")
    parser.add_argument("--base-path", default="/home/quantylab/quantylab-trainer")
    parser.add_argument("--dataset", default="stock_20260917")
    parser.add_argument("--model", required=True, help="모델 디렉터리 이름 또는 절대경로")
    parser.add_argument("--start-date", default="20260101")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--checkpoint", choices=["best", "final"], default="best")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--min-sequence-rows", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = args.model if os.path.isabs(args.model) else os.path.join(args.base_path, "models", args.model)
    config_path = os.path.join(model_dir, "train_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"모델 설정이 없습니다: {config_path}")
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA를 사용할 수 없습니다.")

    sequences = load_stock_sequences(
        args.base_path, args.dataset, start_date=args.start_date,
        end_date=args.end_date, min_rows=args.min_sequence_rows,
    )
    first_features = sequences[0][2]
    input_dim = first_features.shape[1] + 6
    policy_net, value_net = create_networks(
        input_dim, config.get("network_type", "mamba"), device,
        config.get("min_concentration", 1.5),
        d_model=config.get("d_model", 128), n_blocks=config.get("n_blocks", 3),
        d_state=config.get("d_state", 16),
        policy_dropout=config.get("policy_dropout", 0.15),
        value_dropout=config.get("value_dropout", 0.15),
    )
    agent = TradingAgent(policy_net, value_net, device=device, use_lstm=config.get("network_type") == "lstm")
    policy_path = os.path.join(model_dir, f"policy_{args.checkpoint}.pt")
    value_path = os.path.join(model_dir, f"value_{args.checkpoint}.pt")
    agent.load(policy_path, value_path)

    stats = []
    env_kwargs = {
        "initial_balance": config.get("initial_balance", 10_000_000.0),
        "trading_fee": config.get("trading_fee", 0.00015),
        "trading_tax": config.get("trading_tax", 0.002),
        "slippage": config.get("slippage", 0.0003),
        "hold_threshold": config.get("hold_threshold", 0.10),
        "reward_scale": config.get("reward_scale", 30.0),
        "fee_penalty_scale": config.get("fee_penalty_scale", 15.0),
    }
    for code, env_data, features in sequences:
        env = StockSwingTradingEnvironment(env_data, features, **env_kwargs)
        state = env.reset()
        done = False
        while not done:
            action, _, _ = agent.get_action(state, training=False)
            state, _, done, _ = env.step(action)
        item = env.get_stats()
        item["stock_code"] = code
        item["stock_name"] = str(env_data["stock_name"].iloc[0]) if "stock_name" in env_data else ""
        item["market_name"] = str(env_data["market_name"].iloc[0]) if "market_name" in env_data else ""
        stats.append(item)

    metric_names = ["profit_rate", "bnh_return", "excess_bnh", "max_drawdown", "sharpe_ratio", "num_buy"]
    aggregate = {f"mean_{key}": float(np.mean([s.get(key, 0.0) for s in stats])) for key in metric_names}
    aggregate.update({f"median_{key}": float(np.median([s.get(key, 0.0) for s in stats])) for key in metric_names})
    aggregate["stocks_evaluated"] = len(stats)
    aggregate["positive_excess_fraction"] = float(np.mean([s.get("excess_bnh", 0.0) > 0 for s in stats]))
    aggregate["positive_profit_fraction"] = float(np.mean([s.get("profit_rate", 0.0) > 0 for s in stats]))

    output_dir = args.output_dir or os.path.join(args.base_path, "output", "stock_single_swing_backtest", datetime.now().strftime("%Y%m%d_%H%M%S"))
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
    with open(os.path.join(output_dir, "backtest_result.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    manifest_path = os.path.join(model_dir, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest["locked_evaluation"] = {
            "start_date": args.start_date,
            "end_date": args.end_date,
            "checkpoint": args.checkpoint,
            "result_path": os.path.abspath(os.path.join(output_dir, "backtest_result.json")),
            **aggregate,
        }
        # A backtest records evidence; it does not by itself authorize live deployment.
        manifest["deployment_approved"] = False
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    print(f"결과 저장: {output_dir}")


if __name__ == "__main__":
    main()
