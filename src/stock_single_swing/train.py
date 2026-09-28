"""Train the stock-single-swing PPO policy."""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime, timezone

import numpy as np
import torch

from ..etf_single_swing.agent import TradingAgent
from ..etf_single_swing.train import create_networks
from ..etf_single_swing.trainer import PPOTrainer
from .dataset import load_stock_sequences
from .environment import RandomStockSwingEnvironment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stock single swing PPO 학습")
    parser.add_argument("--base-path", default="/home/quantylab/quantylab-trainer")
    parser.add_argument("--dataset", default="stock_20260917")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--model-name", default="stock-single-swing-v1")
    parser.add_argument("--train-end-date", default="20241231")
    parser.add_argument("--validation-start-date", default="20250101")
    parser.add_argument("--validation-end-date", default="20251231")
    parser.add_argument("--window-days", type=int, default=756)
    parser.add_argument("--min-sequence-rows", type=int, default=120)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--update-interval", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--network-type", choices=["standard", "lstm", "grn", "ft_transformer", "gmlp", "mamba"], default="mamba")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-blocks", type=int, default=3)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--policy-dropout", type=float, default=0.15)
    parser.add_argument("--value-dropout", type=float, default=0.15)
    parser.add_argument("--lr-policy", type=float, default=0.0001)
    parser.add_argument("--lr-value", type=float, default=0.0003)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--initial-balance", type=float, default=10_000_000.0)
    parser.add_argument("--trading-fee", type=float, default=0.00015)
    parser.add_argument("--trading-tax", type=float, default=0.002)
    parser.add_argument("--slippage", type=float, default=0.0003)
    parser.add_argument("--hold-threshold", type=float, default=0.10)
    parser.add_argument("--reward-scale", type=float, default=30.0)
    parser.add_argument("--fee-penalty-scale", type=float, default=15.0)
    parser.add_argument("--reward-terminal-scale", type=float, default=30.0)
    parser.add_argument("--drawdown-penalty-scale", type=float, default=25.0)
    parser.add_argument("--drawdown-penalty-threshold", type=float, default=0.12)
    parser.add_argument("--rolling-sharpe-window", type=int, default=20)
    parser.add_argument("--rolling-sharpe-scale", type=float, default=2.0)
    parser.add_argument("--loss-aversion", type=float, default=1.2)
    parser.add_argument("--min-concentration", type=float, default=1.5)
    parser.add_argument("--policy-weight-decay", type=float, default=1e-4)
    parser.add_argument("--value-weight-decay", type=float, default=3e-4)
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--early-stop-patience", type=int, default=35)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.10)
    parser.add_argument("--early-stop-warmup-episodes", type=int, default=120)
    parser.add_argument("--no-visualize", action="store_true")
    parser.add_argument("--clean-run", action="store_true")
    return parser.parse_args()


def _device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda를 지정했지만 CUDA를 사용할 수 없습니다.")
    return name


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _environment_kwargs(args: argparse.Namespace) -> dict:
    return {
        "initial_balance": args.initial_balance,
        "trading_fee": args.trading_fee,
        "trading_tax": args.trading_tax,
        "slippage": args.slippage,
        "reward_scale": args.reward_scale,
        "fee_penalty_scale": args.fee_penalty_scale,
        "reward_terminal_scale": args.reward_terminal_scale,
        "hold_threshold": args.hold_threshold,
        "drawdown_penalty_scale": args.drawdown_penalty_scale,
        "drawdown_penalty_threshold": args.drawdown_penalty_threshold,
        "rolling_sharpe_window": args.rolling_sharpe_window,
        "rolling_sharpe_scale": args.rolling_sharpe_scale,
        "loss_aversion": args.loss_aversion,
    }


def main() -> None:
    args = parse_args()
    _seed_everything(args.seed)
    device = _device(args.device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join(args.base_path, "output", "stock_single_swing", timestamp)
    log_dir = args.log_dir or os.path.join(output_dir, "logs")
    if args.clean_run and os.path.isdir(output_dir):
        for name in os.listdir(output_dir):
            path = os.path.join(output_dir, name)
            if os.path.isdir(path):
                import shutil
                shutil.rmtree(path)
            else:
                os.unlink(path)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    train_sequences = load_stock_sequences(
        args.base_path, args.dataset, end_date=args.train_end_date,
        min_rows=args.min_sequence_rows,
    )
    val_sequences = load_stock_sequences(
        args.base_path, args.dataset, start_date=args.validation_start_date,
        end_date=args.validation_end_date, min_rows=max(30, min(args.min_sequence_rows, 120)),
    )
    print(f"제품: {args.model_name}")
    print(f"데이터셋: {args.dataset}")
    print(f"Train stocks: {len(train_sequences):,}, Validation stocks: {len(val_sequences):,}")
    print(f"Device: {device}")

    env_kwargs = _environment_kwargs(args)
    train_env = RandomStockSwingEnvironment(
        train_sequences, window_days=args.window_days, randomize=True,
        seed=args.seed, **env_kwargs,
    )
    val_env = RandomStockSwingEnvironment(
        val_sequences, window_days=0, randomize=False,
        seed=args.seed + 1, **env_kwargs,
    )
    input_dim = train_env.num_features
    policy_net, value_net = create_networks(
        input_dim, args.network_type, device, args.min_concentration,
        d_model=args.d_model, n_blocks=args.n_blocks, d_state=args.d_state,
        policy_dropout=args.policy_dropout, value_dropout=args.value_dropout,
    )
    agent = TradingAgent(
        policy_network=policy_net,
        value_network=value_net,
        lr_policy=args.lr_policy,
        lr_value=args.lr_value,
        gamma=args.gamma,
        policy_weight_decay=args.policy_weight_decay,
        value_weight_decay=args.value_weight_decay,
        device=device,
        use_lstm=(args.network_type == "lstm"),
    )

    config = {
        "model_name": args.model_name,
        "model_type": "stock-single-swing",
        "dataset": args.dataset,
        "train_end_date": args.train_end_date,
        "validation_start_date": args.validation_start_date,
        "validation_end_date": args.validation_end_date,
        "locked_evaluation_start_date": "20260101",
        "initial_balance": args.initial_balance,
        "trading_fee": args.trading_fee,
        "trading_tax": args.trading_tax,
        "slippage": args.slippage,
        "reward_scale": args.reward_scale,
        "fee_penalty_scale": args.fee_penalty_scale,
        "reward_terminal_scale": args.reward_terminal_scale,
        "min_concentration": args.min_concentration,
        "policy_dropout": args.policy_dropout,
        "value_dropout": args.value_dropout,
        "train_stocks": len(train_sequences),
        "validation_stocks": len(val_sequences),
        "feature_dim": input_dim - train_env.PORTFOLIO_FEATURE_NUM,
        "input_dim": input_dim,
        "episodes": args.episodes,
        "window_days": args.window_days,
        "network_type": args.network_type,
        "d_model": args.d_model,
        "n_blocks": args.n_blocks,
        "d_state": args.d_state,
        "lr_policy": args.lr_policy,
        "lr_value": args.lr_value,
        "gamma": args.gamma,
        "hold_threshold": args.hold_threshold,
        "trading_fee": args.trading_fee,
        "trading_tax": args.trading_tax,
        "slippage": args.slippage,
        "seed": args.seed,
        "device": device,
        "deployment_approved": False,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(output_dir, "train_config.json"), "w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)

    trainer = PPOTrainer(
        env=train_env,
        agent=agent,
        num_episodes=args.episodes,
        update_interval=args.update_interval,
        log_dir=log_dir,
        output_dir=output_dir,
        visualize=not args.no_visualize,
        entropy_coef_start=0.05,
        entropy_coef_end=0.01,
        entropy_decay_episodes=min(300, args.episodes),
        target_bias_low=0.10,
        target_bias_high=0.90,
        trade_rate_threshold=0.15,
        action_mix_start=0.05,
        action_mix_end=0.01,
        validation_interval=args.validation_interval,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        early_stop_warmup_episodes=args.early_stop_warmup_episodes,
        val_env=val_env,
        chunk_info={
            "iteration_name": timestamp,
            "model_name": args.model_name,
            "product": "stock-single-swing",
            "dataset": args.dataset,
        },
    )
    print("학습 시작")
    trainer.train()

    manifest = {
        **config,
        "candidate_output_dir": os.path.abspath(output_dir),
        "checkpoints": ["policy_best.pt", "value_best.pt", "policy_final.pt", "value_final.pt"],
        "validation_artifact": os.path.join(os.path.abspath(log_dir), "episodes.jsonl"),
        "cost_assumptions": {
            "fee_one_way": args.trading_fee,
            "stock_transaction_tax": args.trading_tax,
            "slippage_one_way": args.slippage,
            "execution": "previous-day feature -> next trading-day open -> close mark",
        },
        "deployment_approved": False,
    }
    with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(f"학습 완료: {output_dir}")


if __name__ == "__main__":
    main()
