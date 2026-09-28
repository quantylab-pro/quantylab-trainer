"""Supervised stock swing model utilities.

The original stock single-swing candidate used PPO to learn a continuous
position target.  This module provides a simpler, directly supervised
alternative: predict the forward return available from the information date
and translate that forecast into a bounded target position.
"""

from __future__ import annotations

import os
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .environment import StockSwingTradingEnvironment


def load_training_frame(base_path: str, dataset: str):
    """Load the aligned stock environment and scaled feature matrix."""

    dataset_dir = dataset if os.path.isabs(dataset) else os.path.join(base_path, "data", dataset)
    env_path = os.path.join(dataset_dir, "environment.csv")
    feature_path = os.path.join(dataset_dir, "training_scaled.csv")
    env = pd.read_csv(env_path)
    feature_frame = pd.read_csv(feature_path)
    if len(env) != len(feature_frame):
        raise ValueError(f"stock 데이터 정렬 불일치: env={len(env)}, features={len(feature_frame)}")

    env = env.copy()
    env["stock_code"] = env["stock_code"].astype(str).str.zfill(6)
    env["date"] = env["date"].astype(int)
    return env, feature_frame.to_numpy(dtype=np.float32), list(feature_frame.columns)


def make_forward_target(env: pd.DataFrame, horizon: int):
    """Return log forward returns and their future dates, without look-ahead."""

    if horizon < 1:
        raise ValueError("horizon은 1 이상이어야 합니다.")
    grouped = env.groupby("stock_code", sort=False)
    future_close = grouped["close"].shift(-horizon)
    future_date = grouped["date"].shift(-horizon)
    with np.errstate(divide="ignore", invalid="ignore"):
        target = np.log(future_close.to_numpy(dtype=float) / env["close"].to_numpy(dtype=float))
    target = np.clip(target, -0.8, 0.8).astype(np.float32)
    target[~np.isfinite(target)] = np.nan
    return target, future_date.to_numpy()


def fit_forward_model(
    features: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    max_iter: int = 180,
    learning_rate: float = 0.05,
    max_leaf_nodes: int = 31,
    min_samples_leaf: int = 100,
    l2_regularization: float = 1.0,
    random_state: int = 42,
) -> HistGradientBoostingRegressor:
    """Fit a robust forward-return model on the time-boxed training rows."""

    valid = np.asarray(mask, dtype=bool) & np.isfinite(target)
    if int(valid.sum()) < 1000:
        raise ValueError(f"학습 가능한 label이 너무 적습니다: {int(valid.sum())}")
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        random_state=random_state,
    )
    model.fit(features[valid], target[valid])
    return model


def prediction_to_position(prediction: np.ndarray | float, position_scale: float = 0.10):
    """Map a forward log-return forecast to a smooth position in [0, 1].

    The old linear mapping used a 1% scale and clipped values outside the
    interval.  The supervised model forecasts 20-day log returns, which are
    often materially larger than 1%; that made a large part of the universe
    indistinguishable at exactly 0 or 1.  A sigmoid preserves the ordering
    without hard saturation while keeping the existing position contract.
    """

    if position_scale <= 0:
        raise ValueError("position_scale은 0보다 커야 합니다.")
    pred = np.asarray(prediction, dtype=np.float32)
    with np.errstate(over="ignore", under="ignore"):
        return 1.0 / (1.0 + np.exp(-np.clip(pred / position_scale, -60.0, 60.0)))


def _aggregate(stats: list[dict]) -> dict:
    metric_names = [
        "profit_rate", "bnh_return", "excess_bnh", "max_drawdown",
        "sharpe_ratio", "num_buy", "total_fee_paid",
    ]
    result = {
        f"mean_{name}": float(np.mean([item.get(name, 0.0) for item in stats]))
        for name in metric_names
    }
    result.update({
        f"median_{name}": float(np.median([item.get(name, 0.0) for item in stats]))
        for name in metric_names
    })
    result["stocks_evaluated"] = len(stats)
    result["positive_excess_fraction"] = float(
        np.mean([item.get("excess_bnh", 0.0) > 0 for item in stats])
    )
    result["positive_profit_fraction"] = float(
        np.mean([item.get("profit_rate", 0.0) > 0 for item in stats])
    )
    return result


def evaluate_model(
    model,
    sequences: Iterable[tuple[str, pd.DataFrame, np.ndarray]],
    *,
    position_scale: float = 0.01,
    environment_kwargs: dict | None = None,
) -> tuple[dict, list[dict]]:
    """Evaluate one forecast model with the real stock swing environment."""

    env_kwargs = dict(environment_kwargs or {})
    stats = []
    for code, env_data, features in sequences:
        predictions = np.asarray(model.predict(features), dtype=np.float32).reshape(-1)
        positions = prediction_to_position(predictions, position_scale)
        environment = StockSwingTradingEnvironment(env_data, features, **env_kwargs)
        state = environment.reset()
        done = False
        step = 0
        while not done:
            # The state at step t uses feature row t-1, so prediction row t-1
            # is the forecast available for the next trading day.
            state, _, done, _ = environment.step(float(positions[step]))
            step += 1
        item = environment.get_stats()
        item["stock_code"] = str(code).zfill(6)
        item["stock_name"] = str(env_data["stock_name"].iloc[0]) if "stock_name" in env_data else ""
        item["market_name"] = str(env_data["market_name"].iloc[0]) if "market_name" in env_data else ""
        item["mean_position"] = float(np.mean(positions[:max(step, 1)]))
        stats.append(item)
    if not stats:
        raise ValueError("평가 가능한 stock sequence가 없습니다.")
    return _aggregate(stats), stats


def save_model(model, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # HistGradientBoostingRegressor stores a NumPy Generator used only while
    # fitting.  Removing it from the serialized artifact keeps a model trained
    # with NumPy 2.x loadable by the production worker on NumPy 1.26.x.
    feature_subsample_rng = getattr(model, "_feature_subsample_rng", None)
    if feature_subsample_rng is not None:
        model._feature_subsample_rng = None
    try:
        joblib.dump(model, path, compress=3)
    finally:
        if feature_subsample_rng is not None:
            model._feature_subsample_rng = feature_subsample_rng


def load_model(path: str):
    return joblib.load(path)
