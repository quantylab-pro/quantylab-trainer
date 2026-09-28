"""Research stock single/portfolio swing alpha candidates with next-open fills.

The historical universe contains only surviving stocks. All evaluation here is
retrospective; this script never promotes a model or submits orders.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


def metrics(returns: np.ndarray, benchmark: np.ndarray, turnover: np.ndarray) -> dict:
    wealth = np.cumprod(1 + returns)
    base = np.cumprod(1 + benchmark)
    peak = np.maximum.accumulate(np.r_[1.0, wealth])[1:]
    return {
        "return": float(wealth[-1] - 1),
        "benchmark_return": float(base[-1] - 1),
        "excess_percentage_points": float((wealth[-1] - base[-1]) * 100),
        "sharpe": float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() else 0.0,
        "max_drawdown": float(np.max(1 - wealth / peak)),
        "mean_turnover": float(turnover.mean()),
        "days": int(len(returns)),
    }


def make_target(env: pd.DataFrame, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Forward stock log return minus same-exchange equal-weight market proxy."""
    prices = env.pivot(index="date", columns="stock_code", values="close").sort_index()
    markets = env.drop_duplicates("stock_code").set_index("stock_code")["market_name"]
    daily = np.log(prices / prices.shift(1))
    alpha_by_market = {}
    for market in ("kospi", "kosdaq"):
        cols = markets[markets.eq(market)].index.intersection(prices.columns)
        market_daily = daily[cols].mean(axis=1)
        # At t, the target is returns from t+1 through t+h.
        alpha_by_market[market] = market_daily.shift(-1).iloc[::-1].rolling(horizon, min_periods=horizon).sum().iloc[::-1]
    forward = env.groupby("stock_code", sort=False)["close"].shift(-horizon)
    future_date = env.groupby("stock_code", sort=False)["date"].shift(-horizon).fillna("")
    raw = np.log(forward.to_numpy(float) / env["close"].to_numpy(float))
    market_forward = np.full(len(env), np.nan)
    for market, series in alpha_by_market.items():
        mask = env["market_name"].eq(market).to_numpy()
        market_forward[mask] = env.loc[mask, "date"].map(series).to_numpy(float)
    return np.clip(raw - market_forward, -0.8, 0.8).astype(np.float32), future_date.to_numpy()


def single_eval(env: pd.DataFrame, pred: np.ndarray, start: str, end: str, threshold: float) -> dict:
    frame = env.loc[env["date"].between(start, end), ["stock_code", "date", "open"]].copy()
    frame["pred"] = pred[env["date"].between(start, end).to_numpy()]
    rows = []
    for code, stock in frame.groupby("stock_code", sort=False):
        stock = stock.sort_values("date")
        price = stock["open"].to_numpy(float)
        if len(price) < 30 or np.any(price <= 0):
            continue
        signal = stock["pred"].to_numpy(float)
        position = (signal[:-2] > threshold).astype(float)
        next_return = price[2:] / price[1:-1] - 1
        prior = np.r_[0.0, position[:-1]]
        turnover = np.abs(position - prior)
        cost = np.maximum(position - prior, 0) * 0.00045 + np.maximum(prior - position, 0) * 0.00245
        strategy = position * next_return - cost
        # Liquidate at the final mark so the strategy and buy-and-hold
        # comparator both pay an exit cost over the same window.
        if position[-1] > 0:
            strategy[-1] -= 0.00245
        buy_hold = next_return.copy()
        # A buy-and-hold account pays entry and exit costs once.
        buy_hold[0] -= 0.00045
        buy_hold[-1] -= 0.00245
        row = metrics(strategy, buy_hold, turnover)
        row["stock_code"] = code
        row["active_fraction"] = float(position.mean())
        rows.append(row)
    if not rows:
        raise ValueError("single evaluation has no eligible stocks")
    return {
        "stocks": len(rows),
        "mean_return": float(np.mean([r["return"] for r in rows])),
        "mean_buy_hold_return": float(np.mean([r["benchmark_return"] for r in rows])),
        "mean_excess_percentage_points": float(np.mean([r["excess_percentage_points"] for r in rows])),
        "median_excess_percentage_points": float(np.median([r["excess_percentage_points"] for r in rows])),
        "positive_excess_fraction": float(np.mean([r["excess_percentage_points"] > 0 for r in rows])),
        "mean_turnover": float(np.mean([r["mean_turnover"] for r in rows])),
        "mean_active_fraction": float(np.mean([r["active_fraction"] for r in rows])),
        "by_stock": rows,
    }


def portfolio_eval(env: pd.DataFrame, pred: np.ndarray, codes: list[str], start: str, end: str, top_k: int,
                   cost_multiplier: float = 1.0) -> dict:
    mask = env["stock_code"].isin(codes) & env["date"].between(start, end)
    sub = env.loc[mask, ["date", "stock_code", "open"]].copy()
    sub["pred"] = pred[mask.to_numpy()]
    price = sub.pivot(index="date", columns="stock_code", values="open").reindex(columns=codes).sort_index()
    forecast = sub.pivot(index="date", columns="stock_code", values="pred").reindex(columns=codes).sort_index()
    if price.isna().any().any() or forecast.isna().any().any():
        raise ValueError("portfolio common calendar is incomplete")
    open_values = price.to_numpy(float)
    scores = forecast.to_numpy(float)
    previous = np.zeros(len(codes))
    daily, benchmark, turns = [], [], []
    for i in range(len(price) - 2):
        # Signal after close i; fill at open i+1; mark at open i+2.
        order = np.argsort(-scores[i], kind="stable")[:top_k]
        desired = np.zeros(len(codes))
        selected = order[scores[i, order] > 0]
        if len(selected):
            desired[selected] = 1.0 / len(selected)
        ret = open_values[i + 2] / open_values[i + 1] - 1
        buys = np.maximum(desired - previous, 0).sum()
        sells = np.maximum(previous - desired, 0).sum()
        final_exit = desired.sum() * 0.00245 if i == len(price) - 3 else 0.0
        daily.append(float(desired @ ret - cost_multiplier * (buys * 0.00045 + sells * 0.00245 + final_exit)))
        benchmark.append(float(np.mean(ret)))
        turns.append(float(buys + sells))
        # Mark-to-market weights before the next rebalance.
        grown = desired * (1 + ret)
        nav = grown.sum() + (1 - desired.sum())
        previous = grown / nav
    result = metrics(np.asarray(daily), np.asarray(benchmark), np.asarray(turns))
    result.update({"top_k": top_k, "cost_multiplier": cost_multiplier,
                   "benchmark": "same 100-stock equal-weight open-to-open proxy",
                   "execution": "signal at close t, trade at open t+1, mark at open t+2"})
    return result


def train(features: np.ndarray, target: np.ndarray, env: pd.DataFrame, future_date: np.ndarray, seed: int,
          train_end: str = "20241231") -> HistGradientBoostingRegressor:
    dates = env["date"].to_numpy()
    mask = (dates <= train_end) & (future_date <= train_end) & np.isfinite(target)
    model = HistGradientBoostingRegressor(max_iter=160, learning_rate=0.045,
        max_leaf_nodes=24, min_samples_leaf=150, l2_regularization=3.0, random_state=seed)
    model.fit(features[mask], target[mask])
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="/home/quantylab/quantylab-trainer/data/stock_20260917")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    dataset = Path(args.dataset)
    output = Path(args.output or f"/home/quantylab/quantylab-trainer/output/stock_swing_alpha/{datetime.now():%Y%m%d_%H%M%S}")
    output.mkdir(parents=True, exist_ok=False)
    env = pd.read_csv(dataset / "environment.csv", dtype={"date": str, "stock_code": str}, low_memory=False)
    env["stock_code"] = env["stock_code"].str.zfill(6)
    features = pd.read_csv(dataset / "training_scaled.csv", dtype=np.float32).to_numpy(np.float32)
    if len(env) != len(features):
        raise ValueError("environment/features length mismatch")
    counts = env.groupby("stock_code")["date"].size().sort_values(ascending=False)
    codes = sorted(counts.head(100).index.tolist())
    manifest = {"dataset": str(dataset), "train_end": "20241231", "retrospective_periods": ["2025", "2026"],
                "feature_dim": int(features.shape[1]), "universe_warning": "200 surviving stocks; survivorship bias; no independent holdout",
                "deployment_approved": False, "costs": {"buy": 0.00045, "sell": 0.00245},
                "market_target": "same-exchange equal-weight future close return proxy"}
    for product, horizon in (("stock-single-swing", 5), ("stock-portfolio-swing", 20)):
        print(f"training {product} horizon={horizon}", flush=True)
        target, future_date = make_target(env, horizon)
        model = train(features, target, env, future_date, seed=42)
        predictions = model.predict(features).astype(np.float32)
        product_dir = output / product
        product_dir.mkdir()
        joblib.dump(model, product_dir / "model.joblib", compress=3)
        shutil.copy2(dataset / "scaler.pkl", product_dir / "scaler.pkl")
        config = {"product": product, "dataset": str(dataset), "feature_names": list(pd.read_csv(dataset / "training_scaled.csv", nrows=0).columns),
                  "target": {"type": "future_log_return_minus_same_market_equal_weight_proxy", "horizon_trading_days": horizon},
                  "execution": "signal at close t; fill at next open; evaluate to following open",
                  "train_end_date": "20241231", "scaler_fit_end_date": "20241231", "seed": 42,
                  "model_params": {"estimator": "HistGradientBoostingRegressor", "max_iter": 160,
                                   "learning_rate": 0.045, "max_leaf_nodes": 24, "min_samples_leaf": 150,
                                   "l2_regularization": 3.0}, "deployment_approved": False}
        (product_dir / "train_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2))
        if product == "stock-single-swing":
            validation = {str(t): single_eval(env, predictions, "20250101", "20251231", t) for t in (0.0, 0.005, 0.01)}
            chosen = max(validation, key=lambda t: validation[t]["mean_excess_percentage_points"])
            retrospective = single_eval(env, predictions, "20260101", "20260916", float(chosen))
            manifest[product] = {"horizon": horizon, "threshold": float(chosen),
                "validation": {k: {m: v for m, v in val.items() if m != "by_stock"} for k, val in validation.items()},
                "retrospective": {k: v for k, v in retrospective.items() if k != "by_stock"}}
            for name, obj in (("validation", validation), ("retrospective_2026", retrospective)):
                (product_dir / f"{name}.json").write_text(json.dumps(obj, ensure_ascii=False, indent=2))
        else:
            validation = {str(k): portfolio_eval(env, predictions, codes, "20250101", "20251231", k) for k in (5, 10, 20)}
            chosen = max(validation, key=lambda k: validation[k]["excess_percentage_points"])
            retrospective = portfolio_eval(env, predictions, codes, "20260101", "20260916", int(chosen))
            stress = {str(multiplier): portfolio_eval(env, predictions, codes, "20260101", "20260916", int(chosen), multiplier)
                      for multiplier in (2.0, 4.0)}
            manifest[product] = {"horizon": horizon, "top_k": int(chosen), "universe": codes,
                                  "validation": validation, "retrospective": retrospective, "cost_stress_2026": stress}
        print(json.dumps({product: manifest[product]}, ensure_ascii=False)[:2500], flush=True)
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        product_manifest = {**config, "model_name": f"{product}-alpha-research",
                            "run_directory": str(output.resolve()), "candidate_metrics": manifest[product],
                            "artifacts": ["model.joblib", "scaler.pkl", "train_config.json"],
                            "deployment_approved": False}
        (product_dir / "manifest.json").write_text(json.dumps(product_manifest, ensure_ascii=False, indent=2))
    print(f"candidate output: {output}", flush=True)


if __name__ == "__main__":
    main()
