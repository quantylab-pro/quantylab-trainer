"""Causal stock-vs-market swing research with next-open execution.

The output is an individual stock's expected excess return versus its own
market. Each test account owns that stock or the corresponding index proxy.
Index levels are a reference and are not an investable ETF simulation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


HORIZONS = (5, 20)
ARCHITECTURES = ("ridge", "hgb_small", "hgb_medium")
ENTRY_COST = 0.0009  # sell index proxy and buy stock
EXIT_COST = 0.0029  # sell stock with tax and buy index proxy


def make_features(env: pd.DataFrame, index: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    group = env.groupby("stock_code", sort=False)
    close = env["close"].astype(float)
    opened = env["open"].astype(float)
    volume = env["volume"].astype(float)
    daily_log = np.log(close / group["close"].shift(1).astype(float))
    frame = pd.DataFrame(index=env.index)
    for days in (1, 5, 20, 60, 120):
        frame[f"stock_momentum_{days}d"] = np.log(close / group["close"].shift(days).astype(float))
    for days in (20, 60):
        frame[f"stock_volatility_{days}d"] = daily_log.groupby(env["stock_code"], sort=False).transform(
            lambda s: s.rolling(days, min_periods=days).std())
    frame["stock_day_range"] = (env["high"].astype(float) - env["low"].astype(float)) / close
    frame["stock_day_body"] = close / opened - 1
    frame["stock_gap"] = opened / group["close"].shift(1).astype(float) - 1
    for days in (5, 20):
        mean_volume = volume.groupby(env["stock_code"], sort=False).transform(
            lambda s: s.rolling(days, min_periods=days).mean())
        frame[f"volume_vs_{days}d"] = np.log1p(volume) - np.log1p(mean_volume)
    frame["log_dollar_volume"] = np.log1p((close * volume).clip(lower=0))

    index_close = index.pivot(index="date", columns="market", values="close").sort_index().reindex(columns=["kospi", "kosdaq"])
    markets = env["market_name"].to_numpy()
    dates = env["date"].to_numpy()
    for days in (5, 20, 60):
        mom = np.log(index_close / index_close.shift(days))
        frame[f"market_momentum_{days}d"] = np.where(
            markets == "kospi", pd.Series(dates).map(mom["kospi"]).to_numpy(),
            pd.Series(dates).map(mom["kosdaq"]).to_numpy())
    market_daily = np.log(index_close / index_close.shift(1))
    market_vol = market_daily.rolling(20, min_periods=20).std()
    frame["market_volatility_20d"] = np.where(
        markets == "kospi", pd.Series(dates).map(market_vol["kospi"]).to_numpy(),
        pd.Series(dates).map(market_vol["kosdaq"]).to_numpy())
    frame["stock_minus_market_momentum_20d"] = frame["stock_momentum_20d"] - frame["market_momentum_20d"]
    frame["market_kosdaq"] = (markets == "kosdaq").astype(float)
    frame = frame.replace([np.inf, -np.inf], np.nan).clip(-20, 20)
    return frame.to_numpy(np.float32), list(frame.columns)


def make_target(env: pd.DataFrame, index: pd.DataFrame, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Stock minus market open-to-open return, starting at the next open."""
    group = env.groupby("stock_code", sort=False)
    entry = group["open"].shift(-1).astype(float)
    exit_price = group["open"].shift(-(horizon + 1)).astype(float)
    exit_date = group["date"].shift(-(horizon + 1)).fillna("").to_numpy()
    index_open = index.pivot(index="date", columns="market", values="open").sort_index().reindex(columns=["kospi", "kosdaq"])
    daily = env["date"].to_numpy()
    markets = env["market_name"].to_numpy()
    market_entry = np.full(len(env), np.nan)
    market_exit = np.full(len(env), np.nan)
    for market in ("kospi", "kosdaq"):
        series = index_open[market]
        mask = markets == market
        entry_dates = group["date"].shift(-1).fillna("").to_numpy()[mask]
        market_entry[mask] = pd.Series(entry_dates).map(series).to_numpy(float)
        market_exit[mask] = pd.Series(exit_date[mask]).map(series).to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        target = np.log(exit_price.to_numpy(float) / entry.to_numpy(float)) - np.log(market_exit / market_entry)
    target = np.clip(target, -0.8, 0.8).astype(np.float32)
    return target, exit_date


def fit_model(architecture: str, features: np.ndarray, target: np.ndarray,
              dates: np.ndarray, end_dates: np.ndarray, train_end: str):
    mask = (dates <= train_end) & (end_dates <= train_end) & np.isfinite(target)
    if architecture == "ridge":
        model = make_pipeline(SimpleImputer(strategy="median", add_indicator=False),
                              StandardScaler(), Ridge(alpha=1000.0))
    else:
        params = {"hgb_small": (100, 15, 200, 10.0),
                  "hgb_medium": (180, 31, 120, 3.0)}[architecture]
        model = HistGradientBoostingRegressor(max_iter=params[0], max_leaf_nodes=params[1],
            min_samples_leaf=params[2], l2_regularization=params[3], learning_rate=0.05,
            random_state=42)
    model.fit(features[mask], target[mask])
    return model, int(mask.sum())


def evaluate(env: pd.DataFrame, predictions: np.ndarray, index: pd.DataFrame,
             start: str, end: str, threshold: float = 0.0,
             cost_multiplier: float = 1.0) -> dict:
    """Separate accounts: each holds its stock when forecast alpha > threshold, else its market."""
    mask = env["date"].between(start, end).to_numpy()
    frame = env.loc[mask, ["stock_code", "market_name", "date", "open", "volume"]].copy()
    frame["prediction"] = predictions[mask]
    index_open = index.pivot(index="date", columns="market", values="open").sort_index()
    rows = []
    for code, stock in frame.groupby("stock_code", sort=False):
        stock = stock.sort_values("date")
        if len(stock) < 35:
            continue
        market = stock["market_name"].iloc[0]
        price = stock["open"].to_numpy(float)
        market_price = stock["date"].map(index_open[market]).to_numpy(float)
        if np.any(~np.isfinite(price)) or np.any(price <= 0) or np.any(~np.isfinite(market_price)):
            continue
        position = (stock["prediction"].to_numpy(float)[:-2] > threshold).astype(float)
        stock_return = price[2:] / price[1:-1] - 1
        market_return = market_price[2:] / market_price[1:-1] - 1
        previous = np.r_[0, position[:-1]]
        cost = cost_multiplier * (
            np.maximum(position - previous, 0) * ENTRY_COST
            + np.maximum(previous - position, 0) * EXIT_COST
        )
        strategy = position * stock_return + (1 - position) * market_return - cost
        if position[-1] > 0:
            strategy[-1] -= cost_multiplier * EXIT_COST
        wealth = np.cumprod(1 + strategy)
        baseline = np.cumprod(1 + market_return)
        excess = float((wealth[-1] - baseline[-1]) * 100)
        peak = np.maximum.accumulate(np.r_[1, wealth])[1:]
        rows.append({"stock_code": code, "market_name": market,
                     "return": float(wealth[-1] - 1), "market_return": float(baseline[-1] - 1),
                     "excess_percentage_points": excess, "mean_stock_weight": float(position.mean()),
                     "switches": int(np.abs(position - previous).sum()),
                     "max_drawdown": float(np.max(1 - wealth / peak))})
    if not rows:
        raise ValueError("No valid stock accounts")
    result = {"stocks": len(rows), "threshold": threshold,
              "cost_multiplier": cost_multiplier,
              "mean_return": float(np.mean([r["return"] for r in rows])),
              "mean_market_return": float(np.mean([r["market_return"] for r in rows])),
              "mean_excess_percentage_points": float(np.mean([r["excess_percentage_points"] for r in rows])),
              "median_excess_percentage_points": float(np.median([r["excess_percentage_points"] for r in rows])),
              "positive_excess_fraction": float(np.mean([r["excess_percentage_points"] > 0 for r in rows])),
              "mean_stock_weight": float(np.mean([r["mean_stock_weight"] for r in rows])),
              "mean_switches": float(np.mean([r["switches"] for r in rows])),
              "mean_max_drawdown": float(np.mean([r["max_drawdown"] for r in rows])),
              "by_stock": rows}
    return result


def compact(result: dict) -> dict:
    return {key: value for key, value in result.items() if key != "by_stock"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("/home/quantylab/quantylab-trainer/data/stock_20260917"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    env = pd.read_csv(args.dataset / "environment.csv", dtype={"date": str, "stock_code": str}, low_memory=False)
    env["stock_code"] = env["stock_code"].str.zfill(6)
    env = env.sort_values(["stock_code", "date"]).reset_index(drop=True)
    index = pd.read_csv(args.dataset / "market_index_daily_2015_2026.csv", dtype={"date": str})
    features, feature_names = make_features(env, index)
    dates = env["date"].to_numpy()
    report = {"product": "stock-single-swing", "run_type": "research", "dataset": str(args.dataset),
              "feature_names": feature_names, "feature_origin": "OHLCV and same-date index closes only",
              "target": "next-open stock log return minus same-market index log return over 5/20 trading days",
              "decision": "stock if predicted alpha > 0, otherwise corresponding market index proxy",
              "execution": "signal at close t; fill at open t+1; mark at open t+2",
              "cost_assumptions": {"switch_index_to_stock": ENTRY_COST, "switch_stock_to_index": EXIT_COST},
              "data_warning": "200 stocks surviving to 2026; index proxy is not ETF; 2025/2026 are previously inspected",
              "trial_count": len(HORIZONS) * len(ARCHITECTURES), "deployment_approved": False,
              "architectures": {}}
    for horizon in HORIZONS:
        target, end_dates = make_target(env, index, horizon)
        for architecture in ARCHITECTURES:
            key = f"h{horizon}_{architecture}"
            folds = {}
            for year in (2022, 2023, 2024):
                model, samples = fit_model(architecture, features, target, dates, end_dates, f"{year-1}1231")
                pred = model.predict(features).astype(np.float32)
                result = evaluate(env, pred, index, f"{year}0101", f"{year}1231")
                folds[str(year)] = compact(result)
                folds[str(year)]["train_samples"] = samples
            report["architectures"][key] = {"horizon": horizon, "architecture": architecture, "folds": folds,
                "selection_score": float(np.mean([folds[str(y)]["mean_excess_percentage_points"] for y in (2022, 2023, 2024)]))}
            print(key, "fold excess pp", [round(folds[str(y)]["mean_excess_percentage_points"], 2) for y in (2022, 2023, 2024)], flush=True)
            (args.output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    selected = max(report["architectures"], key=lambda key: report["architectures"][key]["selection_score"])
    choice = report["architectures"][selected]
    target, end_dates = make_target(env, index, choice["horizon"])
    model, samples = fit_model(choice["architecture"], features, target, dates, end_dates, "20241231")
    pred = model.predict(features).astype(np.float32)
    joblib.dump(model, args.output / "model.joblib", compress=3)
    report["selected"] = selected
    report["selected_train_samples"] = samples
    report["retrospective"] = {}
    for year, start, end in (("2025", "20250101", "20251231"), ("2026", "20260101", "20260916")):
        result = evaluate(env, pred, index, start, end)
        report["retrospective"][year] = compact(result)
        (args.output / f"{year}_by_stock.json").write_text(json.dumps(result["by_stock"], ensure_ascii=False, indent=2))
        print(year, json.dumps(compact(result), ensure_ascii=False), flush=True)
    (args.output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print("output", args.output, flush=True)


if __name__ == "__main__":
    main()
