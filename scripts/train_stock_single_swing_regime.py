"""Research an absolute-return stock swing model with causal market context.

One shared model scores each stock independently. The output is a forecast,
not an order; evaluation uses next-day open fills and separate stock accounts.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

try:
    from scripts.train_stock_swing_alpha_candidates import metrics
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    from train_stock_swing_alpha_candidates import metrics


POLICIES = ("binary", "soft", "regime_floor", "core_half")


def market_features(env: pd.DataFrame, index: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    prices = index.pivot(index="date", columns="market", values="close").sort_index()
    dates = env["date"].to_numpy()
    markets = env["market_name"].to_numpy()
    columns = {}
    for days in (5, 20, 60):
        momentum = prices.pct_change(days, fill_method=None)
        columns[f"market_momentum_{days}d"] = np.where(
            markets == "kospi", pd.Series(dates).map(momentum["kospi"]).to_numpy(),
            pd.Series(dates).map(momentum["kosdaq"]).to_numpy())
    daily = prices.pct_change(fill_method=None)
    volatility = daily.rolling(20, min_periods=10).std()
    columns["market_volatility_20d"] = np.where(
        markets == "kospi", pd.Series(dates).map(volatility["kospi"]).to_numpy(),
        pd.Series(dates).map(volatility["kosdaq"]).to_numpy())
    frame = pd.DataFrame(columns).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    frame = frame.clip(-2, 2)
    return frame.to_numpy(np.float32), frame["market_momentum_60d"].to_numpy(np.float32), list(frame.columns)


def forward_target(env: pd.DataFrame, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    group = env.groupby("stock_code", sort=False)
    future_close = group["close"].shift(-horizon).to_numpy(float)
    future_date = group["date"].shift(-horizon).fillna("").to_numpy()
    target = np.log(future_close / env["close"].to_numpy(float))
    return np.clip(target, -0.8, 0.8).astype(np.float32), future_date


def fit(features: np.ndarray, target: np.ndarray, dates: np.ndarray,
        future_date: np.ndarray, train_end: str) -> HistGradientBoostingRegressor:
    mask = (dates <= train_end) & (future_date <= train_end) & np.isfinite(target)
    model = HistGradientBoostingRegressor(max_iter=180, learning_rate=0.05,
        max_leaf_nodes=24, min_samples_leaf=150, l2_regularization=3.0,
        random_state=42)
    model.fit(features[mask], target[mask])
    return model


def positions(prediction: np.ndarray, momentum_60d: np.ndarray, policy: str) -> np.ndarray:
    if policy == "binary":
        return (prediction > 0).astype(float)
    soft = 1 / (1 + np.exp(-np.clip(prediction / 0.04, -30, 30)))
    if policy == "soft":
        return soft
    if policy == "regime_floor":
        return np.maximum(soft, np.where(momentum_60d > 0, 0.8, 0.0))
    if policy == "core_half":
        return 0.5 + 0.5 * soft
    raise ValueError(policy)


def evaluate(env: pd.DataFrame, pred: np.ndarray, market_mom60: np.ndarray,
             start: str, end: str, policy: str, cost_multiplier: float = 1.0) -> dict:
    mask = env["date"].between(start, end).to_numpy()
    frame = env.loc[mask, ["stock_code", "date", "market_name", "open"]].copy()
    frame["pred"] = pred[mask]
    frame["market_mom60"] = market_mom60[mask]
    rows = []
    for code, stock in frame.groupby("stock_code", sort=False):
        stock = stock.sort_values("date")
        price = stock["open"].to_numpy(float)
        if len(price) < 30 or np.any(price <= 0):
            continue
        position = positions(stock["pred"].to_numpy(float)[:-2],
                             stock["market_mom60"].to_numpy(float)[:-2], policy)
        asset_return = price[2:] / price[1:-1] - 1
        previous = np.r_[0.0, position[:-1]]
        buys = np.maximum(position - previous, 0)
        sells = np.maximum(previous - position, 0)
        strategy = position * asset_return - cost_multiplier * (buys * 0.00045 + sells * 0.00245)
        strategy[-1] -= cost_multiplier * position[-1] * 0.00245
        hold = asset_return.copy()
        hold[0] -= 0.00045
        hold[-1] -= 0.00245
        result = metrics(strategy, hold, buys + sells)
        result.update({"stock_code": code, "market_name": stock["market_name"].iloc[0],
                       "active_fraction": float(position.mean()), "mean_position": float(position.mean()),
                       "buys": float(buys.sum()), "sells": float(sells.sum())})
        rows.append(result)
    if not rows:
        raise ValueError("No stock accounts to evaluate")
    return {"stocks": len(rows), "policy": policy, "cost_multiplier": cost_multiplier,
            "mean_return": float(np.mean([x["return"] for x in rows])),
            "mean_buy_hold_return": float(np.mean([x["benchmark_return"] for x in rows])),
            "mean_excess_percentage_points": float(np.mean([x["excess_percentage_points"] for x in rows])),
            "median_excess_percentage_points": float(np.median([x["excess_percentage_points"] for x in rows])),
            "positive_excess_fraction": float(np.mean([x["excess_percentage_points"] > 0 for x in rows])),
            "mean_position": float(np.mean([x["mean_position"] for x in rows])),
            "mean_turnover": float(np.mean([x["mean_turnover"] for x in rows])),
            "mean_sharpe": float(np.mean([x["sharpe"] for x in rows])),
            "mean_max_drawdown": float(np.mean([x["max_drawdown"] for x in rows])),
            "mean_buys": float(np.mean([x["buys"] for x in rows])),
            "by_stock": rows}


def concise(result: dict) -> dict:
    return {k: v for k, v in result.items() if k != "by_stock"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("/home/quantylab/quantylab-trainer/data/stock_20260917"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    env = pd.read_csv(args.dataset / "environment.csv", dtype={"date": str, "stock_code": str}, low_memory=False)
    env["stock_code"] = env["stock_code"].str.zfill(6)
    old_features = pd.read_csv(args.dataset / "training_scaled.csv", dtype=np.float32)
    index_path = args.dataset / "market_index_daily_2015_2026.csv"
    index = pd.read_csv(index_path, dtype={"date": str})
    market_values, momentum_60d, market_names = market_features(env, index)
    features = np.column_stack([old_features.to_numpy(np.float32), market_values])
    names = list(old_features.columns) + market_names
    report = {"product": "stock-single-swing", "candidate": "absolute_return_regime_research",
              "dataset": str(args.dataset), "index_source": str(index_path), "feature_names": names,
              "execution": "signal after close t, fill at open t+1, mark at open t+2",
              "costs": {"buy": 0.00045, "sell": 0.00245},
              "warning": "200 surviving stocks; 2025/2026 previously inspected; fold base scaler fitted through 2024",
              "deployment_approved": False, "horizons": {}}
    dates = env["date"].to_numpy()
    for horizon in (5, 20):
        print(f"horizon {horizon}", flush=True)
        target, future_date = forward_target(env, horizon)
        fold_results = {}
        for year in (2022, 2023, 2024):
            model = fit(features, target, dates, future_date, f"{year-1}1231")
            prediction = model.predict(features).astype(np.float32)
            fold_results[str(year)] = {policy: concise(evaluate(env, prediction, momentum_60d,
                f"{year}0101", f"{year}1231", policy)) for policy in POLICIES}
            print("fold", year, {p: round(fold_results[str(year)][p]["mean_excess_percentage_points"], 1) for p in POLICIES}, flush=True)
        # Policy selection uses only 2022-2024 folds; 2025/2026 are retrospective diagnostics.
        scores = {policy: float(np.mean([fold_results[str(year)][policy]["mean_excess_percentage_points"]
                for year in (2022, 2023, 2024)])) for policy in POLICIES}
        selected = max(scores, key=scores.get)
        model = fit(features, target, dates, future_date, "20241231")
        prediction = model.predict(features).astype(np.float32)
        model_path = args.output / f"h{horizon}_model.joblib"
        joblib.dump(model, model_path, compress=3)
        retrospective = {year: {policy: concise(evaluate(env, prediction, momentum_60d, start, end, policy))
                                  for policy in POLICIES}
                         for year, start, end in (("2025", "20250101", "20251231"),
                                                  ("2026", "20260101", "20260916"))}
        selected_2026 = {str(mult): concise(evaluate(env, prediction, momentum_60d,
            "20260101", "20260916", selected, mult)) for mult in (2.0, 4.0)}
        report["horizons"][str(horizon)] = {"selected_policy": selected, "selection_scores": scores,
            "folds": fold_results, "retrospective": retrospective, "cost_stress_2026": selected_2026,
            "model_file": model_path.name}
        (args.output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print("retrospective", horizon, selected,
              {year: round(retrospective[year][selected]["mean_excess_percentage_points"], 1)
               for year in ("2025", "2026")}, flush=True)
    shutil.copy2(args.dataset / "scaler.pkl", args.output / "base_scaler.pkl")
    shutil.copy2(index_path, args.output / "market_index_daily_2015_2026.csv")
    print("output", args.output, flush=True)


if __name__ == "__main__":
    main()
