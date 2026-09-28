"""Evaluate a fixed market-relative Ridge specification on broader shares.

Architecture, target and threshold were fixed before this run from the
200-stock research. The price universe is a current KRX snapshot, so the
result remains retrospective research and cannot be deployment-approved.
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

from train_stock_single_swing_market_relative import make_features, make_target, evaluate, compact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--price-data", required=True, type=Path)
    parser.add_argument("--index-data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--architecture", choices=("ridge", "hgb_small"), default="ridge")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    env = pd.read_csv(args.price_data / "ohlcv.csv.gz",
        dtype={"code": str, "date": str, "open": np.float32, "high": np.float32,
               "low": np.float32, "close": np.float32, "volume": np.float32,
               "market_name": str, "is_active": str},
        usecols=["code", "date", "open", "high", "low", "close", "volume", "market_name", "is_active"])
    env = env.loc[env.date.ge("20200101")].rename(columns={"code": "stock_code"})
    env.stock_code = env.stock_code.str.zfill(6)
    env = env.sort_values(["stock_code", "date"]).reset_index(drop=True)
    print("loaded", len(env), "rows", env.stock_code.nunique(), "codes", flush=True)
    index = pd.read_csv(args.index_data, dtype={"date": str})
    features, names = make_features(env, index)
    target, end_dates = make_target(env, index, 20)
    dates = env.date.to_numpy()
    sample = np.random.default_rng(42).random(len(env)) < (1 / 3)

    def train(train_end: str):
        mask = (dates <= train_end) & (end_dates <= train_end) & np.isfinite(target) & sample
        if args.architecture == "ridge":
            model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1000.0))
        else:
            model = HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=15,
                min_samples_leaf=200, l2_regularization=10.0, learning_rate=0.05,
                random_state=42)
        model.fit(features[mask], target[mask])
        return model, int(mask.sum())

    report = {"product": "stock-single-swing", "model_type": f"market_relative_{args.architecture}_research",
              "source": str(args.price_data), "feature_names": names,
              "target": "20-day next-open stock return minus matched-market index return",
              "policy": "stock if predicted alpha > 0, otherwise matching index proxy",
              "train_sample_fraction": 1/3, "seed": 42,
              "hyperparameters": {"ridge_alpha": 1000.0} if args.architecture == "ridge" else
                {"max_iter": 100, "max_leaf_nodes": 15, "min_samples_leaf": 200,
                 "l2_regularization": 10.0, "learning_rate": 0.05},
              "universe_warning": "Current KRX ordinary-share classification; delisted histories and point-in-time membership incomplete",
              "benchmark_warning": "Index levels are a reference, not tradable ETF fills",
              "deployment_approved": False, "folds": {}, "retrospective": {}}
    for year in (2022, 2023, 2024):
        model, count = train(f"{year-1}1231")
        pred = model.predict(features).astype(np.float32)
        result = evaluate(env, pred, index, f"{year}0101", f"{year}1231")
        report["folds"][str(year)] = {**compact(result), "train_rows": count}
        print("fold", year, json.dumps(report["folds"][str(year)], ensure_ascii=False), flush=True)
        (args.output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    model, count = train("20241231")
    joblib.dump(model, args.output / "model.joblib", compress=3)
    pred = model.predict(features).astype(np.float32)
    report["final_train_rows"] = count
    for year, start, end in (("2025", "20250101", "20251231"), ("2026", "20260101", "20260916")):
        result = evaluate(env, pred, index, start, end)
        report["retrospective"][year] = compact(result)
        (args.output / f"{year}_by_stock.json").write_text(json.dumps(result["by_stock"], ensure_ascii=False, indent=2))
        print("retrospective", year, json.dumps(compact(result), ensure_ascii=False), flush=True)
    (args.output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
