"""Test the 200-stock research model on unseen ordinary-share codes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from train_stock_single_swing_market_relative import make_features, evaluate, compact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-run", required=True, type=Path)
    parser.add_argument("--price-data", required=True, type=Path)
    args = parser.parse_args()
    model_manifest = json.loads((args.model_run / "manifest.json").read_text())
    original = pd.read_csv(Path(model_manifest["dataset"]) / "environment.csv",
                           usecols=["stock_code"], dtype=str)
    trained_codes = set(original.stock_code.str.zfill(6).unique())
    env = pd.read_csv(args.price_data / "ohlcv.csv.gz",
        dtype={"code": str, "date": str, "open": np.float32, "high": np.float32,
               "low": np.float32, "close": np.float32, "volume": np.float32,
               "market_name": str, "is_active": str},
        usecols=["code", "date", "open", "high", "low", "close", "volume", "market_name", "is_active"])
    env = env.loc[env.date.ge("20240101")].rename(columns={"code": "stock_code"})
    env.stock_code = env.stock_code.str.zfill(6)
    env = env.sort_values(["stock_code", "date"]).reset_index(drop=True)
    index = pd.read_csv(Path(model_manifest["dataset"]) / "market_index_daily_2015_2026.csv", dtype={"date": str})
    features, names = make_features(env, index)
    if names != model_manifest["feature_names"]:
        raise ValueError("Feature contract differs from trained candidate")
    model = joblib.load(args.model_run / "model.joblib")
    predictions = model.predict(features).astype(np.float32)
    external = ~env.stock_code.isin(trained_codes)
    external_env = env.loc[external].reset_index(drop=True)
    external_predictions = predictions[external.to_numpy()]
    history = external_env.loc[external_env.date.between("20240101", "20241231")].copy()
    history["dollar_volume"] = history.close.astype(float) * history.volume.astype(float)
    history_stats = history.groupby("stock_code").agg(
        trading_days=("date", "nunique"), median_dollar_volume=("dollar_volume", "median"))
    liquid_codes = set(history_stats.loc[
        history_stats.trading_days.ge(180) & history_stats.median_dollar_volume.ge(1_000_000_000)
    ].index)
    output = {"model_run": str(args.model_run), "external_source": str(args.price_data),
              "train_universe_codes": len(trained_codes), "external_codes": int(external_env.stock_code.nunique()),
              "liquid_external_codes": len(liquid_codes),
              "liquid_rule": "At least 180 trading days and median close*volume >= KRW 1bn in 2024",
              "warning": "Current KRX ordinary-share classification, not a historical point-in-time universe; delisted outcomes incomplete.",
              "periods": {}}
    for year, start, end in (("2025", "20250101", "20251231"), ("2026", "20260101", "20260916")):
        result = evaluate(external_env, external_predictions, index, start, end)
        liquid_mask = external_env.stock_code.isin(liquid_codes).to_numpy()
        liquid_result = evaluate(external_env.loc[liquid_mask].reset_index(drop=True),
                                 external_predictions[liquid_mask], index, start, end)
        output["periods"][year] = {"all_external": compact(result), "liquid_external": compact(liquid_result)}
        (args.model_run / f"external_{year}_by_stock.json").write_text(
            json.dumps(result["by_stock"], ensure_ascii=False, indent=2))
        print(year, json.dumps(output["periods"][year], ensure_ascii=False), flush=True)
    (args.model_run / "external_universe_audit.json").write_text(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
