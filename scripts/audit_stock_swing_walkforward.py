"""Fixed-rule historical walk-forward audit for stock swing alpha research."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_stock_swing_alpha_candidates import make_target, portfolio_eval, single_eval, train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    run = args.run
    manifest = json.loads((run / "manifest.json").read_text())
    dataset = Path(manifest["dataset"])
    env = pd.read_csv(dataset / "environment.csv", dtype={"date": str, "stock_code": str}, low_memory=False)
    env["stock_code"] = env["stock_code"].str.zfill(6)
    features = pd.read_csv(dataset / "training_scaled.csv", dtype=np.float32).to_numpy(np.float32)
    codes = manifest["stock-portfolio-swing"]["universe"]
    report = {"note": "Retrospective folds; the dataset scaler was fitted through 2024 and the universe contains only surviving stocks.",
              "rules": {"single_threshold": 0.0, "portfolio_top_k": 5}, "folds": {}}
    for product, horizon in (("stock-single-swing", 5), ("stock-portfolio-swing", 20)):
        target, future_date = make_target(env, horizon)
        report["folds"][product] = {}
        for year in (2022, 2023, 2024):
            train_end = f"{year - 1}1231"
            print(f"{product} train_end={train_end} evaluate={year}", flush=True)
            model = train(features, target, env, future_date, seed=42, train_end=train_end)
            pred = model.predict(features).astype(np.float32)
            if product == "stock-single-swing":
                raw = single_eval(env, pred, f"{year}0101", f"{year}1231", 0.0)
                result = {k: v for k, v in raw.items() if k != "by_stock"}
            else:
                result = portfolio_eval(env, pred, codes, f"{year}0101", f"{year}1231", 5)
            report["folds"][product][str(year)] = result
            (run / "walkforward_2022_2024.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
            print(json.dumps(result, ensure_ascii=False), flush=True)
    manifest["walkforward_audit"] = "walkforward_2022_2024.json"
    (run / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
