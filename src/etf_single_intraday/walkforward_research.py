"""Expanding-window walk-forward selection for the intraday feature set."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from ..intraday_features import FEATURE_COLUMNS, load_minute_candles
from .research import evaluate, samples
from .market_research import ETF_AI_COLUMNS, MARKET_COLUMNS, TECHNICAL_COLUMNS


VARIANTS = {
    "technical": TECHNICAL_COLUMNS,
    "etf_ai": TECHNICAL_COLUMNS + ETF_AI_COLUMNS,
    "market_regime": TECHNICAL_COLUMNS + MARKET_COLUMNS,
    "etf_ai_market": FEATURE_COLUMNS,
}
FOLDS = (("20260529", "20260601", "20260612"),
         ("20260630", "20260701", "20260715"),
         ("20260731", "20260803", "20260814"))


def fit(train, frame, columns):
    model = HistGradientBoostingRegressor(
        max_iter=180, max_leaf_nodes=15, min_samples_leaf=120,
        l2_regularization=15.0, learning_rate=.035,
        early_stopping=False, random_state=42,
    )
    with threadpool_limits(limits=2):
        model.fit(train[columns], train.gross.clip(-.03, .03))
        return model, model.predict(frame[columns])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    args = p.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    candles = load_minute_candles("20260129", "20260910")
    base = candles[candles.date <= "20260731"]
    counts = base.groupby(["code", "date"]).size().groupby("code").median()
    codes = sorted(counts[counts >= 330].index)
    data = samples(candles[candles.code.isin(codes)].copy(), 15)
    thresholds = (.0009, .0015, .0025, .004)
    results = []
    for variant, columns in VARIANTS.items():
        fold_metrics = []
        for train_end, val_start, val_end in FOLDS:
            train = data[data.date <= train_end].copy()
            val = data[data.date.between(val_start, val_end)].reset_index(drop=True)
            model, prediction = fit(train, val, columns)
            dates = sorted(val.date.unique())
            for threshold in thresholds:
                metric = evaluate(val, prediction, threshold, codes, dates, cost=.0018)
                fold_metrics.append({"threshold": threshold, "train_end": train_end, "validation": [val_start, val_end],
                                     "return_pct": metric["mean_return_pct"], "trades": metric["trades"]})
        for threshold in thresholds:
            selected = [m for m in fold_metrics if m["threshold"] == threshold]
            returns = np.array([m["return_pct"] for m in selected])
            trades = sum(m["trades"] for m in selected)
            results.append({"variant": variant, "threshold": threshold,
                            "fold_returns_pct": returns.tolist(),
                            "fold_trades": [m["trades"] for m in selected],
                            "mean_return_pct": float(returns.mean()),
                            "worst_fold_pct": float(returns.min()),
                            "stability_score": float(returns.mean() - .5 * returns.std()),
                            "total_trades": trades})
    viable = [r for r in results if r["total_trades"] >= 15]
    ranking = sorted(viable or results, key=lambda r: r["stability_score"], reverse=True)
    best = ranking[0]
    columns = VARIANTS[best["variant"]]
    train = data[data.date <= "20260731"].copy()
    forward = data[data.date >= "20260818"].reset_index(drop=True)
    model, prediction = fit(train, forward, columns)
    forward_dates = sorted(forward.date.unique())
    report = {
        "protocol": "Three expanding-window validation folds; stability-score selection; final recent-date audit",
        "folds": FOLDS, "codes": codes, "train_rows": len(train),
        "forward_rows": len(forward), "best": best, "ranking": ranking,
        "forward_base": evaluate(forward, prediction, best["threshold"], codes, forward_dates),
        "forward_double_cost": evaluate(forward, prediction, best["threshold"], codes, forward_dates, cost=.0018),
        "deployment_approved": False,
        "warning": "Forward through 20260910 is retrospective audit because it was used in earlier experiments.",
    }
    (output / "walkforward_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    joblib.dump({"model": model, "features": columns, "threshold": best["threshold"],
                 "variant": best["variant"], "deployment_approved": False}, output / "best_candidate.joblib")
    print(json.dumps({"best": best, "forward_base": report["forward_base"],
                      "forward_double_cost": report["forward_double_cost"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
