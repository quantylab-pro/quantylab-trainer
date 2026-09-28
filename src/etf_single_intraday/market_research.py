"""Long-history research with previous-session AI market-regime features."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from ..intraday_features import AI_FEATURE_COLUMNS, CORE_FEATURE_COLUMNS, FEATURE_COLUMNS, MARKET_FEATURE_COLUMNS, load_minute_candles
from .research import evaluate, samples


MARKET_COLUMNS = MARKET_FEATURE_COLUMNS
ETF_AI_COLUMNS = AI_FEATURE_COLUMNS
TECHNICAL_COLUMNS = CORE_FEATURE_COLUMNS


def fit_model(name, train, frame):
    columns = {
        "technical": TECHNICAL_COLUMNS,
        "etf_ai": TECHNICAL_COLUMNS + ETF_AI_COLUMNS,
        "etf_ai_market": FEATURE_COLUMNS,
        "market_regime": TECHNICAL_COLUMNS + MARKET_COLUMNS,
    }[name]
    model = HistGradientBoostingRegressor(
        max_iter=180, max_leaf_nodes=15, min_samples_leaf=120,
        l2_regularization=15.0, learning_rate=.035,
        early_stopping=False, random_state=42,
    )
    target = train.gross.clip(-.03, .03)
    with threadpool_limits(limits=2):
        model.fit(train[columns], target)
        prediction = model.predict(frame[columns])
    return model, columns, prediction


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--start-date", default="20260129")
    p.add_argument("--end-date", default="20260910")
    p.add_argument("--train-end", default="20260731")
    p.add_argument("--validation-start", default="20260803")
    p.add_argument("--validation-end", default="20260814")
    p.add_argument("--forward-start", default="20260818")
    args = p.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)

    candles = load_minute_candles(args.start_date, args.end_date)
    train_candles = candles[candles.date <= args.train_end]
    counts = train_candles.groupby(["code", "date"]).size().groupby("code").median()
    codes = sorted(counts[counts >= 330].index)
    candles = candles[candles.code.isin(codes)].copy()
    data = samples(candles, 15).sort_values(["date", "code", "time"]).reset_index(drop=True)
    train = data[data.date <= args.train_end].copy()
    val = data[data.date.between(args.validation_start, args.validation_end)].reset_index(drop=True)
    forward = data[data.date >= args.forward_start].reset_index(drop=True)
    val_dates = sorted(val.date.unique())
    forward_dates = sorted(forward.date.unique())
    results = []
    fitted = {}
    # Thresholds are deliberately narrow and declared before seeing forward data.
    for name in ("technical", "etf_ai", "etf_ai_market", "market_regime"):
        model, columns, vp = fit_model(name, train, val)
        # The same model must score validation and forward data.
        with threadpool_limits(limits=2):
            fp = model.predict(forward[columns])
        fitted[name] = (model, columns)
        for threshold in (.0009, .0015, .0025, .004):
            v = evaluate(val, vp, threshold, codes, val_dates, cost=.0018)
            f = evaluate(forward, fp, threshold, codes, forward_dates, cost=.0018)
            results.append({"variant": name, "threshold": threshold,
                            "validation_double_cost_pct": v["mean_return_pct"],
                            "validation_trades": v["trades"],
                            "forward_double_cost_pct": f["mean_return_pct"],
                            "forward_trades": f["trades"]})
    viable = [r for r in results if r["validation_trades"] >= 10]
    ranking = sorted(viable or results,
                     key=lambda r: r["validation_double_cost_pct"], reverse=True)
    best = ranking[0]
    model, columns = fitted[best["variant"]]
    report = {
        "protocol": "Long-history 15-minute research; train-only liquidity universe; previous-session AI ETF and market analyses; validation-only selection",
        "periods": {"train_end": args.train_end, "validation": [args.validation_start, args.validation_end],
                    "forward_start": args.forward_start, "end_date": args.end_date},
        "codes": codes, "train_rows": len(train), "validation_rows": len(val),
        "forward_rows": len(forward), "best": best, "ranking": ranking,
        "deployment_approved": False,
        "warning": "Forward dates through 20260910 have appeared in earlier research and are retrospective audit, not a fresh holdout.",
    }
    (output / "market_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    joblib.dump({"model": model, "features": columns, "threshold": best["threshold"],
                 "variant": best["variant"], "deployment_approved": False}, output / "best_candidate.joblib")
    print(json.dumps({"best": best, "ranking": ranking[:8]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
