"""Robustness experiments for the 15-minute ETF model.

Hypotheses tested here address calibration drift and over-trading rather than
adding another large feature grid: cross-sectional top-k selection, a market
regime filter, robust labels, recency weighting, and volatility-normalized
labels. Forward dates are reported only after validation selection.
"""
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


def ranked_prediction(frame, prediction, top_k, threshold=0.0, regime=False):
    """Rank only simultaneous signals; later bars cannot affect earlier orders."""
    selected = np.full(len(frame), -np.inf, dtype=float)
    for _, indices in frame.groupby(["date", "time"], sort=False).groups.items():
        indices = np.asarray(indices, dtype=int)
        if regime:
            indices = indices[frame.loc[indices, "xsec_mean_ret_1"].to_numpy() > 0]
        indices = indices[prediction[indices] > threshold]
        if len(indices):
            order = indices[np.argsort(prediction[indices])[::-1][:top_k]]
            selected[order] = prediction[order]
    return selected


def fit_variant(name, train, val, forward):
    model = HistGradientBoostingRegressor(
        max_iter=120, max_leaf_nodes=15, min_samples_leaf=100,
        l2_regularization=10.0, learning_rate=.04,
        early_stopping=False, random_state=42,
    )
    target = train.gross.clip(-.05, .05)
    weights = None
    if name == "robust_label":
        target = train.gross.clip(-.02, .02)
    elif name == "volatility_normalized":
        target = (train.gross / train.atr_pct_14.clip(lower=.001)).clip(-5, 5)
    elif name == "recent_weighted":
        dates = sorted(train.date.unique())
        age = train.date.map({d: len(dates) - i for i, d in enumerate(dates)}).to_numpy()
        weights = np.exp(-np.log(2) * age / 20.0)
    with threadpool_limits(limits=2):
        model.fit(train[FEATURE_COLUMNS], target, sample_weight=weights)
        val_prediction = model.predict(val[FEATURE_COLUMNS])
        forward_prediction = model.predict(forward[FEATURE_COLUMNS])
    return model, val_prediction, forward_prediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    candles = load_minute_candles("20260709", "20260910")
    train_candles = candles[candles.date <= "20260819"]
    counts = train_candles.groupby(["code", "date"]).size().groupby("code").median()
    codes = sorted(counts[counts >= 330].index)
    data = samples(candles[candles.code.isin(codes)].copy(), 15)
    data = data.sort_values(["date", "code", "time"]).reset_index(drop=True)
    train = data[data.date <= "20260819"].copy()
    val = data[data.date.between("20260820", "20260827")].reset_index(drop=True)
    forward = data[data.date >= "20260908"].reset_index(drop=True)
    val_dates, forward_dates = sorted(val.date.unique()), sorted(forward.date.unique())

    variants = ("raw_topk", "regime_topk", "robust_label", "recent_weighted",
                "volatility_normalized")
    results = []
    fitted = {}
    for name in variants:
        model, vp, fp = fit_variant(name, train, val, forward)
        fitted[name] = (model, vp, fp)
        for top_k in (3, 5, 10):
            regime = name == "regime_topk"
            # Raw-label models require a minimum gross forecast; normalized
            # labels use only positive risk-adjusted forecasts.
            minimum = .0009 if name in ("raw_topk", "regime_topk", "robust_label", "recent_weighted") else 0.0
            v_selected = ranked_prediction(val, vp, top_k, minimum, regime)
            f_selected = ranked_prediction(forward, fp, top_k, minimum, regime)
            v = evaluate(val, v_selected, -np.inf, codes, val_dates, cost=.0018)
            f = evaluate(forward, f_selected, -np.inf, codes, forward_dates, cost=.0018)
            results.append({"variant": name, "top_k": top_k,
                            "validation_double_cost_pct": v["mean_return_pct"],
                            "validation_trades": v["trades"],
                            "forward_double_cost_pct": f["mean_return_pct"],
                            "forward_trades": f["trades"]})

    viable = [r for r in results if r["validation_trades"] >= 10]
    ranking = sorted(viable or results,
                     key=lambda r: r["validation_double_cost_pct"], reverse=True)
    best = ranking[0]
    model, vp, fp = fitted[best["variant"]]
    result = {
        "protocol": "15-minute top-k robustness matrix; validation-only selection; forward untouched",
        "codes": codes, "train_rows": len(train), "validation_rows": len(val),
        "forward_rows": len(forward), "ranking": ranking, "best": best,
        "deployment_approved": False,
    }
    (output / "robust_report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    joblib.dump({"model": model, "features": FEATURE_COLUMNS, "variant": best["variant"],
                 "top_k": best["top_k"], "deployment_approved": False}, output / "best_candidate.joblib")
    print(json.dumps({"best": best, "ranking": ranking[:8]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
