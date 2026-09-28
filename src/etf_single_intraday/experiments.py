"""Run a small, predeclared 15-minute intraday model experiment matrix.

The forward dates are never used for model or threshold selection.  This is
intentionally separate from ``research.py`` so the original reports remain
reproducible and comparable.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from ..intraday_features import AI_FEATURE_COLUMNS, CORE_FEATURE_COLUMNS, FEATURE_COLUMNS, load_minute_candles
from .research import evaluate, samples


AI_COLUMNS = AI_FEATURE_COLUMNS
TECHNICAL_COLUMNS = CORE_FEATURE_COLUMNS


def _fit_predict(name, train, part, target):
    common = dict(max_iter=120, min_samples_leaf=100, l2_regularization=10.0,
                  learning_rate=.04, random_state=42, early_stopping=False)
    if name == "ai_abs":
        model = HistGradientBoostingRegressor(loss="absolute_error", max_leaf_nodes=7, **common)
    elif name == "ai_quantile25":
        model = HistGradientBoostingRegressor(loss="quantile", quantile=.25,
                                              max_leaf_nodes=7, **common)
    elif name == "ai_classifier":
        model = HistGradientBoostingClassifier(max_leaf_nodes=7, **common)
    else:
        model = HistGradientBoostingRegressor(max_leaf_nodes=15, **common)
    with threadpool_limits(limits=2):
        if name == "ai_classifier":
            model.fit(train[FEATURE_COLUMNS], (train.gross > .0018).astype(int))
            prediction = model.predict_proba(part[FEATURE_COLUMNS])[:, 1]
        else:
            model.fit(train[target], train.gross.clip(-.05, .05))
            prediction = model.predict(part[target])
    return model, prediction


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
    candles = candles[candles.code.isin(codes)].copy()
    data = samples(candles, 15).sort_values(["date", "code", "time"]).reset_index(drop=True)
    train = data[data.date <= "20260819"].copy()
    val = data[data.date.between("20260820", "20260827")].reset_index(drop=True)
    forward = data[data.date >= "20260908"].reset_index(drop=True)
    val_dates = sorted(val.date.unique())
    forward_dates = sorted(forward.date.unique())

    # Each row is an independently specified hypothesis, not a post-hoc grid.
    variants = {
        "ai_squared": (FEATURE_COLUMNS, FEATURE_COLUMNS),
        "ai_abs": (FEATURE_COLUMNS, FEATURE_COLUMNS),
        "ai_quantile25": (FEATURE_COLUMNS, FEATURE_COLUMNS),
        "ai_classifier": (FEATURE_COLUMNS, FEATURE_COLUMNS),
        "technical_only": (TECHNICAL_COLUMNS, TECHNICAL_COLUMNS),
        "ai_fresh5": (FEATURE_COLUMNS, FEATURE_COLUMNS),
    }
    thresholds = {
        "ai_classifier": (.55, .60, .65, .70),
        "ai_quantile25": (.0009, .0015, .0025, .004),
    }
    all_results = []
    fitted = {}
    for name, (train_columns, part_columns) in variants.items():
        train_view, val_view, forward_view = train.copy(), val.copy(), forward.copy()
        if name == "ai_fresh5":
            # Stale AI opinions are treated as unavailable; the explicit
            # ai_available/ai_age_days columns remain visible to the model.
            for view in (train_view, val_view, forward_view):
                stale = view.ai_age_days > (5.0 / 30.0)
                view.loc[stale, AI_COLUMNS[:4] + AI_COLUMNS[5:]] = 0.0
        model, val_prediction = _fit_predict(name, train_view, val_view, train_columns)
        forward_prediction = (model.predict_proba(forward_view[part_columns])[:, 1]
                              if name == "ai_classifier" else model.predict(forward_view[part_columns]))
        fitted[name] = (model, train_columns)
        for threshold in thresholds.get(name, (.0009, .0015, .0025, .004)):
            val_result = evaluate(val_view, val_prediction, threshold, codes, val_dates, cost=.0018)
            forward_result = evaluate(forward_view, forward_prediction, threshold,
                                      codes, forward_dates, cost=.0018)
            all_results.append({"variant": name, "threshold": threshold,
                                "validation_double_cost_pct": val_result["mean_return_pct"],
                                "validation_trades": val_result["trades"],
                                "forward_double_cost_pct": forward_result["mean_return_pct"],
                                "forward_trades": forward_result["trades"]})

    ranking = sorted(all_results, key=lambda row: row["validation_double_cost_pct"], reverse=True)
    best = ranking[0]
    model, train_columns = fitted[best["variant"]]
    # Recreate the selected variant's forward features exactly.
    forward_selected = forward.copy()
    if best["variant"] == "ai_fresh5":
        stale = forward_selected.ai_age_days > (5.0 / 30.0)
        forward_selected.loc[stale, AI_COLUMNS[:4] + AI_COLUMNS[5:]] = 0.0
    with threadpool_limits(limits=2):
        prediction = (model.predict_proba(forward_selected[train_columns])[:, 1]
                      if best["variant"] == "ai_classifier" else model.predict(forward_selected[train_columns]))
    result = {
        "protocol": "15-minute fixed horizon; train-only selection; validation double-cost selection; forward dates untouched until final evaluation",
        "codes": codes, "train_rows": len(train), "validation_rows": len(val),
        "forward_rows": len(forward), "ranking": ranking, "best": best,
        "forward_base_cost": evaluate(forward_selected, prediction, best["threshold"], codes, forward_dates),
        "forward_double_cost": evaluate(forward_selected, prediction, best["threshold"], codes, forward_dates, cost=.0018),
        "deployment_approved": False,
    }
    (output / "experiment_report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    joblib.dump({"model": model, "features": train_columns, "threshold": best["threshold"],
                 "variant": best["variant"], "deployment_approved": False}, output / "best_candidate.joblib")
    print(json.dumps({"best": best, "forward_base_cost": result["forward_base_cost"],
                      "forward_double_cost": result["forward_double_cost"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
