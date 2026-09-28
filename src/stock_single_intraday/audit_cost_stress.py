"""Cost-only sensitivity audit for the strongest frequently trading EOD setting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .research_eod import (
    EOD_FEATURES,
    FOLDS,
    MAX_TRAIN_ROWS,
    _fit_eod,
    _predict_scores,
    _round_trip_costs,
    simulate_eod,
    add_eod_target,
)


def run(features_path: str, output_path: str) -> dict:
    with Path(features_path).open("rb") as stream:
        data = pd.read_pickle(stream)
    dates = data.date.astype(str)
    target = add_eod_target(data)
    label = (target > _round_trip_costs(data.date, "historical")).where(target.notna())
    fold_rows = []
    for fold, (train_end, val_start, val_end) in enumerate(FOLDS, 1):
        train_idx = np.flatnonzero((dates.le(train_end) & target.notna()).to_numpy())
        if len(train_idx) > MAX_TRAIN_ROWS:
            rng = np.random.default_rng(20260925 + fold)
            train_idx = np.sort(rng.choice(train_idx, MAX_TRAIN_ROWS, replace=False))
        model = _fit_eod(
            data.iloc[train_idx][EOD_FEATURES].astype(np.float32).reset_index(drop=True),
            label.iloc[train_idx].reset_index(drop=True), "hgb_classifier",
        )
        val = data.loc[dates.between(val_start, val_end)].copy().reset_index(drop=True)
        pred = _predict_scores(model, val, "hgb_classifier")
        for multiplier in (1.0, 1.5, 2.0):
            fold_rows.append({
                "fold": fold, "training_end": train_end,
                "validation": [val_start, val_end],
                "slippage_multiplier": multiplier,
                "metrics": simulate_eod(val, pred, 0.50,
                                         slippage_multiplier=multiplier,
                                         tax_policy="historical"),
            })
        del model, val, pred

    summaries = {}
    for multiplier in (1.0, 1.5, 2.0):
        items = [x for x in fold_rows if x["slippage_multiplier"] == multiplier]
        metrics = [x["metrics"] for x in items]
        weights = [x["trades"] for x in metrics]
        total_trades = sum(weights)
        summaries[str(multiplier)] = {
            "trades": total_trades,
            "pooled_mean_trade_bps": float(sum(m["mean_trade_bps"] * w for m, w in zip(metrics, weights)) / total_trades) if total_trades else 0.0,
            "positive_trade_expectancy_folds": sum(m["mean_trade_bps"] > 0 for m in metrics),
            "minimum_fold_trades": min(weights),
            "mean_excess_baseline_pct": float(np.mean([m["mean_excess_baseline_pct"] for m in metrics])),
        }
    result = {
        "purpose": "fixed-candidate slippage sensitivity only; not a new selection or independent test",
        "candidate": {"model_spec": "hgb_classifier", "probability_threshold": 0.50},
        "candidate_rationale": "v5/v6 validation had a positive pooled trade average but failed the stability and per-fold trade-count gates",
        "fee_and_tax_unchanged": True,
        "slippage_multipliers": [1.0, 1.5, 2.0],
        "summaries": summaries,
        "fold_results": fold_rows,
        "independent_lock": False,
    }
    path = Path(output_path)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.features, args.output), ensure_ascii=False, indent=2))
