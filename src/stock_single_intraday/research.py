"""Train and assess a research-only stock single-intraday model candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from ..etf_single_intraday.minute_engine import BASE, EXTRA, features, simulate
from .data import load_stock_minute_candles

# Expanding windows are fixed before looking at model results. July onward is
# kept untouched until the validation winner and threshold have been selected.
FOLDS = [
    ("20260430", "20260504", "20260515"),
    ("20260529", "20260601", "20260612"),
    ("20260630", "20260701", "20260715"),
]
LOCKED = ("20260731", "20260803", "20260923")
HORIZONS = (5, 15, 30)
THRESHOLDS = (0.0025, 0.004, 0.006)
FEATURES = BASE + EXTRA


def _fit(train: pd.DataFrame, horizon: int):
    model = HistGradientBoostingRegressor(
        max_iter=120,
        max_leaf_nodes=7,
        min_samples_leaf=100,
        l2_regularization=20,
        learning_rate=0.04,
        early_stopping=False,
        random_state=42,
    )
    target_col = f"y{horizon}"
    with threadpool_limits(limits=2):
        model.fit(train[FEATURES], train[target_col].clip(-0.03, 0.03))
    return model


def _tradeable_rows(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    # Labels end before the close and are only sampled every five minutes.
    return frame[(frame.minute % 5 == 0) & frame[f"y{horizon}"].notna()]


def _stock_metrics(metric: dict) -> dict:
    """Rename the ETF engine's legacy instrument counter in report output."""
    metric = dict(metric)
    if "etfs" in metric:
        metric["stocks"] = metric.pop("etfs")
    return metric


def run(output: str, start_date: str = "20260129", end_date: str = "20260923",
        cost: float = 0.0029, min_train_rows: int = 10_000) -> dict:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    if cost < 0:
        raise ValueError("cost must be non-negative")
    candles = load_stock_minute_candles(start_date, end_date)
    if candles.empty:
        raise ValueError("대상 종목 분봉이 없습니다. stock_minute_candle 수집 상태를 확인하세요.")
    data = features(candles)
    dates = sorted(data.date.astype(str).unique())
    needed = {x for fold in FOLDS for x in fold[1:]} | {LOCKED[1], LOCKED[2]}
    absent = sorted(d for d in needed if not any(d <= observed for observed in dates))
    if absent:
        raise ValueError(f"검증/잠금 날짜 범위의 분봉이 부족합니다: {absent}")

    rows = []
    for horizon in HORIZONS:
        per_threshold = {threshold: [] for threshold in THRESHOLDS}
        for train_end, val_start, val_end in FOLDS:
            train = _tradeable_rows(data[data.date <= train_end], horizon)
            val = data[data.date.between(val_start, val_end)]
            if len(train) < min_train_rows or val.empty:
                raise ValueError(
                    f"학습 표본/검증 구간 부족: horizon={horizon}, "
                    f"train_rows={len(train)}, validation={val_start}..{val_end}"
                )
            model = _fit(train, horizon)
            with threadpool_limits(limits=2):
                prediction = model.predict(val[FEATURES])
            for threshold in THRESHOLDS:
                metric = _stock_metrics(simulate(
                    val, prediction, horizon, threshold, cost=cost, detail=False
                ))
                per_threshold[threshold].append(metric)
        for threshold, folds in per_threshold.items():
            returns = [item["mean_return_pct"] for item in folds]
            eligible = all(item["trades"] >= 20 and item["truncated_exits"] == 0
                           for item in folds)
            rows.append({
                "horizon_minutes": horizon,
                "entry_threshold": threshold,
                "folds": folds,
                "score_mean_minus_half_std_pct": float(np.mean(returns) - 0.5 * np.std(returns)),
                "eligible": eligible,
            })

    eligible = [
        row for row in rows
        if row["eligible"]
        and row["score_mean_minus_half_std_pct"] > 0
        and all(fold["mean_return_pct"] > 0 for fold in row["folds"])
    ]
    if not eligible:
        report = {
            "model_name": "stock-single-intraday-research",
            "product_family": "stock-single-intraday",
            "universe": sorted(data.code.unique()),
            "universe_limitation": "fixed 2026 major-stock list applied retrospectively; point-in-time membership is unavailable",
            "source": "stock_minute_candle; ETF codes excluded",
            "data_period": {"start": str(data.date.min()), "end": str(data.date.max())},
            "rows": len(data),
            "train_validation_protocol": FOLDS,
            "locked_period": {"start": LOCKED[1], "end": LOCKED[2]},
            "selection_status": "no_candidate_passed_validation_quality_gate",
            "selection_gate": "all validation folds net-positive, at least 20 trades per fold, zero truncated exits, and positive mean_minus_half_std score",
            "selected": None,
            "locked_evaluation": None,
            "experiments": rows,
            "cost_assumption": {"symmetric_round_trip_rate": cost},
            "deployment_approved": False,
        }
        serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        (out / "report.json").write_text(serialized)
        (out / "manifest.json").write_text(json.dumps({
            "model_name": report["model_name"],
            "product_family": report["product_family"],
            "data_period": report["data_period"],
            "feature_names": FEATURES,
            "validation_folds": FOLDS,
            "selection_status": report["selection_status"],
            "selection_gate": report["selection_gate"],
            "deployment_approved": False,
        }, ensure_ascii=False, indent=2) + "\n")
        return report
    selected = max(eligible, key=lambda row: row["score_mean_minus_half_std_pct"])
    train = _tradeable_rows(data[data.date <= LOCKED[0]], selected["horizon_minutes"])
    model = _fit(train, selected["horizon_minutes"])
    locked = data[data.date.between(LOCKED[1], LOCKED[2])]
    if locked.empty:
        raise ValueError(f"잠금 평가 분봉이 없습니다: {LOCKED[1]}..{LOCKED[2]}")
    with threadpool_limits(limits=2):
        prediction = model.predict(locked[FEATURES])
    cost_sensitivity = {}
    base_metric = base_trades = base_daily = None
    for label, multiplier in (("base", 1.0), ("1_5x_cost", 1.5), ("2x_cost", 2.0)):
        metric, trades, daily = simulate(
            locked, prediction, selected["horizon_minutes"], selected["entry_threshold"],
            cost=cost * multiplier, detail=True,
        )
        cost_sensitivity[label] = _stock_metrics(metric)
        if multiplier == 1.0:
            base_metric, base_trades, base_daily = metric, trades, daily
    metric = _stock_metrics(base_metric)
    candidate = {
        "model": model,
        "features": FEATURES,
        "codes": sorted(data.code.unique()),
        "horizon_minutes": selected["horizon_minutes"],
        "entry_threshold": selected["entry_threshold"],
        "training_end": LOCKED[0],
        "cost_assumption_symmetric_round_trip": cost,
        "deployment_approved": False,
    }
    joblib.dump(candidate, out / "selected_candidate.joblib")
    base_trades.to_csv(out / "locked_trades.csv", index=False)
    base_daily.to_csv(out / "locked_daily_by_stock.csv", index=False)
    report = {
        "model_name": "stock-single-intraday-research",
        "product_family": "stock-single-intraday",
        "universe": candidate["codes"],
        "universe_limitation": "fixed 2026 major-stock list applied retrospectively; point-in-time membership is unavailable",
        "source": "stock_minute_candle; ETF codes excluded",
        "data_period": {"start": str(data.date.min()), "end": str(data.date.max())},
        "rows": len(data),
        "train_validation_protocol": FOLDS,
        "locked_period": {"start": LOCKED[1], "end": LOCKED[2]},
        "selection_status": "validation_quality_gate_passed",
        "selection_gate": "all validation folds net-positive, at least 20 trades per fold, zero truncated exits, and positive mean_minus_half_std score",
        "label": "next-open entry to next-open exit at fixed same-session horizon",
        "feature_availability": "completed minute bar; next minute open execution",
        "cost_assumption": {
            "symmetric_round_trip_rate": cost,
            "note": "Research approximation based on existing stock swing assumptions; asymmetric sell tax, measured spread, slippage and fill delay remain to be validated.",
        },
        "selected": selected,
        "locked_evaluation": metric,
        "locked_cost_sensitivity": cost_sensitivity,
        "experiments": rows,
        "deployment_approved": False,
    }
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    (out / "report.json").write_text(serialized)
    manifest = {
        "model_name": "stock-single-intraday-research",
        "product_family": "stock-single-intraday",
        "dataset": "stock_minute_candle major-stock universe",
        "train_end_date": LOCKED[0],
        "validation_folds": FOLDS,
        "locked_evaluation_period": {"start": LOCKED[1], "end": LOCKED[2]},
        "feature_names": FEATURES,
        "target": report["label"],
        "execution": report["feature_availability"],
        "cost_assumptions": report["cost_assumption"],
        "selected": selected,
        "locked_evaluation": metric,
        "locked_cost_sensitivity": cost_sensitivity,
        "code_version": "workspace research candidate",
        "deployment_approved": False,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start-date", default="20260129")
    parser.add_argument("--end-date", default="20260923")
    parser.add_argument("--cost", type=float, default=0.0029,
                        help="symmetric round-trip cost assumption (default 29bp)")
    parser.add_argument("--min-train-rows", type=int, default=10_000)
    args = parser.parse_args()
    report = run(args.output, args.start_date, args.end_date, args.cost, args.min_train_rows)
    print(json.dumps({"selected": report["selected"],
                      "locked_evaluation": report["locked_evaluation"],
                      "deployment_approved": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
