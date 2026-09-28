"""Research a distinct 5-minute stock model for 15/30/60-minute holds."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .data import load_stock_minute_candles
from .nminute import (
    FEATURES,
    add_forward_target,
    aggregate_five_minute,
    build_features,
    simulate_five_minute,
)


# Fold dates are fixed before this model's result is read. Each training end
# leaves a full 5-minute target horizon before validation begins.
FOLDS = [
    ("20251031", "20251103", "20251114"),
    ("20251128", "20251201", "20251212"),
    ("20251231", "20260105", "20260116"),
    ("20260130", "20260202", "20260213"),
    ("20260227", "20260302", "20260313"),
    ("20260331", "20260401", "20260414"),
    ("20260430", "20260504", "20260515"),
    ("20260529", "20260601", "20260612"),
]
FINAL_TRAIN_END = "20260731"
RETROSPECTIVE_AUDIT = ("20260803", "20260923")
HORIZONS = (15, 30, 60)
THRESHOLDS = (0.0029, 0.0045, 0.006, 0.008)
MODEL_SPECS = {
    "hgb_shallow": {
        "type": "HistGradientBoostingRegressor", "max_iter": 100,
        "learning_rate": 0.04, "max_leaf_nodes": 7,
        "min_samples_leaf": 100, "l2_regularization": 20.0,
    },
    "hgb_regularized": {
        "type": "HistGradientBoostingRegressor", "max_iter": 140,
        "learning_rate": 0.03, "max_leaf_nodes": 15,
        "min_samples_leaf": 250, "l2_regularization": 50.0,
    },
    "ridge": {"type": "Ridge", "alpha": 30.0},
}
INITIAL_BALANCE = 10_000_000.0
ALLOCATION = 0.25
PARTICIPATION = 0.01
BUY_COST = 0.00045  # 1.5bp fee + 3bp slippage
SELL_COST = 0.00245  # buy-side costs + 20bp stock transaction tax


def _fit(frame: pd.DataFrame, target: pd.Series, spec_name: str):
    spec = MODEL_SPECS[spec_name]
    if spec["type"] == "HistGradientBoostingRegressor":
        model = HistGradientBoostingRegressor(
            loss="squared_error", early_stopping=False, random_state=42,
            max_iter=spec["max_iter"], learning_rate=spec["learning_rate"],
            max_leaf_nodes=spec["max_leaf_nodes"],
            min_samples_leaf=spec["min_samples_leaf"],
            l2_regularization=spec["l2_regularization"],
        )
    else:
        model = make_pipeline(StandardScaler(), Ridge(alpha=spec["alpha"]))
    valid = target.notna().to_numpy()
    if int(valid.sum()) < 10_000:
        raise ValueError(f"학습 라벨 부족: {int(valid.sum()):,}")
    with threadpool_limits(limits=2):
        model.fit(frame.loc[valid, FEATURES], target.loc[valid].clip(-0.03, 0.03))
    return model


def _evaluate(model, frame: pd.DataFrame, horizon: int, threshold: float,
              *, cost_multiplier: float = 1.0, detail: bool = False):
    with threadpool_limits(limits=2):
        prediction = model.predict(frame[FEATURES])
    return simulate_five_minute(
        frame, prediction, horizon, threshold,
        initial_balance=INITIAL_BALANCE,
        allocation=ALLOCATION,
        participation=PARTICIPATION,
        buy_cost=BUY_COST * cost_multiplier,
        sell_cost=SELL_COST * cost_multiplier,
        detail=detail,
    )


def _candidate_rows(experiments: list[dict]) -> list[dict]:
    candidates = []
    for model_name in MODEL_SPECS:
        for horizon in HORIZONS:
            for threshold in THRESHOLDS:
                folds = [
                    row for row in experiments
                    if row["model_spec"] == model_name
                    and row["horizon_minutes"] == horizon
                    and row["entry_threshold"] == threshold
                ]
                if len(folds) != len(FOLDS):
                    continue
                excess = np.array([x["metrics"]["mean_excess_baseline_pct"] for x in folds])
                trades_ok = all(x["metrics"]["trades"] >= 30 for x in folds)
                truncations_ok = all(x["metrics"]["truncated_exits"] == 0 for x in folds)
                positive_trade_edge = all(x["metrics"]["mean_trade_bps"] > 0 for x in folds)
                score = float(excess.mean() - 0.5 * excess.std(ddof=0))
                candidates.append({
                    "model_spec": model_name,
                    "horizon_minutes": horizon,
                    "entry_threshold": threshold,
                    "folds": folds,
                    "score_mean_excess_minus_half_std_pct": score,
                    "mean_excess_baseline_pct": float(excess.mean()),
                    "positive_excess_folds": int((excess > 0).sum()),
                    "eligible": bool(trades_ok and truncations_ok and positive_trade_edge),
                })
    return candidates


def run(output: str, start_date: str = "20250901", end_date: str = "20260923",
        run_retrospective_audit: bool = True, resume: bool = False) -> dict:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=resume)
    feature_cache = out / "features.pkl"
    if resume and feature_cache.exists():
        with feature_cache.open("rb") as stream:
            data = pickle.load(stream)
    else:
        minute = load_stock_minute_candles(start_date, end_date)
        if minute.empty:
            raise ValueError("주식 분봉이 없습니다.")
        bars = aggregate_five_minute(minute)
        data = build_features(bars)
        with feature_cache.open("wb") as stream:
            pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    date_text = data.date.astype(str)
    session_quality = data.groupby(["code", "date"], sort=False).bar_start.agg(
        bars="size", first="min", last="max"
    )
    quality = {
        "five_minute_rows": int(len(data)),
        "date_min": str(date_text.min()),
        "date_max": str(date_text.max()),
        "stocks": int(data.code.nunique()),
        "stock_sessions": int(len(session_quality)),
        "median_complete_bars_per_session": float(session_quality.bars.median()),
        "p05_complete_bars_per_session": float(session_quality.bars.quantile(0.05)),
        "sessions_with_70_plus_complete_bars_fraction": float((session_quality.bars >= 70).mean()),
    }
    if quality["date_min"] > FOLDS[0][0] or quality["date_max"] < RETROSPECTIVE_AUDIT[1]:
        raise ValueError(f"필요한 분봉 날짜 범위가 부족합니다: {quality}")

    targets = {horizon: add_forward_target(data, horizon) for horizon in HORIZONS}
    checkpoint = out / "experiments.json"
    experiments = json.loads(checkpoint.read_text(encoding="utf-8")) if resume and checkpoint.exists() else []
    completed = {(x["fold"], x["horizon_minutes"], x["model_spec"]) for x in experiments}
    for fold_index, (train_end, val_start, val_end) in enumerate(FOLDS, start=1):
        train_dates = date_text <= train_end
        validation = data[date_text.between(val_start, val_end)].copy().reset_index(drop=True)
        if validation.empty:
            raise ValueError(f"검증 분봉이 없습니다: {val_start}..{val_end}")
        for horizon in HORIZONS:
            target = targets[horizon]
            train_target = target.loc[train_dates].reset_index(drop=True)
            train_features = data.loc[train_dates, FEATURES].reset_index(drop=True)
            for model_name in MODEL_SPECS:
                key = (fold_index, horizon, model_name)
                if key in completed:
                    continue
                model = _fit(train_features, train_target, model_name)
                # All thresholds share this fold/model prediction and simulator input.
                with threadpool_limits(limits=2):
                    prediction = model.predict(validation[FEATURES])
                for threshold in THRESHOLDS:
                    metrics = simulate_five_minute(
                        validation, prediction, horizon, threshold,
                        initial_balance=INITIAL_BALANCE, allocation=ALLOCATION,
                        participation=PARTICIPATION, buy_cost=BUY_COST,
                        sell_cost=SELL_COST,
                    )
                    experiments.append({
                        "fold": fold_index,
                        "train_end": train_end,
                        "validation_start": val_start,
                        "validation_end": val_end,
                        "model_spec": model_name,
                        "horizon_minutes": horizon,
                        "entry_threshold": threshold,
                        "metrics": metrics,
                    })
                checkpoint_tmp = checkpoint.with_suffix(".json.tmp")
                checkpoint_tmp.write_text(
                    json.dumps(experiments, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                checkpoint_tmp.replace(checkpoint)
                completed.add(key)
                print(f"fold {fold_index}/{len(FOLDS)} {model_name} horizon={horizon} done", flush=True)

    # Store every experiment result before considering the old v1 audit dates.
    (out / "experiments.json").write_text(
        json.dumps(experiments, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    candidates = _candidate_rows(experiments)
    passed = [
        row for row in candidates
        if row["eligible"]
        and row["mean_excess_baseline_pct"] > 0
        and row["positive_excess_folds"] >= 6
        and row["score_mean_excess_minus_half_std_pct"] > 0
    ]
    selected = max(passed, key=lambda row: row["score_mean_excess_minus_half_std_pct"]) if passed else None
    report = {
        "model_name": "stock-single-intraday-nminute-research-v2",
        "product_family": "stock-single-intraday",
        "model_distinction": "5-minute OHLCV aggregation with 15/30/60-minute fixed holding targets; separate from the prior 1-minute signal research",
        "source": "Kiwoom ka10080 stock_minute_candle; ETF codes excluded",
        "data_period": {"start": quality["date_min"], "end": quality["date_max"]},
        "universe": sorted(data.code.unique()),
        "universe_limitation": "fixed 2026 major-stock list applied retrospectively; point-in-time membership is unavailable",
        "data_quality": quality,
        "feature_names": FEATURES,
        "feature_count": len(FEATURES),
        "target": "5-minute bar close signal; enter next 5-minute bar open; exit at next-open after fixed 15/30/60-minute hold",
        "validation_folds": FOLDS,
        "selection_gate": "at least 30 trades and zero truncated exits in every fold; positive mean trade edge; positive mean excess over matched 09:30-15:15 baseline; positive risk-adjusted excess; positive excess in at least 6/8 folds",
        "cost_assumptions": {
            "buy_fee_and_slippage": BUY_COST,
            "sell_fee_tax_and_slippage": SELL_COST,
            "round_trip_total": BUY_COST + SELL_COST,
            "note": "0.015% fee each way, 0.03% slippage each way, and 0.20% sell tax; no quote-level spread/fill data",
        },
        "allocation_assumptions": {
            "starting_balance_per_stock": INITIAL_BALANCE,
            "maximum_cash_fraction_per_trade": ALLOCATION,
            "maximum_participation": PARTICIPATION,
        },
        "model_specs": MODEL_SPECS,
        "selection_status": "validation_quality_gate_passed" if selected else "no_candidate_passed_validation_quality_gate",
        "selected": selected,
        "candidates": candidates,
        "retrospective_audit_period": {
            "start": RETROSPECTIVE_AUDIT[0], "end": RETROSPECTIVE_AUDIT[1],
            "status": "already examined for prior 1-minute intraday v1; not a fresh independent lock",
        },
        "deployment_approved": False,
    }

    # A passing validation candidate may be fitted through July and reviewed on
    # the previously-seen August/September dates, but that audit cannot grant approval.
    if selected and run_retrospective_audit:
        horizon = selected["horizon_minutes"]
        train_dates = date_text <= FINAL_TRAIN_END
        final_target = targets[horizon].loc[train_dates].reset_index(drop=True)
        final_features = data.loc[train_dates, FEATURES].reset_index(drop=True)
        final_model = _fit(final_features, final_target, selected["model_spec"])
        locked = data[date_text.between(*RETROSPECTIVE_AUDIT)].copy().reset_index(drop=True)
        locked_prediction = final_model.predict(locked[FEATURES])
        cost_sensitivity = {}
        base_details = None
        for label, multiplier in (("base", 1.0), ("1_5x", 1.5), ("2x", 2.0)):
            metric = simulate_five_minute(
                locked, locked_prediction, horizon, selected["entry_threshold"],
                initial_balance=INITIAL_BALANCE, allocation=ALLOCATION,
                participation=PARTICIPATION, buy_cost=BUY_COST * multiplier,
                sell_cost=SELL_COST * multiplier, detail=(multiplier == 1.0),
            )
            if multiplier == 1.0:
                summary, trades, daily, stock_metrics = metric
                cost_sensitivity[label] = summary
                base_details = (trades, daily, stock_metrics)
            else:
                cost_sensitivity[label] = metric
        candidate = {
            "model": final_model,
            "feature_names": FEATURES,
            "horizon_minutes": horizon,
            "entry_threshold": selected["entry_threshold"],
            "train_end_date": FINAL_TRAIN_END,
            "deployment_approved": False,
        }
        joblib.dump(candidate, out / "diagnostic_candidate.joblib", compress=3)
        trades, daily, stock_metrics = base_details
        trades.to_csv(out / "retrospective_trades.csv", index=False)
        daily.to_csv(out / "retrospective_daily.csv", index=False)
        stock_metrics.to_csv(out / "retrospective_by_stock.csv", index=False)
        report["retrospective_audit"] = cost_sensitivity
        report["retrospective_audit_result_status"] = "historical_recheck_not_independent"

    manifest = {
        "model_name": report["model_name"],
        "product_family": report["product_family"],
        "data_period": report["data_period"],
        "feature_names": FEATURES,
        "validation_folds": FOLDS,
        "selection_gate": report["selection_gate"],
        "selection_status": report["selection_status"],
        "selected": selected,
        "retrospective_audit_period": report["retrospective_audit_period"],
        "deployment_approved": False,
    }
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start-date", default="20250901")
    parser.add_argument("--end-date", default="20260923")
    parser.add_argument("--skip-retrospective-audit", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run(
        args.output, args.start_date, args.end_date,
        run_retrospective_audit=not args.skip_retrospective_audit,
        resume=args.resume,
    )
    print(json.dumps({
        "selection_status": result["selection_status"],
        "selected": result["selected"],
        "deployment_approved": False,
    }, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
