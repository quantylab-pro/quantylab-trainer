"""Explore causal same-day exit rules for stock-single-intraday-v1."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .research_eod import (
    BROKER_FEE_PER_SIDE,
    EOD_FEATURES,
    FOLDS,
    MAX_TRAIN_ROWS,
    SLIPPAGE_PER_SIDE,
    _fit_eod,
    _predict_scores,
    _round_trip_costs,
    add_eod_target,
    simulate_eod,
)
from .research_nminute import ALLOCATION, INITIAL_BALANCE


MODEL_SPEC = "hgb_classifier"
PROBABILITY_THRESHOLDS = (0.38, 0.42)
EXIT_RULES = {
    "eod": {"stop": None, "take": None, "trailing": None},
    "stop_50bp": {"stop": 0.005, "take": None, "trailing": None},
    "stop_100bp": {"stop": 0.010, "take": None, "trailing": None},
    "stop_50_take_80bp": {"stop": 0.005, "take": 0.008, "trailing": None},
    "stop_100_take_120bp": {"stop": 0.010, "take": 0.012, "trailing": None},
    "trailing_75bp": {"stop": None, "take": None, "trailing": 0.0075},
}
BUY_COST = BROKER_FEE_PER_SIDE + SLIPPAGE_PER_SIDE
SELL_COST = BUY_COST + 0.0020


def _trade_exit(session: pd.DataFrame, entry_i: int, entry_price: float,
                rule: dict) -> tuple[int, float, str]:
    """Check completed bars, then use the following bar's open for exits."""
    starts = session.bar_start.to_numpy(dtype=int)
    closes = session.close.to_numpy(dtype=float)
    opens = session.open.to_numpy(dtype=float)
    last_i = len(session) - 1
    peak_close = entry_price
    for i in range(entry_i, last_i):
        next_i = i + 1
        if starts[next_i] - starts[i] != 5:
            return next_i, float(opens[next_i]), "data_gap_next_open"
        close = float(closes[i])
        peak_close = max(peak_close, close)
        gross = close / entry_price - 1.0
        if rule["stop"] is not None and gross <= -rule["stop"]:
            return next_i, float(opens[next_i]), "stop_next_open"
        if rule["take"] is not None and gross >= rule["take"]:
            return next_i, float(opens[next_i]), "take_next_open"
        if rule["trailing"] is not None and close / peak_close - 1.0 <= -rule["trailing"]:
            return next_i, float(opens[next_i]), "trailing_next_open"
    return last_i, float(closes[last_i]), "final_complete_bar_close"


def _evaluate(data: pd.DataFrame, prediction: np.ndarray, threshold: float,
              rule_name: str, baseline_by_code: dict[str, float]) -> tuple[dict, list[dict]]:
    rule = EXIT_RULES[rule_name]
    bars = data.reset_index(drop=True).copy()
    bars["prediction"] = np.asarray(prediction, dtype=float)
    balances = {code: INITIAL_BALANCE for code in bars.code.unique()}
    peaks = balances.copy()
    drawdowns = {code: 0.0 for code in balances}
    trades: list[dict] = []
    daily: list[dict] = []

    for (code, date), session in bars.groupby(["code", "date"], sort=True):
        session = session.sort_values("bar_start").reset_index(drop=True)
        cash_before = balances[code]
        cash = cash_before
        trade = None
        if len(session) > 2 and int(session.bar_start.iloc[-1]) == 915:
            candidates = session.index[
                session.bar_start.between(565, 885)
                & session.prediction.gt(threshold)
            ]
            for signal_i in candidates:
                entry_i = int(signal_i) + 1
                if entry_i >= len(session) - 1:
                    continue
                if int(session.bar_start.iloc[entry_i] - session.bar_start.iloc[signal_i]) != 5:
                    continue
                entry_price = float(session.open.iloc[entry_i])
                cap = max(int(session.capacity_shares.iloc[signal_i]), 0)
                shares = int(min(cash * ALLOCATION / (entry_price * (1 + BUY_COST)), cap)) if entry_price > 0 else 0
                if shares <= 0:
                    continue
                exit_i, exit_price, reason = _trade_exit(session, entry_i, entry_price, rule)
                if exit_price <= 0:
                    continue
                buy_notional = shares * entry_price
                sell_notional = shares * exit_price
                invested = buy_notional * (1 + BUY_COST)
                proceeds = sell_notional * (1 - SELL_COST)
                cash += proceeds - invested
                trade = {
                    "code": code, "date": str(date), "signal_bar_start": int(session.bar_start.iloc[signal_i]),
                    "entry_bar_start": int(session.bar_start.iloc[entry_i]),
                    "exit_bar_start": int(session.bar_start.iloc[exit_i]),
                    "entry_price": entry_price, "exit_price": exit_price,
                    "shares": shares, "prediction": float(session.prediction.iloc[signal_i]),
                    "gross_bps": (exit_price / entry_price - 1) * 10000,
                    "net_bps": (proceeds / invested - 1) * 10000,
                    "net_pnl": proceeds - invested,
                    "holding_minutes": int(session.bar_start.iloc[exit_i] - session.bar_start.iloc[entry_i]
                                           + (5 if reason == "final_complete_bar_close" else 0)),
                    "exit_reason": reason,
                }
                break
        balances[code] = cash
        peaks[code] = max(peaks[code], cash)
        drawdowns[code] = min(drawdowns[code], cash / peaks[code] - 1.0)
        daily.append({"code": code, "date": str(date), "model_return": cash / cash_before - 1.0})
        if trade is not None:
            trades.append(trade)

    by_stock = pd.DataFrame([
        {"code": code, "model_return_pct": (cash / INITIAL_BALANCE - 1) * 100,
         "baseline_return_pct": baseline_by_code[code], "max_drawdown_pct": drawdowns[code] * 100}
        for code, cash in balances.items()
    ])
    trade_frame = pd.DataFrame(trades)
    daily_frame = pd.DataFrame(daily)
    date_returns = daily_frame.groupby("date", sort=True).model_return.mean().to_numpy(dtype=float)
    summary = {
        "stocks": len(by_stock), "trades": len(trade_frame),
        "mean_return_pct": float(by_stock.model_return_pct.mean()),
        "mean_excess_baseline_pct": float((by_stock.model_return_pct - by_stock.baseline_return_pct).mean()),
        "mean_trade_bps": float(trade_frame.net_bps.mean()) if len(trade_frame) else 0.0,
        "median_trade_bps": float(trade_frame.net_bps.median()) if len(trade_frame) else 0.0,
        "win_rate": float((trade_frame.net_pnl > 0).mean()) if len(trade_frame) else 0.0,
        "mean_holding_minutes": float(trade_frame.holding_minutes.mean()) if len(trade_frame) else 0.0,
        "gap_exits": int(trade_frame.exit_reason.eq("data_gap_next_open").sum()) if len(trade_frame) else 0,
        "worst_stock_drawdown_pct": float(by_stock.max_drawdown_pct.min()),
        "cross_sectional_daily_sharpe": float(date_returns.mean() / (date_returns.std(ddof=0) + 1e-12) * np.sqrt(252)) if len(date_returns) > 1 else 0.0,
    }
    return summary, trades


def run_fold(features_cache: str, output: str, fold: int) -> None:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    with Path(features_cache).open("rb") as stream:
        data = pickle.load(stream)
    date_text = data.date.astype(str)
    target = add_eod_target(data)
    labels = (target > _round_trip_costs(data.date, "current")).where(target.notna())
    train_end, val_start, val_end = FOLDS[fold - 1]
    train_idx = np.flatnonzero((date_text.le(train_end) & target.notna()).to_numpy())
    if len(train_idx) > MAX_TRAIN_ROWS:
        train_idx = np.sort(np.random.default_rng(20260925 + fold).choice(
            train_idx, MAX_TRAIN_ROWS, replace=False))
    model = _fit_eod(
        data.iloc[train_idx][EOD_FEATURES].astype(np.float32).reset_index(drop=True),
        labels.iloc[train_idx].reset_index(drop=True), MODEL_SPEC)
    val = data.loc[date_text.between(val_start, val_end)].copy().reset_index(drop=True)
    prediction = _predict_scores(model, val, MODEL_SPEC)
    _, _, _, baseline = simulate_eod(val, np.zeros(len(val)), 1.0,
                                      tax_policy="current", detail=True)
    baseline_by_code = dict(zip(baseline.code, baseline.baseline_return_pct))
    experiments = []
    all_trades = []
    for threshold in PROBABILITY_THRESHOLDS:
        for rule_name in EXIT_RULES:
            metrics, trades = _evaluate(val, prediction, threshold, rule_name, baseline_by_code)
            experiments.append({
                "fold": fold, "train_end": train_end,
                "validation_start": val_start, "validation_end": val_end,
                "training_rows": len(train_idx), "model_spec": MODEL_SPEC,
                "probability_threshold": threshold, "exit_rule": rule_name,
                "metrics": metrics,
            })
            for trade in trades:
                all_trades.append({"fold": fold, "probability_threshold": threshold,
                                   "exit_rule": rule_name, **trade})
            print(f"fold {fold}/8 p>{threshold:.2f} {rule_name}: "
                  f"{metrics['trades']} trades, {metrics['mean_trade_bps']:.1f}bp", flush=True)
    fold_path = out / f"fold_{fold:02d}.json"
    temp = fold_path.with_suffix(".json.tmp")
    temp.write_text(json.dumps({"experiments": experiments, "trades": all_trades},
                               ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(fold_path)


def _audit_bar_gaps(features_cache: str, trades: list[dict], out: Path) -> dict:
    with Path(features_cache).open("rb") as stream:
        data = pickle.load(stream)
    sessions = data.groupby(["code", "date"]).bar_start.agg(
        bars="size", first="min", last="max").reset_index()
    sessions["date"] = sessions.date.astype(str)
    expected = ((sessions["last"] - sessions["first"]) // 5 + 1).astype(int)
    sessions["internal_missing_bars"] = (expected - sessions.bars).clip(lower=0)
    sessions["continuous_observed_range"] = sessions.internal_missing_bars.eq(0)
    sessions["late_start"] = sessions["first"].gt(540)
    sessions["full_0900_to_1515"] = (
        sessions.bars.eq(76) & sessions["first"].eq(540) & sessions["last"].eq(915))

    fold_quality = []
    for fold, (_, start, end) in enumerate(FOLDS, 1):
        subset = sessions[sessions.date.between(start, end)]
        fold_quality.append({
            "fold": fold, "validation_start": start, "validation_end": end,
            "sessions": len(subset),
            "sessions_with_internal_gaps": int((~subset.continuous_observed_range).sum()),
            "internal_missing_five_minute_bars": int(subset.internal_missing_bars.sum()),
            "late_start_sessions": int(subset.late_start.sum()),
        })

    trade_frame = pd.DataFrame(trades).merge(
        sessions[["code", "date", "continuous_observed_range"]],
        on=["code", "date"], how="left")
    sensitivity = []
    for (threshold, rule), subset in trade_frame.groupby(["probability_threshold", "exit_rule"]):
        subset = subset[subset.continuous_observed_range]
        by_fold = subset.groupby("fold").net_bps.agg(["size", "mean"]).reindex(
            range(1, len(FOLDS) + 1), fill_value=0)
        sensitivity.append({
            "probability_threshold": float(threshold), "exit_rule": rule,
            "trades": len(subset), "minimum_fold_trades": int(by_fold["size"].min()),
            "mean_net_bps": float(subset.net_bps.mean()) if len(subset) else 0.0,
            "positive_trade_edge_folds": int((by_fold["mean"] > 0).sum()),
        })

    market_wide_absences = {}
    for date, day in data.groupby("date", sort=False):
        if day.code.nunique() < 25:
            continue
        observed = set(day.bar_start.unique())
        missing = sorted(set(range(int(day.bar_start.min()), int(day.bar_start.max()) + 1, 5)) - observed)
        if len(missing) >= 3:
            market_wide_absences[str(date)] = missing
    worst = sessions.groupby("date").internal_missing_bars.sum().sort_values(ascending=False)
    audit = {
        "scope": "post-hoc data-quality sensitivity; future continuity is not known at entry",
        "expected_full_day_bars": 76, "all_sessions": len(sessions),
        "full_0900_to_1515_sessions": int(sessions.full_0900_to_1515.sum()),
        "late_start_sessions": int(sessions.late_start.sum()),
        "sessions_with_internal_gaps": int((~sessions.continuous_observed_range).sum()),
        "total_internal_missing_five_minute_bars": int(sessions.internal_missing_bars.sum()),
        "fold_quality": fold_quality,
        "worst_dates_internal_missing_bars": {str(date): int(count)
                                               for date, count in worst.head(10).items()},
        "market_wide_absent_internal_bars": market_wide_absences,
        "continuous_session_trade_diagnostics": sensitivity,
    }
    (out / "bar_gap_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit


def finalize(output: str, features_cache: str) -> dict:
    out = Path(output)
    fold_paths = [out / f"fold_{fold:02d}.json" for fold in range(1, len(FOLDS) + 1)]
    if not all(path.exists() for path in fold_paths):
        raise ValueError("all eight fold checkpoints are required")
    parts = [json.loads(path.read_text(encoding="utf-8")) for path in fold_paths]
    experiments = [row for part in parts for row in part["experiments"]]
    trades = [row for part in parts for row in part["trades"]]
    data_quality_audit = _audit_bar_gaps(features_cache, trades, out)
    candidates = []
    for threshold in PROBABILITY_THRESHOLDS:
        for rule_name in EXIT_RULES:
            rows = [row for row in experiments if row["probability_threshold"] == threshold
                    and row["exit_rule"] == rule_name]
            metrics = [row["metrics"] for row in rows]
            counts = [item["trades"] for item in metrics]
            total_trades = sum(counts)
            pooled = sum(item["mean_trade_bps"] * count for item, count in zip(metrics, counts)) / total_trades if total_trades else 0.0
            edge_folds = sum(item["mean_trade_bps"] > 0 for item in metrics)
            return_folds = sum(item["mean_return_pct"] > 0 for item in metrics)
            excess_folds = sum(item["mean_excess_baseline_pct"] > 0 for item in metrics)
            candidates.append({
                "model_spec": MODEL_SPEC, "probability_threshold": threshold,
                "exit_rule": rule_name, "trades": total_trades,
                "minimum_fold_trades": min(counts),
                "pooled_mean_trade_bps": float(pooled),
                "positive_trade_edge_folds": edge_folds,
                "positive_return_folds": return_folds,
                "positive_excess_folds": excess_folds,
                "mean_excess_baseline_pct": float(np.mean([item["mean_excess_baseline_pct"] for item in metrics])),
                "gap_exits": sum(item["gap_exits"] for item in metrics),
                "passes_quality_gate": bool(min(counts) >= 30 and pooled > 0 and
                                            edge_folds >= 6 and return_folds >= 6 and
                                            excess_folds >= 6),
                "folds": rows,
            })
    report = {
        "model_name": "stock-single-intraday-v1",
        "experiment_id": "eod-exit-exp02",
        "hypothesis": "a causal stop, profit target, or trailing exit may improve broad EOD signals",
        "source_experiment": "eod-current-tax-exp01",
        "tax_policy": "current rates applied to every historical row",
        "round_trip_base_cost": 0.0029,
        "entry": "completed 5-minute bar signal; next 5-minute open",
        "exit": "completed 5-minute close triggers next 5-minute open; otherwise final complete bar close",
        "data_gap_policy": "if the next expected 5-minute bar is absent while holding, exit at the next observed open; actual fill timing is uncertain",
        "data_quality_audit": "bar_gap_audit.json",
        "data_quality_summary": {
            "sessions_with_internal_gaps": data_quality_audit["sessions_with_internal_gaps"],
            "total_internal_missing_five_minute_bars": data_quality_audit["total_internal_missing_five_minute_bars"],
        },
        "probability_thresholds": PROBABILITY_THRESHOLDS,
        "exit_rules": EXIT_RULES,
        "validation_folds": FOLDS,
        "selection_gate": "at least 30 trades in every fold, positive pooled net trade mean, and positive trade mean, cumulative return, and baseline excess in at least 6/8 folds",
        "previously_examined_period": "same validation folds as exp01; not independent confirmation",
        "selection_status": "validation_quality_gate_passed" if any(x["passes_quality_gate"] for x in candidates) else "no_candidate_passed_validation_quality_gate",
        "candidates": candidates,
        "deployment_approved": False,
        "official_version_increment": False,
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(trades).to_csv(out / "trades.csv", index=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--features-cache", required=True)
    parser.add_argument("--fold", type=int, choices=range(1, len(FOLDS) + 1))
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.finalize:
        report = finalize(args.output, args.features_cache)
        print(json.dumps({"selection_status": report["selection_status"],
                          "candidates": len(report["candidates"])}, ensure_ascii=False))
    else:
        if args.fold is None:
            parser.error("--fold is required unless --finalize is set")
        run_fold(args.features_cache, args.output, args.fold)


if __name__ == "__main__":
    main()
