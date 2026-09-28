"""Research a 5-minute stock single model that exits at the same day's close."""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .research_nminute import (
    ALLOCATION,
    FOLDS,
    FINAL_TRAIN_END,
    INITIAL_BALANCE,
    PARTICIPATION,
    RETROSPECTIVE_AUDIT,
)
from .nminute import FEATURES as BASE_FEATURES
from quantylab.acquisition.kiwoom.stock_intraday_universe import MAJOR_STOCK_CODES


SESSION_FEATURES = [
    "return_from_open", "drawdown_from_day_high", "rebound_from_day_low",
    "close_location_in_day_range", "realized_volatility", "trend_efficiency",
    "time_of_day_sin", "time_of_day_cos",
    "bar_close_location", "upper_wick", "lower_wick",
    "buy_pressure_15m", "buy_pressure_60m", "cumulative_buy_pressure",
    "same_time_dollar_volume", "overnight_gap", "prior_session_return",
    "prior_5d_momentum", "prior_20d_momentum", "prior_20d_volatility",
    "prior_market_5d_momentum",
]
TICKER_FEATURES = [f"ticker_{str(code).zfill(6)}" for code in MAJOR_STOCK_CODES]
EOD_FEATURES = BASE_FEATURES + SESSION_FEATURES + TICKER_FEATURES
THRESHOLDS = (0.0030, 0.0045, 0.0060, 0.0080)
CLASS_THRESHOLDS = (0.38, 0.42, 0.46, 0.50, 0.54)
BROKER_FEE_PER_SIDE = 0.00015
SLIPPAGE_PER_SIDE = 0.00030
ENTRY_CUTOFF_START = 885  # 14:45 signal bar; next open is 14:50, close is about 15:20.
MIN_FOLD_TRADES = 30
MAX_TRAIN_ROWS = 150_000
RANK_CUTOFFS = (0.70, 0.80, 0.90)
ENTRY_WINDOWS = {"open_to_11": (565, 660), "midday": (660, 780), "afternoon": (780, 885)}
RULE_NAMES = ("rule_vwap_trend", "rule_opening_breakout", "rule_relative_strength")
EOD_MODEL_SPECS = {
    "hgb_shallow": {"type": "hgb", "max_iter": 100, "learning_rate": 0.04,
                    "max_leaf_nodes": 7, "min_samples_leaf": 100, "l2_regularization": 20.0},
    "hgb_regularized": {"type": "hgb", "max_iter": 140, "learning_rate": 0.03,
                        "max_leaf_nodes": 15, "min_samples_leaf": 250, "l2_regularization": 50.0},
    "hgb_tiny": {"type": "hgb", "max_iter": 80, "learning_rate": 0.03,
                 "max_leaf_nodes": 3, "min_samples_leaf": 500, "l2_regularization": 80.0},
    "hgb_balanced": {"type": "hgb", "max_iter": 180, "learning_rate": 0.025,
                     "max_leaf_nodes": 15, "min_samples_leaf": 150, "l2_regularization": 100.0},
    "ridge": {"type": "ridge", "alpha": 30.0},
    "ridge_strong": {"type": "ridge", "alpha": 100.0},
    "hgb_classifier": {"type": "hgb_classifier", "max_iter": 120, "learning_rate": 0.03,
                       "max_leaf_nodes": 7, "min_samples_leaf": 250, "l2_regularization": 60.0},
    "hgb_classifier_tiny": {"type": "hgb_classifier", "max_iter": 100, "learning_rate": 0.025,
                            "max_leaf_nodes": 3, "min_samples_leaf": 500, "l2_regularization": 100.0},
    "logistic_c_0_03": {"type": "logistic_classifier", "C": 0.03},
    "logistic_c_0_3": {"type": "logistic_classifier", "C": 0.3},
}


def _tax_rate(date: str, policy: str = "current") -> float:
    """Total KOSPI sell-side taxes under current-rate or historical-rate policy."""
    if policy == "current":
        return 0.0020
    if policy != "historical":
        raise ValueError(f"unknown tax policy: {policy}")
    return 0.0015 if str(date) < "20260101" else 0.0020


def _tax_components(date: str, policy: str = "current") -> tuple[float, float]:
    """Return securities transaction tax and rural special tax rates."""
    if policy == "current":
        return 0.0005, 0.0015
    if policy != "historical":
        raise ValueError(f"unknown tax policy: {policy}")
    return (0.0, 0.0015) if str(date) < "20260101" else (0.0005, 0.0015)


def _round_trip_costs(dates: pd.Series, tax_policy: str = "current") -> pd.Series:
    return dates.astype(str).map(
        lambda date: 2 * (BROKER_FEE_PER_SIDE + SLIPPAGE_PER_SIDE) + _tax_rate(date, tax_policy)
    ).astype(float)


def _trim_memory() -> None:
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (AttributeError, OSError):
        pass


def add_eod_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add causal session context; each feature uses only bars through that row."""
    result = data.copy()
    sessions = result.groupby(["code", "date"], sort=False)
    result["return_from_open"] = result.close / result.groupby(
        ["code", "date"], sort=False
    ).open.transform("first") - 1
    running_high = sessions.high.cummax()
    running_low = sessions.low.cummin()
    result["drawdown_from_day_high"] = result.close / running_high - 1
    result["rebound_from_day_low"] = result.close / running_low - 1
    day_range = (running_high - running_low).replace(0, np.nan)
    result["close_location_in_day_range"] = ((result.close - running_low) / day_range).fillna(0.5)
    ret = result.ret_5m.clip(-0.1, 0.1)
    result["realized_volatility"] = np.sqrt(
        ret.pow(2).groupby([result.code, result.date], sort=False).cumsum()
    )
    abs_ret = ret.abs().groupby([result.code, result.date], sort=False).cumsum()
    result["trend_efficiency"] = (result.return_from_open.abs() / abs_ret.replace(0, np.nan)).fillna(0)
    phase = np.clip((result.bar_start.to_numpy(dtype=float) - 540) / 375.0, 0.0, 1.0)
    result["time_of_day_sin"] = np.sin(2 * np.pi * phase)
    result["time_of_day_cos"] = np.cos(2 * np.pi * phase)

    bar_range = (result.high - result.low).replace(0, np.nan)
    result["bar_close_location"] = ((result.close - result.low) / bar_range).fillna(0.5)
    result["upper_wick"] = (result.high - result[["open", "close"]].max(axis=1)) / result.open
    result["lower_wick"] = (result[["open", "close"]].min(axis=1) - result.low) / result.open
    signed_volume = result.volume * (2 * result.bar_close_location - 1)
    total_volume = result.groupby(["code", "date"], sort=False).volume
    signed_group = signed_volume.groupby([result.code, result.date], sort=False)
    result["buy_pressure_15m"] = signed_group.rolling(3, min_periods=2).sum().reset_index(level=[0, 1], drop=True) / total_volume.rolling(3, min_periods=2).sum().reset_index(level=[0, 1], drop=True).replace(0, np.nan)
    result["buy_pressure_60m"] = signed_group.rolling(12, min_periods=6).sum().reset_index(level=[0, 1], drop=True) / total_volume.rolling(12, min_periods=6).sum().reset_index(level=[0, 1], drop=True).replace(0, np.nan)
    result["cumulative_buy_pressure"] = signed_group.cumsum() / total_volume.cumsum().replace(0, np.nan)

    dollar_volume = result.volume * result.close
    prior_dollar_volume = result.assign(_dollar_volume=dollar_volume).groupby(
        ["code", "bar_start"], sort=False
    )["_dollar_volume"].transform(lambda values: values.shift(1).rolling(20, min_periods=5).median())
    result["same_time_dollar_volume"] = dollar_volume / prior_dollar_volume.replace(0, np.nan)

    # Prior-session context is shifted one full day before being joined to today's bars.
    daily = result.groupby(["code", "date"], sort=False).agg(
        session_open=("open", "first"), session_close=("close", "last")
    )
    daily["daily_return"] = daily.session_close / daily.session_open - 1
    by_code = daily.groupby(level="code", sort=False)
    previous_close = by_code.session_close.shift(1)
    daily["overnight_gap"] = daily.session_open / previous_close - 1
    daily["prior_session_return"] = by_code.daily_return.shift(1)
    daily["prior_5d_momentum"] = by_code.daily_return.transform(
        lambda values: values.shift(1).rolling(5, min_periods=3).sum()
    )
    daily["prior_20d_momentum"] = by_code.daily_return.transform(
        lambda values: values.shift(1).rolling(20, min_periods=10).sum()
    )
    daily["prior_20d_volatility"] = by_code.daily_return.transform(
        lambda values: values.shift(1).rolling(20, min_periods=10).std()
    )
    market_daily = daily.daily_return.groupby(level="date", sort=False).mean().sort_index()
    prior_market_momentum = market_daily.shift(1).rolling(5, min_periods=3).sum()
    market_lookup = prior_market_momentum.to_dict()
    daily["prior_market_5d_momentum"] = daily.index.get_level_values("date").map(market_lookup)
    session_index = pd.MultiIndex.from_frame(result[["code", "date"]])
    for column in ("overnight_gap", "prior_session_return", "prior_5d_momentum",
                   "prior_20d_momentum", "prior_20d_volatility", "prior_market_5d_momentum"):
        result[column] = daily[column].reindex(session_index).to_numpy()
    one_hot = pd.get_dummies(result.code.astype(str), prefix="ticker", dtype=np.float32)
    for name in TICKER_FEATURES:
        result[name] = one_hot[name] if name in one_hot else 0.0
    result[EOD_FEATURES] = result[EOD_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return result


def add_eod_target(data: pd.DataFrame) -> pd.Series:
    """Return from next complete 5-minute bar open to final complete bar close."""
    sessions = data.groupby(["code", "date"], sort=False)
    entry = sessions.open.shift(-1)
    next_start = sessions.bar_start.shift(-1)
    final_start = sessions.bar_start.transform("last")
    final_close = sessions.close.transform("last")
    valid = (
        next_start.sub(data.bar_start).eq(5)
        & final_start.eq(915)
        & data.bar_start.le(ENTRY_CUTOFF_START)
        & entry.gt(0)
        & final_close.gt(0)
    )
    target = (final_close / entry - 1).where(valid)
    target.name = "target_eod"
    return target.astype(np.float32)


def _new_model(spec_name: str):
    spec = EOD_MODEL_SPECS[spec_name]
    if spec["type"] in ("hgb", "hgb_classifier"):
        model_class = HistGradientBoostingClassifier if spec["type"] == "hgb_classifier" else HistGradientBoostingRegressor
        params = {
            "max_iter": spec["max_iter"], "learning_rate": spec["learning_rate"],
            "max_leaf_nodes": spec["max_leaf_nodes"], "min_samples_leaf": spec["min_samples_leaf"],
            "l2_regularization": spec["l2_regularization"],
            "early_stopping": False, "random_state": 42,
        }
        if spec["type"] == "hgb":
            params["loss"] = "squared_error"
        return model_class(**params)
    if spec["type"] == "logistic_classifier":
        return make_pipeline(StandardScaler(), LogisticRegression(C=spec["C"], max_iter=500))
    return make_pipeline(StandardScaler(), Ridge(alpha=spec["alpha"]))


def _fit_eod(features: pd.DataFrame, target: pd.Series, spec_name: str):
    valid = target.notna().to_numpy()
    if int(valid.sum()) < 10_000:
        raise ValueError(f"EOD 학습 라벨 부족: {int(valid.sum()):,}")
    model = _new_model(spec_name)
    with threadpool_limits(limits=2):
        if EOD_MODEL_SPECS[spec_name]["type"].endswith("classifier"):
            model.fit(features.loc[valid, EOD_FEATURES], target.loc[valid].astype(int))
        else:
            model.fit(features.loc[valid, EOD_FEATURES], target.loc[valid].clip(-0.15, 0.15))
    return model


def _predict_scores(model, features: pd.DataFrame, spec_name: str) -> np.ndarray:
    with threadpool_limits(limits=2):
        if EOD_MODEL_SPECS[spec_name]["type"].endswith("classifier"):
            return model.predict_proba(features[EOD_FEATURES].astype(np.float32))[:, 1]
        return model.predict(features[EOD_FEATURES].astype(np.float32))


def simulate_eod(data: pd.DataFrame, prediction: np.ndarray, threshold: float,
                 *, signal_mask: np.ndarray | None = None,
                 time_window: tuple[int, int] | None = None,
                 slippage_multiplier: float = 1.0,
                 tax_policy: str = "current", detail: bool = False):
    """One long trade per stock/day, next-bar open entry, final-bar close exit."""
    if len(data) != len(prediction):
        raise ValueError("prediction and EOD data length mismatch")
    bars = data.reset_index(drop=True).copy()
    bars["prediction"] = np.asarray(prediction, dtype=float)
    if signal_mask is not None:
        if len(signal_mask) != len(bars):
            raise ValueError("signal mask and EOD data length mismatch")
        bars["rule_signal"] = np.asarray(signal_mask, dtype=bool)
    buy_cost = BROKER_FEE_PER_SIDE + SLIPPAGE_PER_SIDE * slippage_multiplier
    balances = {code: INITIAL_BALANCE for code in bars.code.unique()}
    baseline_balances = balances.copy()
    peaks, worst_dd = balances.copy(), {code: 0.0 for code in balances}
    trades, daily = [], []

    for (code, date), session in bars.groupby(["code", "date"], sort=True):
        security_tax_rate, rural_tax_rate = _tax_components(str(date), tax_policy)
        tax_rate = security_tax_rate + rural_tax_rate
        sell_cost = BROKER_FEE_PER_SIDE + SLIPPAGE_PER_SIDE * slippage_multiplier + tax_rate
        session = session.sort_values("bar_start").reset_index(drop=True)
        last_i = len(session) - 1
        last = session.iloc[last_i]
        cash_before = balances[code]
        cash = cash_before
        trade = None
        if int(last.bar_start) == 915:
            condition = session.rule_signal if signal_mask is not None else session.prediction > threshold
            signals = session.index[(session.bar_start >= 565)
                                    & (session.bar_start <= ENTRY_CUTOFF_START)
                                    & condition]
            if time_window is not None:
                signals = session.index[(session.bar_start >= time_window[0])
                                        & (session.bar_start < time_window[1])
                                        & condition]
            for i in signals:
                next_i = int(i) + 1
                if next_i >= last_i or int(session.bar_start.iloc[next_i] - session.bar_start.iloc[i]) != 5:
                    continue
                entry_price = float(session.open.iloc[next_i])
                exit_price = float(last.close)
                cap = max(int(session.capacity_shares.iloc[i]), 0)
                quantity = int(min(cash * ALLOCATION / (entry_price * (1 + buy_cost)), cap)) if entry_price > 0 else 0
                if quantity <= 0 or exit_price <= 0:
                    continue
                buy_notional = quantity * entry_price
                sell_notional = quantity * exit_price
                invested = buy_notional * (1 + buy_cost)
                proceeds = sell_notional * (1 - sell_cost)
                cash = cash - invested + proceeds
                trade = {
                    "code": code, "date": date,
                    "signal_time": f"{int(session.time.iloc[i]):04d}",
                    "entry_time": f"{int(session.bar_start.iloc[next_i]) // 60:02d}{int(session.bar_start.iloc[next_i]) % 60:02d}",
                    "exit_time": "15:19-minute close; final bar starts at 15:15",
                    "exit_bar_start": int(last.bar_start),
                    "holding_minutes": int(last.bar_start + 5 - session.bar_start.iloc[next_i]),
                    "prediction_gross_return": float(session.prediction.iloc[i]),
                    "shares": quantity, "buy_notional": buy_notional,
                    "sell_notional": sell_notional,
                    "transaction_tax_rate": tax_rate,
                    "securities_transaction_tax_rate": security_tax_rate,
                    "rural_special_tax_rate": rural_tax_rate,
                    "cost_paid": invested - buy_notional + sell_notional * sell_cost,
                    "broker_fee_paid": (buy_notional + sell_notional) * BROKER_FEE_PER_SIDE,
                    "slippage_paid": (buy_notional + sell_notional) * SLIPPAGE_PER_SIDE * slippage_multiplier,
                    "securities_transaction_tax_paid": sell_notional * security_tax_rate,
                    "rural_special_tax_paid": sell_notional * rural_tax_rate,
                    "net_pnl": proceeds - invested,
                    "net_bps": (proceeds / invested - 1) * 10000,
                    "reason": "same_day_final_complete_bar_close",
                }
                break

        balances[code] = cash
        peaks[code] = max(peaks[code], cash)
        worst_dd[code] = min(worst_dd[code], cash / max(peaks[code], 1e-12) - 1)
        daily.append({"code": code, "date": date, "model_return": cash / cash_before - 1,
                      "trade_count": int(trade is not None)})
        if trade:
            trades.append(trade)

        # Fixed matched benchmark: buy at 09:30 open, sell at the same final close.
        baseline_cash = baseline_balances[code]
        entry_rows = session.index[session.bar_start == 570]
        if int(last.bar_start) == 915 and len(entry_rows):
            sell_cost = BROKER_FEE_PER_SIDE + SLIPPAGE_PER_SIDE * slippage_multiplier + tax_rate
            entry_i = int(entry_rows[0])
            cap_i = max(entry_i - 1, 0)
            entry_price = float(session.open.iloc[entry_i])
            exit_price = float(last.close)
            quantity = int(min(
                baseline_cash * ALLOCATION / (entry_price * (1 + buy_cost)),
                max(int(session.capacity_shares.iloc[cap_i]), 0),
            )) if entry_price > 0 else 0
            baseline_balances[code] += (
                quantity * exit_price * (1 - sell_cost)
                - quantity * entry_price * (1 + buy_cost)
            )

    model_frame = pd.DataFrame([{"code": k, "model_return_pct": (v / INITIAL_BALANCE - 1) * 100,
                                 "baseline_return_pct": (baseline_balances[k] / INITIAL_BALANCE - 1) * 100,
                                 "max_drawdown_pct": worst_dd[k] * 100}
                                for k, v in balances.items()])
    trade_frame, daily_frame = pd.DataFrame(trades), pd.DataFrame(daily)
    excess = model_frame.model_return_pct - model_frame.baseline_return_pct
    daily_returns = daily_frame.model_return.to_numpy(dtype=float)
    summary = {
        "stocks": int(len(model_frame)), "trades": int(len(trade_frame)),
        "active_stock_days": int(daily_frame.trade_count.sum()),
        "mean_return_pct": float(model_frame.model_return_pct.mean()),
        "median_return_pct": float(model_frame.model_return_pct.median()),
        "mean_baseline_return_pct": float(model_frame.baseline_return_pct.mean()),
        "mean_excess_baseline_pct": float(excess.mean()),
        "positive_excess_stock_fraction": float((excess > 0).mean()),
        "mean_drawdown_pct": float(model_frame.max_drawdown_pct.mean()),
        "worst_drawdown_pct": float(model_frame.max_drawdown_pct.min()),
        "daily_sharpe": float(daily_returns.mean() / (daily_returns.std(ddof=0) + 1e-12) * np.sqrt(252)) if len(daily_returns) > 1 else 0.0,
        "win_rate": float((trade_frame.net_pnl > 0).mean()) if len(trade_frame) else 0.0,
        "mean_trade_bps": float(trade_frame.net_bps.mean()) if len(trade_frame) else 0.0,
        "median_trade_bps": float(trade_frame.net_bps.median()) if len(trade_frame) else 0.0,
        "mean_holding_minutes": float(trade_frame.holding_minutes.mean()) if len(trade_frame) else 0.0,
        "total_cost_paid": float(trade_frame.cost_paid.sum()) if len(trade_frame) else 0.0,
        "broker_fee_paid": float(trade_frame.broker_fee_paid.sum()) if len(trade_frame) else 0.0,
        "slippage_paid": float(trade_frame.slippage_paid.sum()) if len(trade_frame) else 0.0,
        "securities_transaction_tax_paid": float(trade_frame.securities_transaction_tax_paid.sum()) if len(trade_frame) else 0.0,
        "rural_special_tax_paid": float(trade_frame.rural_special_tax_paid.sum()) if len(trade_frame) else 0.0,
        "truncated_exits": 0,
    }
    if detail:
        return summary, trade_frame, daily_frame, model_frame
    return summary


def _candidates(experiments: list[dict]) -> list[dict]:
    rows = []
    for model_name in (*EOD_MODEL_SPECS.keys(), *RULE_NAMES):
        if model_name in RULE_NAMES:
            thresholds = (None,)
        elif EOD_MODEL_SPECS[model_name]["type"].endswith("classifier"):
            thresholds = CLASS_THRESHOLDS
        else:
            thresholds = THRESHOLDS
        modes = (("rule", None, None),) if model_name in RULE_NAMES else (
            (("absolute", None, None),)
            + tuple(("rank", r, None) for r in RANK_CUTOFFS)
            + tuple(("time_window", None, name) for name in ENTRY_WINDOWS)
        )
        for threshold in thresholds:
          for mode, rank_cutoff, window_name in modes:
            folds = [x for x in experiments if x["model_spec"] == model_name
                     and x["entry_threshold"] == threshold and x.get("signal_mode", "absolute") == mode
                     and x.get("rank_cutoff") == rank_cutoff and x.get("time_window") == window_name]
            if len(folds) != len(FOLDS):
                continue
            metrics = [x["metrics"] for x in folds]
            excess = np.array([x["mean_excess_baseline_pct"] for x in metrics])
            model_returns = np.array([x["mean_return_pct"] for x in metrics])
            trade_edge = np.array([x["mean_trade_bps"] for x in metrics])
            pooled_trade_edge = float(np.average(
                trade_edge, weights=[max(x["trades"], 1) for x in metrics]
            ))
            enough = all(x["trades"] >= MIN_FOLD_TRADES for x in metrics)
            active_enough = sum(x["trades"] > 0 for x in metrics) >= 6
            score = float(excess.mean() - 0.5 * excess.std(ddof=0))
            rows.append({
                "model_spec": model_name, "entry_threshold": threshold, "folds": folds,
                "signal_mode": mode, "rank_cutoff": rank_cutoff, "time_window": window_name,
                "score_mean_excess_minus_half_std_pct": score,
                "mean_excess_baseline_pct": float(excess.mean()),
                "positive_excess_folds": int((excess > 0).sum()),
                "positive_return_folds": int((model_returns > 0).sum()),
                "positive_trade_edge_folds": int((trade_edge > 0).sum()),
                "pooled_mean_trade_bps": pooled_trade_edge,
                "eligible": bool(enough and active_enough and pooled_trade_edge > 0
                                 and (trade_edge > 0).sum() >= 6),
            })
    return rows


def _rule_signal(data: pd.DataFrame, name: str) -> np.ndarray:
    if name == "rule_vwap_trend":
        condition = ((data.vwap_gap > 0) & (data.vwap_slope_15m > 0)
                     & (data.ret_15m > 0) & (data.market_ret_15m > 0))
    elif name == "rule_opening_breakout":
        condition = ((data.or15_high_gap > 0) & (data.ret_15m > 0)
                     & (data.same_time_volume > 1.0))
    elif name == "rule_relative_strength":
        condition = ((data.relative_ret_15m > 0) & (data.relative_momentum_rank > 0.25)
                     & (data.market_ret_15m > 0))
    else:
        raise ValueError(f"unknown rule strategy: {name}")
    return condition.fillna(False).to_numpy(dtype=bool)


def run(output: str, features_cache: str, *, resume: bool = False,
        fold_start: int = 1, fold_end: int = len(FOLDS),
        tax_policy: str = "current", experiment_id: str | None = None) -> dict:
    if not (1 <= fold_start <= fold_end <= len(FOLDS)):
        raise ValueError(f"fold 범위가 올바르지 않습니다: {fold_start}..{fold_end}")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=resume)
    cache = out / "eod_features.pkl"
    if resume and cache.exists():
        with cache.open("rb") as stream:
            data = pickle.load(stream)
    else:
        with Path(features_cache).open("rb") as stream:
            base = pickle.load(stream)
        data = add_eod_features(base)
        with cache.open("wb") as stream:
            pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    date_text = data.date.astype(str)
    sessions = data.groupby(["code", "date"], sort=False).bar_start.agg(bars="size", last="max")
    target = add_eod_target(data)
    classification_target = (target > _round_trip_costs(data.date, tax_policy)).where(target.notna())
    quality = {
        "five_minute_rows": int(len(data)), "date_min": str(date_text.min()),
        "date_max": str(date_text.max()), "stocks": int(data.code.nunique()),
        "stock_sessions": int(len(sessions)),
        "median_complete_bars_per_session": float(sessions.bars.median()),
        "sessions_ending_at_1515_fraction": float((sessions["last"] == 915).mean()),
        "eod_target_rows": int(target.notna().sum()),
        "latest_complete_bar_start": "15:15",
    }
    checkpoint = out / "experiments.json"
    experiments = json.loads(checkpoint.read_text(encoding="utf-8")) if resume and checkpoint.exists() else []
    done = {(x["fold"], x["model_spec"]) for x in experiments}
    for fold_no, (train_end, val_start, val_end) in enumerate(FOLDS, 1):
        if not fold_start <= fold_no <= fold_end:
            continue
        train_mask = date_text <= train_end
        val = data.loc[date_text.between(val_start, val_end)].copy().reset_index(drop=True)
        train_indices = np.flatnonzero(train_mask.to_numpy() & target.notna().to_numpy())
        if len(train_indices) > MAX_TRAIN_ROWS:
            rng = np.random.default_rng(20260925 + fold_no)
            train_indices = np.sort(rng.choice(train_indices, MAX_TRAIN_ROWS, replace=False))
        train_features = data.iloc[train_indices][EOD_FEATURES].astype(np.float32).reset_index(drop=True)
        regression_train_target = target.iloc[train_indices].reset_index(drop=True)
        classifier_train_target = classification_target.iloc[train_indices].reset_index(drop=True)
        for model_name in EOD_MODEL_SPECS:
            if (fold_no, model_name) in done:
                continue
            is_classifier = EOD_MODEL_SPECS[model_name]["type"].endswith("classifier")
            train_target = classifier_train_target if is_classifier else regression_train_target
            model = _fit_eod(train_features, train_target, model_name)
            pred = _predict_scores(model, val, model_name)
            thresholds = CLASS_THRESHOLDS if is_classifier else THRESHOLDS
            for threshold in thresholds:
                metrics = simulate_eod(val, pred, threshold, tax_policy=tax_policy)
                experiments.append({
                    "fold": fold_no, "train_end": train_end,
                    "validation_start": val_start, "validation_end": val_end,
                    "training_rows": int(len(train_target)),
                    "model_spec": model_name, "entry_threshold": threshold,
                    "signal_mode": "absolute", "rank_cutoff": None,
                    "time_window": None,
                    "metrics": metrics,
                })
                ranks = pd.Series(pred).groupby([val.date.astype(str), val.bar_start], sort=False).rank(pct=True).to_numpy()
                for rank_cutoff in RANK_CUTOFFS:
                    rank_mask = (ranks >= rank_cutoff) & (pred > threshold)
                    rank_metrics = simulate_eod(val, pred, threshold, signal_mask=rank_mask,
                                                tax_policy=tax_policy)
                    experiments.append({
                        "fold": fold_no, "train_end": train_end,
                        "validation_start": val_start, "validation_end": val_end,
                        "training_rows": int(len(train_target)),
                        "model_spec": model_name, "entry_threshold": threshold,
                        "signal_mode": "rank", "rank_cutoff": rank_cutoff,
                        "time_window": None,
                        "metrics": rank_metrics,
                    })
                for window_name, time_window in ENTRY_WINDOWS.items():
                    window_metrics = simulate_eod(val, pred, threshold, time_window=time_window,
                                                  tax_policy=tax_policy)
                    experiments.append({
                        "fold": fold_no, "train_end": train_end,
                        "validation_start": val_start, "validation_end": val_end,
                        "training_rows": int(len(train_target)),
                        "model_spec": model_name, "entry_threshold": threshold,
                        "signal_mode": "time_window", "rank_cutoff": None,
                        "time_window": window_name, "metrics": window_metrics,
                    })
            temp = checkpoint.with_suffix(".json.tmp")
            temp.write_text(json.dumps(experiments, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temp.replace(checkpoint)
            done.add((fold_no, model_name))
            print(f"fold {fold_no}/{len(FOLDS)} {model_name} EOD done", flush=True)
            del model, pred, metrics
            _trim_memory()
        for rule_name in RULE_NAMES:
            if (fold_no, rule_name) in done:
                continue
            rule_mask = _rule_signal(val, rule_name)
            metrics = simulate_eod(val, np.zeros(len(val)), 0.0, signal_mask=rule_mask,
                                   tax_policy=tax_policy)
            experiments.append({
                "fold": fold_no, "train_end": train_end,
                "validation_start": val_start, "validation_end": val_end,
                "model_spec": rule_name, "entry_threshold": None,
                "signal_mode": "rule", "rank_cutoff": None, "time_window": None,
                "metrics": metrics,
            })
            temp = checkpoint.with_suffix(".json.tmp")
            temp.write_text(json.dumps(experiments, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temp.replace(checkpoint)
            done.add((fold_no, rule_name))
            print(f"fold {fold_no}/{len(FOLDS)} {rule_name} EOD done", flush=True)
            del rule_mask, metrics
            _trim_memory()
        del val, train_features, regression_train_target, classifier_train_target
        _trim_memory()

    candidates = _candidates(experiments)
    passed = [x for x in candidates if x["eligible"] and x["mean_excess_baseline_pct"] > 0
              and x["positive_excess_folds"] >= 6 and x["positive_return_folds"] >= 6
              and x["score_mean_excess_minus_half_std_pct"] > 0]
    selected = max(passed, key=lambda x: x["score_mean_excess_minus_half_std_pct"]) if passed else None
    report = {
        "model_name": "stock-single-intraday-v1",
        "experiment_id": experiment_id,
        "official_version_increment": False,
        "product_family": "stock-single-intraday",
        "distinction": "5-minute decisions; one same-day trade per stock; hold to final complete intraday bar close",
        "data_period": {"start": quality["date_min"], "end": quality["date_max"]},
        "data_quality": quality, "feature_names": EOD_FEATURES,
        "model_specs": EOD_MODEL_SPECS,
        "rule_baselines": {
            "rule_vwap_trend": "price above session VWAP, rising VWAP, positive 15-minute stock and market returns",
            "rule_opening_breakout": "above opening 15-minute high, positive 15-minute return and relative volume above 1",
            "rule_relative_strength": "positive market-relative 15-minute return, cross-sectional rank above 0.25 and positive market return",
        },
        "target": "regress same-day gross return, or classify whether return exceeds date-specific round-trip costs",
        "classifier_label": ("gross EOD return >29bp under current tax rates for every historical row"
                             if tax_policy == "current" else
                             "gross EOD return >24bp through 2025-12-31; >29bp from 2026-01-01"),
        "threshold_semantics": "return thresholds for regressors; predicted positive-net-return probability for classifiers",
        "entry_time_windows": {name: {"bar_start_minute_inclusive": limits[0],
                                      "bar_start_minute_exclusive": limits[1]}
                               for name, limits in ENTRY_WINDOWS.items()},
        "entry_cutoff": "signal bar start <= 14:45; next-bar entry at 14:50 or earlier",
        "validation_folds": FOLDS,
        "max_training_rows_per_fold": MAX_TRAIN_ROWS,
        "selection_gate": "at least 30 trades in every fold; pooled net trade expectancy positive and positive mean trade bps in at least 6/8 folds; positive cumulative model return and benchmark excess in at least 6/8 folds; positive excess minus half standard deviation",
        "cost_assumptions": {
            "tax_policy": tax_policy,
            "broker_fee_per_side": BROKER_FEE_PER_SIDE,
            "broker_fee_round_trip": 2 * BROKER_FEE_PER_SIDE,
            "broker_fee_note": "assumed 1.5bp per side; replace with the account's actual rate if different",
            "slippage_per_side_base": SLIPPAGE_PER_SIDE,
            "slippage_round_trip_base": 2 * SLIPPAGE_PER_SIDE,
            "sell_tax_components": ({"securities_transaction_tax": 0.0005,
                                     "rural_special_tax": 0.0015, "total_sell_tax": 0.0020}
                                    if tax_policy == "current" else {
                                        "through_2025_12_31": {"securities_transaction_tax": 0.0,
                                                                "rural_special_tax": 0.0015, "total_sell_tax": 0.0015},
                                        "from_2026_01_01": {"securities_transaction_tax": 0.0005,
                                                             "rural_special_tax": 0.0015, "total_sell_tax": 0.0020},
                                    }),
            "round_trip_total": (0.0029 if tax_policy == "current" else
                                 {"through_2025_12_31": 0.0024, "from_2026_01_01": 0.0029}),
            "stress_multipliers_apply_to": "slippage only; statutory taxes and broker fees remain unchanged",
            "stress_slippage_multipliers": [1.0, 1.5, 2.0],
            "limitations": "slippage is a fixed 3bp-per-side proxy; quoted spread and realized fill delay are not measured separately",
        },
        "execution_assumptions": {"allocation": ALLOCATION, "volume_participation": PARTICIPATION,
                                  "max_trades_per_stock_day": 1,
                                  "exit": "last complete 5-minute bar close from the 15:19 minute; bar starts 15:15"},
        "universe_limitation": "fixed 2026 major-stock list applied retrospectively; point-in-time membership unavailable",
        "selection_status": "validation_quality_gate_passed" if selected else "no_candidate_passed_validation_quality_gate",
        "selected": selected, "candidates": candidates,
        "retrospective_audit_period": {"start": RETROSPECTIVE_AUDIT[0], "end": RETROSPECTIVE_AUDIT[1],
                                        "status": "previously examined for prior intraday v1; not independent"},
        "deployment_approved": False,
    }
    (out / "experiments.json").write_text(json.dumps(experiments, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({
        "model_name": report["model_name"], "product_family": report["product_family"],
        "experiment_id": experiment_id,
        "official_version_increment": False,
        "data_period": report["data_period"], "feature_names": EOD_FEATURES,
        "target": report["target"], "validation_folds": FOLDS,
        "cost_assumptions": report["cost_assumptions"],
        "execution_assumptions": report["execution_assumptions"],
        "universe_limitation": report["universe_limitation"],
        "selection_gate": report["selection_gate"],
        "selection_status": report["selection_status"], "selected": selected,
        "deployment_approved": False,
    }, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    if selected:
        audit = data.loc[date_text.between(*RETROSPECTIVE_AUDIT)].copy().reset_index(drop=True)
        is_rule = selected["model_spec"] in RULE_NAMES
        final = None
        if is_rule:
            audit_pred = np.zeros(len(audit))
            audit_mask = _rule_signal(audit, selected["model_spec"])
        else:
            mask = date_text <= FINAL_TRAIN_END
            is_classifier = EOD_MODEL_SPECS[selected["model_spec"]]["type"].endswith("classifier")
            final_target = classification_target if is_classifier else target
            final = _fit_eod(data.loc[mask, EOD_FEATURES].reset_index(drop=True),
                             final_target.loc[mask].reset_index(drop=True), selected["model_spec"])
            audit_pred = _predict_scores(final, audit, selected["model_spec"])
            if selected.get("signal_mode") == "rank":
                ranks = pd.Series(audit_pred).groupby(
                    [audit.date.astype(str), audit.bar_start], sort=False
                ).rank(pct=True).to_numpy()
                audit_mask = (ranks >= selected["rank_cutoff"]) & (audit_pred > selected["entry_threshold"])
            else:
                audit_mask = None
        audit_results = {}
        for multiplier in (1.0, 1.5, 2.0):
            label = "base" if multiplier == 1 else f"{multiplier:g}x_slippage"
            result = simulate_eod(audit, audit_pred, selected["entry_threshold"],
                                  signal_mask=audit_mask,
                                  time_window=ENTRY_WINDOWS.get(selected.get("time_window")),
                                  slippage_multiplier=multiplier,
                                  tax_policy=tax_policy,
                                  detail=(multiplier == 1))
            if multiplier == 1:
                metrics, trades, daily, stocks = result
                audit_results[label] = metrics
                trades.to_csv(out / "retrospective_trades.csv", index=False)
                daily.to_csv(out / "retrospective_daily.csv", index=False)
                stocks.to_csv(out / "retrospective_by_stock.csv", index=False)
            else:
                audit_results[label] = result
        report["retrospective_audit"] = audit_results
        if final is not None:
            joblib.dump({"model": final, "feature_names": EOD_FEATURES,
                         "entry_threshold": selected["entry_threshold"],
                         "train_end_date": FINAL_TRAIN_END, "deployment_approved": False},
                        out / "diagnostic_candidate.joblib", compress=3)
        (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--features-cache", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fold-start", type=int, default=1)
    parser.add_argument("--fold-end", type=int, default=len(FOLDS))
    parser.add_argument("--tax-policy", choices=("current", "historical"), default="current")
    parser.add_argument("--experiment-id")
    args = parser.parse_args()
    report = run(args.output, args.features_cache, resume=args.resume,
                 fold_start=args.fold_start, fold_end=args.fold_end,
                 tax_policy=args.tax_policy, experiment_id=args.experiment_id)
    print(json.dumps({"selection_status": report["selection_status"],
                      "selected": report["selected"], "deployment_approved": False},
                     ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
