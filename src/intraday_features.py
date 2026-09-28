"""ETF 1분봉 feature builder.

The existing ``feature_vector`` module is daily-bar oriented.  This module
keeps the intraday timing contract explicit:

* every feature at bar ``t`` uses data available at the close of bar ``t``;
* the trainer must use that row to trade at bar ``t+1`` (the current trading
  environments already read ``training_data[tick - 1]`` for this purpose);
* rolling indicators reset at the Korean cash-session boundary instead of
  treating the overnight gap as a one-minute return.

The output layout intentionally mirrors the trainer's existing datasets while
also writing identifiers and split metadata needed for minute-level research.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sqlalchemy import bindparam, text

from quantylab.common.db import psql

try:
    from .target_etfs import TARGET_ETFS
except ImportError:  # pragma: no cover - direct script execution fallback
    from target_etfs import TARGET_ETFS


SESSION_OPEN_MINUTE = 9 * 60
SESSION_CLOSE_MINUTE = 15 * 60 + 30
ROLLING_WINDOWS = (5, 15, 30, 60, 120)

FEATURE_COLUMNS = [
    # Bar/return structure
    "intra_ret_1", "intra_ret_3", "intra_ret_5", "intra_ret_10",
    "intra_ret_15", "intra_ret_30", "intra_ret_60",
    "overnight_gap", "range_pct", "body_pct", "upper_wick_pct",
    "lower_wick_pct", "close_location",
    # Trend and mean reversion
    "close_to_sma_5", "close_to_sma_15", "close_to_sma_30",
    "close_to_sma_60", "close_to_sma_120", "close_to_ema_12",
    "close_to_ema_26", "ema_spread_12_26", "rsi_14", "rsi_30",
    "bb_width_20", "bb_position_20", "price_position_60",
    # Risk/volatility
    "volatility_5", "volatility_15", "volatility_30", "volatility_60",
    "atr_pct_14", "atr_pct_30",
    # Volume and flow
    "log_volume", "volume_ratio_20", "volume_ratio_60", "volume_z_60",
    "obv_z_60", "signed_volume_ratio_20",
    # Session context
    "session_return", "session_high_gap", "session_low_gap",
    "session_vwap_gap", "minutes_from_open", "minutes_sin", "minutes_cos",
    # Cross-sectional context
    "xsec_rel_ret_1", "xsec_rel_ret_5", "xsec_mean_ret_1",
    "xsec_dispersion_1", "xsec_up_ratio_1",
    # Completed-session context (computed only from prior sessions)
    "prior_day_ret_1", "prior_day_ret_3", "prior_day_ret_5",
    "prior_day_volatility_5", "prior_day_range_5", "prior_day_volume_ratio_20",
    "prior_day_close_to_sma_20",
    # Previous-session AI ETF analysis (as-of only; no same-day leakage)
    "ai_score", "ai_opinion", "ai_confidence", "ai_available", "ai_age_days",
    "ai_factor_theme", "ai_factor_portfolio", "ai_factor_stability",
    "ai_factor_technical", "ai_factor_supply_demand",
    "ai_upside_gap", "ai_downside_gap", "ai_score_delta", "ai_opinion_delta",
    # Previous-session AI KOSPI/KOSDAQ market regime
    "market_kospi_score", "market_kosdaq_score", "market_direction_mean",
    "market_risk_on", "market_available", "market_age_days",
]

AI_FEATURE_COLUMNS = [
    "ai_score", "ai_opinion", "ai_confidence", "ai_available", "ai_age_days",
    "ai_factor_theme", "ai_factor_portfolio", "ai_factor_stability",
    "ai_factor_technical", "ai_factor_supply_demand", "ai_upside_gap",
    "ai_downside_gap", "ai_score_delta", "ai_opinion_delta",
]
MARKET_FEATURE_COLUMNS = [
    "market_kospi_score", "market_kosdaq_score", "market_direction_mean",
    "market_risk_on", "market_available", "market_age_days",
]
CORE_FEATURE_COLUMNS = [c for c in FEATURE_COLUMNS
                        if c not in AI_FEATURE_COLUMNS + MARKET_FEATURE_COLUMNS]


def _as_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float)


def load_minute_candles(
    start_date: str,
    end_date: str,
    codes: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Load valid target ETF minute candles from the shared PostgreSQL DB."""
    source_codes = TARGET_ETFS if codes is None else codes
    target_codes = [str(c).zfill(6) for c in source_codes]
    query = text(
        """
        SELECT code, date, time, open, high, low, close, volume,
               diff, diff_ratio, amount
        FROM stock_minute_candle
        WHERE code IN :codes
          AND date BETWEEN :start_date AND :end_date
          AND date ~ '^[0-9]{8}$'
          AND time ~ '^[0-9]{4}$'
          AND time BETWEEN '0900' AND '1530'
          AND open > 0 AND high > 0 AND low > 0 AND close > 0
          AND volume >= 0
        ORDER BY code, date, time
        """
    ).bindparams(bindparam("codes", expanding=True))

    with psql.get_session() as session:
        rows = session.execute(
            query,
            {"codes": target_codes, "start_date": start_date, "end_date": end_date},
        ).mappings().all()

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for column in ("open", "high", "low", "close", "volume", "diff", "diff_ratio", "amount"):
        df[column] = _as_float(df[column])
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["date"] = df["date"].astype(str)
    df["time"] = df["time"].astype(str).str.zfill(4)
    return df.sort_values(["code", "date", "time"]).reset_index(drop=True)


def load_previous_ai_etf_analysis(start_date: str, end_date: str,
                                  codes: Iterable[str] | None = None) -> pd.DataFrame:
    """Load numeric ETF AI analysis records for causal as-of joins.

    The caller joins with ``analysis_date < candle_date``.  Keeping this
    query separate makes the information boundary explicit and avoids using
    the analysis generated after a session has started.
    """
    source_codes = TARGET_ETFS if codes is None else codes
    target_codes = [str(c).zfill(6) for c in source_codes]
    query = text(
        """
        SELECT code, date, investment_score, investment_opinion,
               score_confidence, factor_scores, upside, downside
        FROM ai_etf_analysis
        WHERE code IN :codes
          AND create_time IS NOT NULL AND update_time IS NOT NULL
          AND GREATEST(create_time, update_time) <
              (to_date(date, 'YYYYMMDD') + interval '1 day 9 hours') AT TIME ZONE 'Asia/Seoul'
          AND date >= :start_date
          AND date <= :end_date
          AND date ~ '^[0-9]{8}$'
        ORDER BY code, date
        """
    ).bindparams(bindparam("codes", expanding=True))
    with psql.get_session() as session:
        rows = session.execute(query, {
            "codes": target_codes, "start_date": start_date, "end_date": end_date,
        }).mappings().all()
    if not rows:
        return pd.DataFrame(columns=["code", "date", "investment_score",
                                     "investment_opinion", "score_confidence",
                                     "factor_scores", "upside", "downside"])
    result = pd.DataFrame(rows)
    result["code"] = result["code"].astype(str).str.zfill(6)
    result["date"] = result["date"].astype(str)
    return result


def _add_previous_ai_features(features: pd.DataFrame, start_date: str,
                              end_date: str) -> pd.DataFrame:
    """Attach the latest AI result strictly before each session date."""
    ai = load_previous_ai_etf_analysis(start_date, end_date,
                                       features["code"].unique())
    output = features.copy()
    if ai.empty:
        for column in AI_FEATURE_COLUMNS:
            output[column] = 0.0
        return output

    factor_names = ("theme", "portfolio", "stability", "technical", "supply_demand")
    for name in factor_names:
        ai[f"ai_factor_{name}"] = ai["factor_scores"].map(
            lambda value: float(value.get(name, 0)) if isinstance(value, dict) else 0.0
        )
    ai = ai.rename(columns={"date": "analysis_date"})
    for column in ("investment_score", "investment_opinion", "score_confidence"):
        ai[column] = pd.to_numeric(ai[column], errors="coerce")
    ai["upside"] = pd.to_numeric(ai["upside"], errors="coerce")
    ai["downside"] = pd.to_numeric(ai["downside"], errors="coerce")
    ai["investment_score"] = ai["investment_score"].fillna(50.0) / 100.0
    ai["investment_opinion"] = ai["investment_opinion"].fillna(0.0) / 2.0
    ai["score_confidence"] = ai["score_confidence"].fillna(0.0).clip(0, 1)
    ai = ai.sort_values(["code", "analysis_date"])
    ai["ai_score_delta"] = ai.groupby("code")["investment_score"].diff().fillna(0.0)
    ai["ai_opinion_delta"] = ai.groupby("code")["investment_opinion"].diff().fillna(0.0)
    output["_session_date"] = output["date"].astype(str)
    output["_session_ts"] = pd.to_datetime(output["_session_date"], format="%Y%m%d")
    ai["analysis_ts"] = pd.to_datetime(ai["analysis_date"], format="%Y%m%d")
    # pandas merge_asof requires the as-of key itself to be globally sorted;
    # ``by`` still keeps each ETF's history independent.
    output = output.sort_values(["_session_ts", "code"])
    ai = ai.sort_values(["analysis_ts", "code"])
    # merge_asof with allow_exact_matches=False enforces prior-date causality.
    output = pd.merge_asof(
        output, ai.drop(columns=["factor_scores"]),
        left_on="_session_ts", right_on="analysis_ts", by="code",
        direction="backward", allow_exact_matches=False,
    )
    output["ai_age_days"] = (
        output["_session_ts"] - output["analysis_ts"]
    ).dt.days
    output["ai_available"] = output["analysis_date"].notna().astype(float)
    output["ai_age_days"] = output["ai_age_days"].fillna(999).clip(0, 999) / 30.0
    output["ai_score"] = output["investment_score"].fillna(0.5)
    output["ai_opinion"] = output["investment_opinion"].fillna(0.0)
    output["ai_confidence"] = output["score_confidence"].fillna(0.0)
    current_close = output["close"].astype(float).clip(lower=1.0)
    output["ai_upside_gap"] = (output["upside"] / current_close - 1.0).clip(-1.0, 1.0).fillna(0.0)
    output["ai_downside_gap"] = (output["downside"] / current_close - 1.0).clip(-1.0, 1.0).fillna(0.0)
    output["ai_score_delta"] = output["ai_score_delta"].fillna(0.0)
    output["ai_opinion_delta"] = output["ai_opinion_delta"].fillna(0.0)
    for name in factor_names:
        output[f"ai_factor_{name}"] = output[f"ai_factor_{name}"].fillna(0.0) / 2.0
    return output.drop(columns=["_session_date", "_session_ts", "analysis_date", "analysis_ts", "investment_score",
                               "investment_opinion", "score_confidence", "upside", "downside"])


def load_previous_ai_market_analysis(start_date: str, end_date: str) -> pd.DataFrame:
    """Load numeric KOSPI/KOSDAQ AI regime analyses."""
    query = text(
        """
        SELECT market, date, market_score, market_direction
        FROM ai_stock_market_analysis
        WHERE market IN ('kospi', 'kosdaq')
          AND create_time IS NOT NULL AND update_time IS NOT NULL
          AND GREATEST(create_time, update_time) <
              (to_date(date, 'YYYYMMDD') + interval '1 day 9 hours') AT TIME ZONE 'Asia/Seoul'
          AND date >= :start_date AND date <= :end_date
          AND date ~ '^[0-9]{8}$'
        ORDER BY date, market
        """
    )
    with psql.get_session() as session:
        rows = session.execute(query, {"start_date": start_date, "end_date": end_date}).mappings().all()
    if not rows:
        return pd.DataFrame(columns=["analysis_date", "market_kospi_score",
                                     "market_kosdaq_score", "market_direction_mean"])
    raw = pd.DataFrame(rows)
    raw["date"] = raw["date"].astype(str)
    raw["market_score"] = pd.to_numeric(raw["market_score"], errors="coerce").fillna(50.0) / 100.0
    raw["market_direction"] = pd.to_numeric(raw["market_direction"], errors="coerce").fillna(0.0) / 2.0
    scores = raw.pivot_table(index="date", columns="market", values="market_score", aggfunc="last")
    directions = raw.groupby("date")["market_direction"].mean()
    result = scores.join(directions.rename("market_direction_mean")).reset_index()
    result = result.rename(columns={"date": "analysis_date",
                                    "kospi": "market_kospi_score",
                                    "kosdaq": "market_kosdaq_score"})
    for column in ("market_kospi_score", "market_kosdaq_score"):
        if column not in result:
            result[column] = np.nan
    return result[["analysis_date", "market_kospi_score", "market_kosdaq_score",
                   "market_direction_mean"]].sort_values("analysis_date")


def _add_previous_market_features(features: pd.DataFrame, start_date: str,
                                  end_date: str) -> pd.DataFrame:
    """Broadcast latest prior market AI analysis to all ETF bars."""
    market = load_previous_ai_market_analysis(start_date, end_date)
    output = features.copy()
    if market.empty:
        for column in MARKET_FEATURE_COLUMNS:
            output[column] = 0.0
        return output
    output["_session_date"] = output["date"].astype(str)
    output["_session_ts"] = pd.to_datetime(output["_session_date"], format="%Y%m%d")
    market["analysis_ts"] = pd.to_datetime(market["analysis_date"], format="%Y%m%d")
    output = output.sort_values(["_session_ts", "code"])
    market = market.sort_values("analysis_ts")
    output = pd.merge_asof(output, market.drop(columns=["analysis_date"]),
                           left_on="_session_ts", right_on="analysis_ts",
                           direction="backward", allow_exact_matches=False)
    output["market_available"] = output["analysis_ts"].notna().astype(float)
    output["market_age_days"] = (output["_session_ts"] - output["analysis_ts"]).dt.days
    output["market_age_days"] = output["market_age_days"].fillna(999).clip(0, 999) / 30.0
    output["market_kospi_score"] = output["market_kospi_score"].fillna(.5)
    output["market_kosdaq_score"] = output["market_kosdaq_score"].fillna(.5)
    output["market_direction_mean"] = output["market_direction_mean"].fillna(0.0)
    output["market_risk_on"] = ((output["market_kospi_score"] + output["market_kosdaq_score"]) / 2.0)
    return output.drop(columns=["_session_date", "_session_ts", "analysis_ts"])


def _rolling_mean(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=1).mean()


def _rolling_std(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=2).std(ddof=0)


def _build_code_features(code_df: pd.DataFrame) -> pd.DataFrame:
    """Build causal, session-aware features for one ETF."""
    parts: list[pd.DataFrame] = []
    previous_session_close: float | None = None
    session_history: list[dict[str, float]] = []

    for session_date, day in code_df.groupby("date", sort=True):
        day = day.sort_values("time").reset_index(drop=True).copy()
        open_ = day["open"]
        high = day["high"]
        low = day["low"]
        close = day["close"]
        volume = day["volume"].clip(lower=0)
        prev_close = close.shift(1)

        # All of the following indicators are calculated only within the
        # current cash session.  The overnight gap is represented separately.
        ret_1 = close.pct_change()
        day["intra_ret_1"] = ret_1
        for window in (3, 5, 10, 15, 30, 60):
            day[f"intra_ret_{window}"] = close.pct_change(window)

        session_open = float(open_.iloc[0])
        if previous_session_close and previous_session_close > 0:
            overnight_gap = session_open / previous_session_close - 1.0
        else:
            overnight_gap = 0.0
        day["overnight_gap"] = overnight_gap

        # Only completed sessions are allowed here. These features provide
        # multi-day context without carrying an intraday rolling window over
        # the overnight boundary.
        history = pd.DataFrame(session_history)
        if history.empty:
            prior_values = {
                "prior_day_ret_1": 0.0, "prior_day_ret_3": 0.0,
                "prior_day_ret_5": 0.0, "prior_day_volatility_5": 0.0,
                "prior_day_range_5": 0.0, "prior_day_volume_ratio_20": 0.0,
                "prior_day_close_to_sma_20": 0.0,
            }
        else:
            last_close = history.close.iloc[-1]
            close_3 = history.close.iloc[-4] if len(history) >= 4 else last_close
            close_5 = history.close.iloc[-6] if len(history) >= 6 else last_close
            sma20 = history.close.tail(20).mean()
            median_volume20 = history.volume.tail(20).median()
            prior_values = {
                "prior_day_ret_1": history.ret.iloc[-1],
                "prior_day_ret_3": last_close / close_3 - 1.0 if close_3 else 0.0,
                "prior_day_ret_5": last_close / close_5 - 1.0 if close_5 else 0.0,
                "prior_day_volatility_5": history.ret.tail(5).std(ddof=0),
                "prior_day_range_5": history.day_range.tail(5).mean(),
                "prior_day_volume_ratio_20": history.volume.iloc[-1] / median_volume20 - 1.0
                if median_volume20 else 0.0,
                "prior_day_close_to_sma_20": last_close / sma20 - 1.0 if sma20 else 0.0,
            }
        for column, value in prior_values.items():
            day[column] = float(value) if np.isfinite(value) else 0.0

        day["range_pct"] = (high - low) / close.clip(lower=1.0)
        day["body_pct"] = (close - open_) / open_.clip(lower=1.0)
        day["upper_wick_pct"] = (high - pd.concat([open_, close], axis=1).max(axis=1)) / open_.clip(lower=1.0)
        day["lower_wick_pct"] = (pd.concat([open_, close], axis=1).min(axis=1) - low) / open_.clip(lower=1.0)
        day["close_location"] = ((close - low) / (high - low).replace(0, np.nan)).clip(0, 1)

        for window in ROLLING_WINDOWS:
            sma = _rolling_mean(close, window)
            day[f"close_to_sma_{window}"] = close / sma.clip(lower=1.0) - 1.0

        ema12 = close.ewm(span=12, adjust=False, min_periods=1).mean()
        ema26 = close.ewm(span=26, adjust=False, min_periods=1).mean()
        day["close_to_ema_12"] = close / ema12.clip(lower=1.0) - 1.0
        day["close_to_ema_26"] = close / ema26.clip(lower=1.0) - 1.0
        day["ema_spread_12_26"] = (ema12 - ema26) / close.clip(lower=1.0)

        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        for window in (14, 30):
            avg_gain = _rolling_mean(gain, window)
            avg_loss = _rolling_mean(loss, window)
            rs = avg_gain / avg_loss.replace(0, np.nan)
            rsi = (100 - 100 / (1 + rs)).fillna(50.0)
            rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
            day[f"rsi_{window}"] = rsi

        mid = _rolling_mean(close, 20)
        std20 = _rolling_std(close, 20).fillna(0.0)
        upper = mid + 2 * std20
        lower = mid - 2 * std20
        band = (upper - lower).replace(0, np.nan)
        day["bb_width_20"] = (band / mid.clip(lower=1.0)).fillna(0.0)
        day["bb_position_20"] = ((close - lower) / band).clip(0, 1).fillna(0.5)

        rolling_high = high.rolling(60, min_periods=1).max()
        rolling_low = low.rolling(60, min_periods=1).min()
        day["price_position_60"] = ((close - rolling_low) / (rolling_high - rolling_low).replace(0, np.nan)).clip(0, 1)

        for window in (5, 15, 30, 60):
            day[f"volatility_{window}"] = _rolling_std(ret_1, window)

        true_range = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
        ).max(axis=1)
        day["atr_pct_14"] = _rolling_mean(true_range, 14) / close.clip(lower=1.0)
        day["atr_pct_30"] = _rolling_mean(true_range, 30) / close.clip(lower=1.0)

        day["log_volume"] = np.log1p(volume)
        volume_mean20 = _rolling_mean(volume, 20).replace(0, np.nan)
        volume_mean60 = _rolling_mean(volume, 60).replace(0, np.nan)
        day["volume_ratio_20"] = volume / volume_mean20
        day["volume_ratio_60"] = volume / volume_mean60
        volume_std60 = _rolling_std(volume, 60).replace(0, np.nan)
        day["volume_z_60"] = (volume - volume_mean60) / volume_std60

        signed_volume = np.sign(ret_1.fillna(0.0)) * volume
        obv = signed_volume.cumsum()
        obv_mean = _rolling_mean(obv, 60)
        obv_std = _rolling_std(obv, 60).replace(0, np.nan)
        day["obv_z_60"] = (obv - obv_mean) / obv_std
        day["signed_volume_ratio_20"] = _rolling_mean(signed_volume, 20) / volume_mean20

        session_high = high.cummax()
        session_low = low.cummin()
        cum_volume = volume.cumsum()
        cum_amount = (close * volume).cumsum()
        session_vwap = cum_amount / cum_volume.replace(0, np.nan)
        day["session_return"] = close / session_open - 1.0
        day["session_high_gap"] = close / session_high.clip(lower=1.0) - 1.0
        day["session_low_gap"] = close / session_low.clip(lower=1.0) - 1.0
        day["session_vwap_gap"] = close / session_vwap.replace(0, np.nan) - 1.0

        minute_of_day = day["time"].astype(int) // 100 * 60 + day["time"].astype(int) % 100
        elapsed = (minute_of_day - SESSION_OPEN_MINUTE).clip(0, SESSION_CLOSE_MINUTE - SESSION_OPEN_MINUTE)
        day["minutes_from_open"] = elapsed / (SESSION_CLOSE_MINUTE - SESSION_OPEN_MINUTE)
        day["minutes_sin"] = np.sin(2 * np.pi * day["minutes_from_open"])
        day["minutes_cos"] = np.cos(2 * np.pi * day["minutes_from_open"])

        day["_bar_index"] = np.arange(len(day), dtype=np.int16)
        parts.append(day)
        session_history.append({
            "close": float(close.iloc[-1]),
            "ret": float(close.iloc[-1] / session_open - 1.0),
            "day_range": float((high.max() - low.min()) / max(session_open, 1.0)),
            "volume": float(volume.sum()),
        })
        previous_session_close = float(close.iloc[-1])

    return pd.concat(parts, ignore_index=True) if parts else code_df.iloc[0:0].copy()


def build_intraday_features(candles: pd.DataFrame) -> pd.DataFrame:
    """Build per-ETF features and cross-sectional market context."""
    if candles.empty:
        return candles.copy()

    per_code = [
        _build_code_features(group)
        for _, group in candles.groupby("code", sort=True)
    ]
    result = pd.concat(per_code, ignore_index=True)
    result = result.sort_values(["date", "time", "code"]).reset_index(drop=True)

    cross = result.groupby(["date", "time"], sort=False)
    for window in (1, 5):
        column = f"intra_ret_{window}"
        mean = cross[column].transform("mean")
        result[f"xsec_rel_ret_{window}"] = result[column] - mean
        if window == 1:
            result["xsec_mean_ret_1"] = mean
            result["xsec_dispersion_1"] = cross[column].transform("std").fillna(0.0)
            result["xsec_up_ratio_1"] = cross[column].transform(lambda s: (s > 0).mean())

    # AI analyses are daily and often sparse.  Join only the latest result
    # strictly before the session date, then broadcast it to every intraday
    # bar of that ETF/session.
    result = _add_previous_ai_features(
        result, start_date="20240101", end_date=str(result["date"].max())
    )
    result = _add_previous_market_features(
        result, start_date="20240101", end_date=str(result["date"].max())
    )

    numeric = result[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    # Neutral values are preferable to forward filling across ETF/session
    # boundaries.  RSI and band position have meaningful neutral defaults.
    numeric["rsi_14"] = numeric["rsi_14"].fillna(50.0)
    numeric["rsi_30"] = numeric["rsi_30"].fillna(50.0)
    numeric["bb_position_20"] = numeric["bb_position_20"].fillna(0.5)
    numeric["close_location"] = numeric["close_location"].fillna(0.5)
    numeric = numeric.fillna(0.0)
    result[FEATURE_COLUMNS] = numeric.astype(np.float32)
    return result


def _date_splits(dates: list[str]) -> dict[str, str]:
    if len(dates) < 3:
        raise ValueError("최소 3거래일이 필요합니다.")
    train_end = dates[max(0, int(len(dates) * 0.70) - 1)]
    val_end = dates[max(1, int(len(dates) * 0.85) - 1)]
    return {"train_end": train_end, "val_end": val_end}


def _split_name(date: str, split_dates: dict[str, str]) -> str:
    if date <= split_dates["train_end"]:
        return "train"
    if date <= split_dates["val_end"]:
        return "validation"
    return "test"


def build_dataset(
    start_date: str,
    end_date: str,
    output_dir: str,
    codes: Iterable[str] | None = None,
) -> str:
    """Build raw/scaled intraday dataset and fit scaler on train dates only."""
    candles = load_minute_candles(start_date, end_date, codes=codes)
    if candles.empty:
        raise ValueError(f"분봉 데이터가 없습니다: {start_date}~{end_date}")

    features = build_intraday_features(candles)
    dates = sorted(features["date"].unique().tolist())
    split_dates = _date_splits(dates)
    features["split"] = features["date"].map(lambda d: _split_name(d, split_dates))
    # Cross-sectional features were calculated in date/time order above.  The
    # trainer, however, expects each ETF to be contiguous so it can reset the
    # portfolio at ETF boundaries.
    features = features.sort_values(["code", "date", "time"]).reset_index(drop=True)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    identifiers = features[["code", "date", "time", "split"]].copy()
    environment = features[["code", "date", "time", "open", "high", "low", "close", "volume", "amount"]].copy()
    environment = environment.rename(columns={"code": "etf_code"})
    raw_numeric = features[FEATURE_COLUMNS].copy()

    train_mask = features["split"].eq("train")
    scaler = StandardScaler()
    scaler.fit(raw_numeric.loc[train_mask])
    scaled = pd.DataFrame(
        np.clip(scaler.transform(raw_numeric), -5.0, 5.0),
        columns=FEATURE_COLUMNS,
        index=raw_numeric.index,
    ).astype(np.float32)

    environment.to_csv(output / "environment.csv", index=False)
    raw_numeric.to_csv(output / "training_full.csv", index=False)
    raw_numeric.to_csv(output / "training_selected.csv", index=False)
    scaled.to_csv(output / "training_scaled.csv", index=False)
    environment[["etf_code"]].to_csv(output / "etf_codes.csv", index=False)
    identifiers.to_csv(output / "timestamps.csv", index=False)
    joblib.dump(scaler, output / "scaler.pkl")

    metadata = {
        "source": "stock_minute_candle",
        "bar_interval": "1min",
        "feature_timing": "bar_t features are consumed for bar t+1 execution",
        "start_date": start_date,
        "end_date": end_date,
        "rows": int(len(features)),
        "etf_count": int(features["code"].nunique()),
        "trading_days": len(dates),
        "feature_dim": len(FEATURE_COLUMNS),
        "feature_names": FEATURE_COLUMNS,
        "split_dates": split_dates,
        "split_rows": features["split"].value_counts().to_dict(),
        "codes": sorted(features["code"].unique().tolist()),
    }
    with open(output / "dataset_meta.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print(f"분봉 feature dataset 생성 완료: {output}")
    print(f"  ETF/거래일/행: {metadata['etf_count']} / {metadata['trading_days']} / {metadata['rows']:,}")
    print(f"  피처 수      : {metadata['feature_dim']}")
    print(f"  split         : {split_dates}")
    print(f"  학습 데이터   : {output / 'training_scaled.csv'}")
    return str(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="TIGER ETF 1분봉 feature dataset builder")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--codes", nargs="*", default=None)
    args = parser.parse_args()
    build_dataset(args.start_date, args.end_date, args.output_dir, args.codes)


if __name__ == "__main__":
    main()
