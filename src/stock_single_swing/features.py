"""Causal price features for stock single-swing research models."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_price_features(env: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Create price trend and risk features known by each row's close.

    Input rows must be grouped by stock and sorted by date, matching the stock
    swing dataset builder. No forward fill across securities or future values
    are used.
    """
    required = {"stock_code", "close", "high", "low"}
    missing = sorted(required - set(env.columns))
    if missing:
        raise ValueError(f"price feature 필수 컬럼 누락: {missing}")

    close = pd.to_numeric(env["close"], errors="coerce").astype(float)
    high = pd.to_numeric(env["high"], errors="coerce").astype(float)
    low = pd.to_numeric(env["low"], errors="coerce").astype(float)
    codes = env["stock_code"].astype(str).to_numpy()
    group = close.groupby(codes, sort=False)
    daily_return = group.pct_change().replace([np.inf, -np.inf], np.nan)

    features: dict[str, pd.Series | np.ndarray] = {
        f"stock_momentum_{days}d": group.pct_change(days)
        for days in (5, 20, 60, 120)
    }
    for days in (20, 60):
        rolling_vol = daily_return.groupby(codes, sort=False).rolling(
            days, min_periods=max(5, days // 2)
        ).std()
        features[f"stock_volatility_{days}d"] = rolling_vol.reset_index(
            level=0, drop=True
        ).reindex(env.index)
        rolling_high = close.groupby(codes, sort=False).transform(
            lambda values: values.rolling(days, min_periods=max(5, days // 2)).max()
        )
        features[f"stock_drawdown_{days}d"] = close / rolling_high - 1.0
    features["stock_intraday_range"] = (high - low) / close.replace(0, np.nan)

    frame = pd.DataFrame(features, index=env.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return frame.to_numpy(dtype=np.float32), list(frame.columns)
