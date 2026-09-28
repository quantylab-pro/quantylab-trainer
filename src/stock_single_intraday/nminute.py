"""Five-minute stock bars, causal features, labels and intraday simulation."""

from __future__ import annotations

import numpy as np
import pandas as pd


INTERVAL_MINUTES = 5
FEATURES = [
    "ret_5m", "ret_10m", "ret_15m", "ret_30m", "ret_60m",
    "vol_30m", "vol_60m", "bar_range", "bar_body", "vwap_gap",
    "vwap_slope_15m", "session_return", "elapsed_session", "remaining_session",
    "or15_high_gap", "or15_low_gap", "or30_high_gap", "or30_low_gap",
    "same_time_volume", "same_time_volatility", "same_time_cum_volume",
    "volume_trend", "market_ret_15m", "market_ret_60m",
    "relative_ret_15m", "relative_ret_60m", "relative_momentum_rank",
]


def _hhmm(minute: int) -> str:
    return f"{minute // 60:02d}{minute % 60:02d}"


def aggregate_five_minute(minute_candles: pd.DataFrame) -> pd.DataFrame:
    """Aggregate exact, complete 1-minute buckets into 5-minute bars.

    Incomplete buckets are dropped. Prices are never forward-filled across a
    missing minute. A bar's ``bar_start`` is its executable open time and
    ``minute`` is its final constituent minute.
    """
    required = {"code", "date", "time", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(minute_candles.columns))
    if missing:
        raise ValueError(f"분봉 필수 컬럼 누락: {missing}")
    data = minute_candles.copy()
    data["code"] = data.code.astype(str).str.zfill(6)
    data["date"] = data.date.astype(str)
    time_text = data.time.astype(str).str.zfill(4).str[:4]
    hour = pd.to_numeric(time_text.str[:2], errors="coerce")
    minute = pd.to_numeric(time_text.str[2:], errors="coerce")
    data["minute"] = hour * 60 + minute
    data = data[data.minute.between(540, 920)].copy()
    data = data.sort_values(["code", "date", "minute"]).drop_duplicates(
        ["code", "date", "minute"], keep="last"
    )
    data["bucket"] = ((data.minute.astype(int) - 540) // INTERVAL_MINUTES).astype(int)
    data["bar_start"] = 540 + data.bucket * INTERVAL_MINUTES

    grouped = data.groupby(["code", "date", "bucket"], sort=False)
    bars = grouped.agg(
        bar_start=("bar_start", "first"),
        n_minutes=("minute", "nunique"),
        first_minute=("minute", "min"),
        last_minute=("minute", "max"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).reset_index()
    bars = bars[
        (bars.n_minutes == INTERVAL_MINUTES)
        & (bars.first_minute == bars.bar_start)
        & (bars.last_minute == bars.bar_start + INTERVAL_MINUTES - 1)
    ].copy()
    bars["minute"] = bars.bar_start + INTERVAL_MINUTES - 1
    bars["time"] = bars.minute.map(_hhmm)
    bars = bars.sort_values(["code", "date", "bar_start"]).reset_index(drop=True)
    if bars.empty:
        raise ValueError("완전한 5분봉이 없습니다.")
    return bars


def build_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Build causal 5-minute features; all histories stop at the current bar."""
    parts = []
    for (code, date), session in bars.groupby(["code", "date"], sort=False):
        d = session.sort_values("bar_start").copy()
        # Indicators that span a missing 5-minute bar restart at the gap.
        segment = d.bar_start.diff().ne(INTERVAL_MINUTES).cumsum()
        by_segment = d.groupby(segment, sort=False)
        for periods, name in ((1, "ret_5m"), (2, "ret_10m"), (3, "ret_15m"),
                              (6, "ret_30m"), (12, "ret_60m")):
            prior_close = by_segment.close.shift(periods)
            d[name] = d.close / prior_close - 1.0
        d[["ret_5m", "ret_10m", "ret_15m", "ret_30m", "ret_60m"]] = d[
            ["ret_5m", "ret_10m", "ret_15m", "ret_30m", "ret_60m"]
        ].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        d["vol_30m"] = by_segment.ret_5m.rolling(6, min_periods=3).std().reset_index(
            level=0, drop=True
        )
        d["vol_60m"] = by_segment.ret_5m.rolling(12, min_periods=6).std().reset_index(
            level=0, drop=True
        )
        d["bar_range"] = (d.high - d.low) / d.close.replace(0, np.nan)
        d["bar_body"] = (d.close - d.open) / d.open.replace(0, np.nan)
        cumulative_volume = d.volume.cumsum()
        vwap = (d.close * d.volume).cumsum() / cumulative_volume.replace(0, np.nan)
        d["vwap_gap"] = d.close / vwap - 1.0
        d["vwap_slope_15m"] = vwap / vwap.shift(3) - 1.0
        d["session_return"] = d.close / float(d.open.iloc[0]) - 1.0
        d["elapsed_session"] = (d.bar_start - 540) / 375.0
        d["remaining_session"] = (915 - (d.bar_start + INTERVAL_MINUTES)) / 375.0

        # Opening-range levels are only exposed after all constituent bars close.
        first_six = d.iloc[:6]
        opening_ready = (
            len(first_six) == 6
            and int(first_six.bar_start.iloc[0]) == 540
            and bool(first_six.bar_start.diff().dropna().eq(5).all())
        )
        if opening_ready:
            high15, low15 = float(first_six.high.iloc[:3].max()), float(first_six.low.iloc[:3].min())
            high30, low30 = float(first_six.high.max()), float(first_six.low.min())
            d["or15_high_gap"] = np.where(d.bar_start >= 555, d.close / high15 - 1.0, 0.0)
            d["or15_low_gap"] = np.where(d.bar_start >= 555, d.close / low15 - 1.0, 0.0)
            d["or30_high_gap"] = np.where(d.bar_start >= 570, d.close / high30 - 1.0, 0.0)
            d["or30_low_gap"] = np.where(d.bar_start >= 570, d.close / low30 - 1.0, 0.0)
        else:
            d[["or15_high_gap", "or15_low_gap", "or30_high_gap", "or30_low_gap"]] = 0.0
        d["code"] = code
        d["date"] = date
        parts.append(d)

    data = pd.concat(parts, ignore_index=True).sort_values(
        ["date", "code", "bar_start"]
    ).reset_index(drop=True)
    # Same-time baselines use only preceding sessions for this stock and bar.
    for source, target in (("volume", "same_time_volume"),
                           ("vol_30m", "same_time_volatility"),
                           ("volume", "same_time_cum_volume")):
        prior = data.groupby(["code", "bar_start"], sort=False)[source].transform(
            lambda values: values.shift(1).rolling(20, min_periods=5).median()
        )
        if target == "same_time_cum_volume":
            current = data.groupby(["code", "date"], sort=False).volume.cumsum()
        else:
            current = data[source]
        data[target] = current / prior.replace(0, np.nan)
    data["volume_trend"] = data.ret_15m * data.same_time_volume.clip(0, 10)

    by_time = data.groupby(["date", "bar_start"], sort=False)
    data["market_ret_15m"] = by_time.ret_15m.transform("mean")
    data["market_ret_60m"] = by_time.ret_60m.transform("mean")
    data["relative_ret_15m"] = data.ret_15m - data.market_ret_15m
    data["relative_ret_60m"] = data.ret_60m - data.market_ret_60m
    data["relative_momentum_rank"] = by_time.ret_15m.rank(pct=True) - 0.5
    data[FEATURES] = data[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    data["capacity_shares"] = data.groupby(["code", "date"], sort=False).volume.transform(
        lambda values: values.rolling(6, min_periods=6).median() * 0.01
    ).fillna(0.0)
    return data


def add_forward_target(bars: pd.DataFrame, horizon_minutes: int) -> pd.Series:
    """Gross open-to-open return after a next-5-minute-open entry."""
    if horizon_minutes not in (15, 30, 60):
        raise ValueError("horizon_minutes must be one of 15, 30 or 60")
    step = horizon_minutes // INTERVAL_MINUTES + 1
    grouped = bars.groupby(["code", "date"], sort=False)
    entry_open = grouped.open.shift(-1)
    exit_open = grouped.open.shift(-step)
    exit_start = grouped.bar_start.shift(-step)
    valid = (
        exit_start.sub(bars.bar_start).eq(step * INTERVAL_MINUTES)
        & exit_start.le(915)
        & entry_open.gt(0)
        & exit_open.gt(0)
    )
    target = (exit_open / entry_open - 1.0).where(valid)
    target.name = f"target_{horizon_minutes}m"
    return target.astype(np.float32)


def _summarize(by_stock: list[dict], trades: list[dict], daily: list[dict],
               benchmark: list[dict], initial_balance: float, truncated: int) -> dict:
    stock_frame = pd.DataFrame(by_stock)
    benchmark_frame = pd.DataFrame(benchmark)
    # ``baseline_return_pct`` in by_stock is the full-period compounded,
    # ticker-matched 09:30–15:15 intraday benchmark.
    excess = stock_frame.model_return_pct - stock_frame.baseline_return_pct
    trade_frame = pd.DataFrame(trades)
    daily_frame = pd.DataFrame(daily)
    gross_trade = float(trade_frame.buy_notional.sum() + trade_frame.sell_notional.sum()) if len(trade_frame) else 0.0
    fees = float(trade_frame.cost_paid.sum()) if len(trade_frame) else 0.0
    daily_returns = daily_frame.model_return.to_numpy(dtype=float) if len(daily_frame) else np.array([])
    sharpe = float(daily_returns.mean() / (daily_returns.std(ddof=0) + 1e-12) * np.sqrt(252)) if len(daily_returns) > 1 else 0.0
    return {
        "stocks": int(len(stock_frame)),
        "trades": int(len(trade_frame)),
        "active_stock_days": int(daily_frame.loc[daily_frame.trade_count > 0, ["code", "date"]].drop_duplicates().shape[0]) if len(daily_frame) else 0,
        "mean_return_pct": float(stock_frame.model_return_pct.mean()),
        "median_return_pct": float(stock_frame.model_return_pct.median()),
        "mean_baseline_return_pct": float(stock_frame.baseline_return_pct.mean()),
        "mean_excess_baseline_pct": float(excess.mean()),
        "median_excess_baseline_pct": float(excess.median()),
        "positive_excess_stock_fraction": float((excess > 0).mean()),
        "mean_drawdown_pct": float(stock_frame.max_drawdown_pct.mean()),
        "worst_drawdown_pct": float(stock_frame.max_drawdown_pct.min()),
        "daily_sharpe": sharpe,
        "win_rate": float((trade_frame.net_pnl > 0).mean()) if len(trade_frame) else 0.0,
        "mean_trade_bps": float(trade_frame.net_bps.mean()) if len(trade_frame) else 0.0,
        "median_trade_bps": float(trade_frame.net_bps.median()) if len(trade_frame) else 0.0,
        "mean_holding_minutes": float(trade_frame.holding_minutes.mean()) if len(trade_frame) else 0.0,
        "fee_and_tax_paid": fees,
        "turnover_x_initial_capital": gross_trade / (initial_balance * max(len(stock_frame), 1)),
        "truncated_exits": int(truncated),
    }


def simulate_five_minute(data: pd.DataFrame, prediction: np.ndarray, horizon_minutes: int,
                         entry_threshold: float, *, initial_balance: float = 10_000_000.0,
                         allocation: float = 0.25, participation: float = 0.01,
                         buy_cost: float = 0.00045, sell_cost: float = 0.00245,
                         detail: bool = False):
    """Simulate single-stock 5-minute signals with next-bar-open fills.

    Positions are long-only, non-overlapping and intraday. The sell-side cost
    includes Korea's 20bp transaction tax in addition to fee and slippage.
    """
    if len(data) != len(prediction):
        raise ValueError("prediction and 5-minute data length mismatch")
    if not 0 <= allocation <= 1 or participation < 0:
        raise ValueError("invalid allocation or participation")
    if buy_cost < 0 or sell_cost < 0:
        raise ValueError("trading costs must be non-negative")

    bars = data.reset_index(drop=True).copy()
    bars["prediction"] = np.asarray(prediction, dtype=float)
    horizon_steps = horizon_minutes // INTERVAL_MINUTES
    balances = {code: float(initial_balance) for code in bars.code.unique()}
    peaks = balances.copy()
    worst_dd = {code: 0.0 for code in balances}
    trades: list[dict] = []
    daily: list[dict] = []
    truncated = 0

    for (code, date), session in bars.groupby(["code", "date"], sort=True):
        session = session.sort_values("bar_start")
        op = session.open.to_numpy(dtype=float)
        cl = session.close.to_numpy(dtype=float)
        volcap = session.capacity_shares.to_numpy(dtype=float)
        signal = session.prediction.to_numpy(dtype=float)
        starts = session.bar_start.to_numpy(dtype=int)
        minutes = session.minute.to_numpy(dtype=int)
        cash = balances[code]
        start_cash = cash
        q = 0
        entry_index = -1
        scheduled_exit = -1
        entry_price = 0.0
        entry_cost_cash = 0.0
        entry_start = 0
        entry_signal = 0.0
        entry_capacity = 0.0
        session_trade_count = 0
        pending_entry = None

        for i in range(len(session)):
            if pending_entry is not None and pending_entry[0] == i:
                _, quantity, scheduled_exit, entry_signal, entry_capacity = pending_entry
                pending_entry = None
                if quantity > 0 and op[i] > 0:
                    q = quantity
                    entry_index = i
                    entry_price = op[i]
                    entry_start = int(starts[i])
                    entry_cost_cash = q * entry_price * (1 + buy_cost)
                    cash -= entry_cost_cash

            if q and i == scheduled_exit:
                exit_price = op[i]
                buy_notional = q * entry_price
                sell_notional = q * exit_price
                proceeds = sell_notional * (1 - sell_cost)
                net_pnl = proceeds - entry_cost_cash
                cost_paid = entry_cost_cash - buy_notional + sell_notional * sell_cost
                cash += proceeds
                trades.append({
                    "code": code, "date": date,
                    "entry_time": _hhmm(entry_start), "exit_time": _hhmm(int(starts[i])),
                    "holding_minutes": int(starts[i] - entry_start),
                    "prediction": float(entry_signal), "capacity_shares": float(entry_capacity),
                    "shares": int(q), "buy_notional": buy_notional,
                    "sell_notional": sell_notional, "cost_paid": cost_paid,
                    "net_pnl": net_pnl, "net_bps": net_pnl / max(entry_cost_cash, 1e-12) * 10000,
                    "reason": "fixed_horizon",
                })
                q = 0
                entry_index = -1
                scheduled_exit = -1
                session_trade_count += 1

            equity = cash + q * cl[i] * (1 - sell_cost)
            peaks[code] = max(peaks[code], equity)
            worst_dd[code] = min(worst_dd[code], equity / max(peaks[code], 1e-12) - 1)

            if q == 0 and pending_entry is None:
                # Wait for a full 30-minute opening profile. A 5-minute bar
                # starting at 09:25 has closed at 09:29; it can signal a 09:30 fill.
                if int(starts[i]) >= 565 and int(minutes[i]) < 900 and signal[i] > entry_threshold:
                    next_i = i + 1
                    exit_i = next_i + horizon_steps
                    if exit_i < len(session):
                        continuous = bool(np.all(np.diff(starts[i:exit_i + 1]) == INTERVAL_MINUTES))
                        exit_ok = int(starts[exit_i]) <= 915
                        if continuous and exit_ok and op[next_i] > 0 and op[exit_i] > 0:
                            available_cash = cash * allocation
                            liquidity_shares = max(int(volcap[i]), 0)
                            quantity = int(min(
                                available_cash / (op[next_i] * (1 + buy_cost)),
                                liquidity_shares,
                            ))
                            if quantity > 0:
                                pending_entry = (next_i, quantity, exit_i, signal[i], volcap[i])

        if q:
            # This path is recorded as truncated and fails the candidate gate.
            idx = len(session) - 1
            sell_notional = q * cl[idx]
            proceeds = sell_notional * (1 - sell_cost)
            net_pnl = proceeds - entry_cost_cash
            cost_paid = entry_cost_cash - q * entry_price + sell_notional * sell_cost
            cash += proceeds
            truncated += 1
            trades.append({
                "code": code, "date": date, "entry_time": _hhmm(entry_start),
                "exit_time": _hhmm(int(minutes[idx])),
                "holding_minutes": int(minutes[idx] - entry_start),
                "prediction": float(entry_signal), "capacity_shares": float(entry_capacity),
                "shares": int(q), "buy_notional": q * entry_price,
                "sell_notional": sell_notional, "cost_paid": cost_paid,
                "net_pnl": net_pnl, "net_bps": net_pnl / max(entry_cost_cash, 1e-12) * 10000,
                "reason": "truncated_session_close",
            })
        balances[code] = cash
        daily.append({
            "code": code, "date": date,
            "model_return": cash / max(start_cash, 1e-12) - 1.0,
            "trade_count": session_trade_count,
        })

    # Matched intraday baseline: use the same 30-minute warm-up and fill cost.
    benchmark_cash = {code: float(initial_balance) for code in bars.code.unique()}
    benchmark: list[dict] = []
    for (code, date), session in bars.groupby(["code", "date"], sort=True):
        session = session.sort_values("bar_start")
        starts = session.bar_start.to_numpy(dtype=int)
        op = session.open.to_numpy(dtype=float)
        volcap = session.capacity_shares.to_numpy(dtype=float)
        entry_candidates = np.flatnonzero(starts == 570)
        exit_candidates = np.flatnonzero(starts == 915)
        cash = benchmark_cash[code]
        before = cash
        if len(entry_candidates) and len(exit_candidates):
            # Signal at the previous 5-minute close; use its trailing volume cap.
            entry_i = int(entry_candidates[0])
            signal_i = entry_i - 1
            exit_i = int(exit_candidates[0])
            quantity = int(min(
                cash * allocation / (op[entry_i] * (1 + buy_cost)),
                max(int(volcap[signal_i]), 0),
            ))
            if quantity > 0:
                cash -= quantity * op[entry_i] * (1 + buy_cost)
                cash += quantity * op[exit_i] * (1 - sell_cost)
        benchmark_cash[code] = cash
        benchmark.append({"code": code, "date": date, "baseline_return_pct": (cash / max(before, 1e-12) - 1) * 100})

    by_stock = []
    for code, ending_cash in balances.items():
        by_stock.append({
            "code": code,
            "model_return_pct": (ending_cash / initial_balance - 1) * 100,
            "max_drawdown_pct": worst_dd[code] * 100,
            "baseline_return_pct": (benchmark_cash[code] / initial_balance - 1) * 100,
        })
    result = _summarize(by_stock, trades, daily, benchmark, initial_balance, truncated)
    if detail:
        return result, pd.DataFrame(trades), pd.DataFrame(daily), pd.DataFrame(by_stock)
    return result
