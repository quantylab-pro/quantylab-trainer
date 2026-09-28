"""Point-in-time minute candles, quotes, and execution-strength observations.

Only live stock_bid snapshots with a recorded receipt time are eligible.  The
legacy daily stock_bid snapshots and daily trade_intensity table are not
minute-level observations.  Missing minutes remain missing; no backward or
forward fill is allowed across a decision boundary.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import bindparam, text

from quantylab.acquisition.kiwoom.stock_intraday_universe import MAJOR_STOCK_CODES
from quantylab.common.db import psql

from .data import load_stock_minute_candles


SEOUL = ZoneInfo("Asia/Seoul")
QUOTE_MAX_AGE_SECONDS = 120
INTENSITY_MAX_AGE_SECONDS = 120
CLOCK_LEAD_TOLERANCE_SECONDS = 30


def _source_age_seconds(frame: pd.DataFrame, column: str) -> pd.Series:
    source = frame[column].astype("string")
    valid = source.str.fullmatch(r"\d{6}").fillna(False)
    source_stamp = pd.to_datetime(
        frame.date.astype(str) + source.where(valid, "000000"),
        format="%Y%m%d%H%M%S", errors="coerce",
    ).dt.tz_localize(SEOUL)
    age = (frame.received_at - source_stamp).dt.total_seconds()
    return age.where(valid)


def load_stock_microstructure(
    start_date: str,
    end_date: str,
    codes: list[str] | None = None,
) -> pd.DataFrame:
    """Return one quote and execution observation per market event minute.

    The API's intensity response contains historical rows. Only the newest row
    already available at receipt is used, never later rows from a later poll.
    """
    target_codes = [str(code).zfill(6) for code in (codes or MAJOR_STOCK_CODES)]
    query = text("""
        SELECT code, date, time AS quote_source_key,
               extra->>'received_at' AS received_at,
               extra->>'source_time' AS quote_source_time,
               extra->>'spread_bps' AS spread_bps,
               extra->>'imbalance' AS imbalance,
               content->0->>'매수잔량' AS best_bid_qty,
               content->0->>'매도잔량' AS best_ask_qty,
               extra->'quote'->>'tot_buy_req' AS bid_depth_10,
               extra->'quote'->>'tot_sel_req' AS ask_depth_10,
               extra->'intensity'->0->>'cntr_tm' AS intensity_source_time,
               extra->'intensity'->0->>'cntr_str' AS trade_intensity,
               extra->'intensity'->0->>'cntr_str_5min' AS trade_intensity_5m,
               extra->'intensity'->0->>'cntr_str_20min' AS trade_intensity_20m,
               extra->'intensity'->0->>'trde_qty' AS trade_quantity
        FROM stock_bid
        WHERE code IN :codes AND date BETWEEN :start_date AND :end_date
          AND extra ? 'received_at'
        ORDER BY code, date, time
    """).bindparams(bindparam("codes", expanding=True))
    with psql.get_session() as session:
        rows = session.execute(query, {
            "codes": target_codes, "start_date": start_date,
            "end_date": end_date,
        }).mappings().all()
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["received_at"] = pd.to_datetime(frame.received_at, utc=True, errors="coerce").dt.tz_convert(SEOUL)
    frame = frame.loc[frame.received_at.notna()].copy()
    frame["receipt_date"] = frame.received_at.dt.strftime("%Y%m%d")
    frame["receipt_minute"] = frame.received_at.dt.strftime("%H%M")
    # Provider time is the market event minute; receipt time controls when it
    # may enter a signal. A poll close to a minute boundary can be received
    # just before the provider's minute label changes.
    frame["time"] = frame.quote_source_time.astype("string").str[:4]
    frame = frame.loc[
        frame.date.eq(frame.receipt_date)
        & frame.receipt_minute.between("0900", "1519")
        & frame.time.str.fullmatch(r"\d{4}").fillna(False)
        & frame.time.between("0900", "1519")
    ].copy()
    frame["quote_age_seconds"] = _source_age_seconds(frame, "quote_source_time")
    frame["intensity_age_seconds"] = _source_age_seconds(frame, "intensity_source_time")
    frame["quote_fresh"] = frame.quote_age_seconds.between(
        -CLOCK_LEAD_TOLERANCE_SECONDS, QUOTE_MAX_AGE_SECONDS
    )
    frame["intensity_fresh"] = frame.intensity_age_seconds.between(
        -CLOCK_LEAD_TOLERANCE_SECONDS, INTENSITY_MAX_AGE_SECONDS
    )
    numeric = ("spread_bps", "imbalance", "best_bid_qty", "best_ask_qty",
               "bid_depth_10", "ask_depth_10",
               "trade_intensity", "trade_intensity_5m", "trade_intensity_20m",
               "trade_quantity")
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["quote_fresh"] &= (
        frame.spread_bps.gt(0) & frame.best_bid_qty.gt(0)
        & frame.best_ask_qty.gt(0) & frame.bid_depth_10.gt(0)
        & frame.ask_depth_10.gt(0)
    )
    total_depth = frame.bid_depth_10 + frame.ask_depth_10
    frame["book_imbalance_10"] = (
        (frame.bid_depth_10 - frame.ask_depth_10) / total_depth.where(total_depth.gt(0))
    )
    frame["intensity_fresh"] &= frame.trade_intensity.notna()
    # Preserve the last received observation for each market event minute.
    frame = frame.sort_values("received_at").drop_duplicates(
        ["code", "date", "time"], keep="last"
    )
    return frame.sort_values(["code", "date", "time"]).reset_index(drop=True)


def load_complete_stock_minutes(
    start_date: str,
    end_date: str,
    codes: list[str] | None = None,
) -> pd.DataFrame:
    """Inner join valid observations from all three streams by stock/minute."""
    candles = load_stock_minute_candles(start_date, end_date, codes=codes)
    quotes = load_stock_microstructure(start_date, end_date, codes=codes)
    if candles.empty or quotes.empty:
        return pd.DataFrame()
    quotes = quotes.loc[quotes.quote_fresh & quotes.intensity_fresh].copy()
    joined = candles.merge(
        quotes[["code", "date", "time", "received_at", "quote_age_seconds",
                "intensity_age_seconds", "spread_bps", "imbalance",
                "best_bid_qty", "best_ask_qty", "bid_depth_10",
                "ask_depth_10", "book_imbalance_10", "trade_intensity",
                "trade_intensity_5m", "trade_intensity_20m", "trade_quantity"]],
        on=["code", "date", "time"], how="inner", validate="one_to_one",
    )
    # A minute candle is final only after that minute. The receipt timestamp
    # makes the quote/strength available, but cannot make the candle available.
    joined["available_at"] = pd.to_datetime(
        joined.date + joined.time, format="%Y%m%d%H%M"
    ).dt.tz_localize(SEOUL) + pd.Timedelta(minutes=1)
    joined["available_at"] = joined[["available_at", "received_at"]].max(axis=1)
    return joined.sort_values(["code", "date", "time"]).reset_index(drop=True)


def aggregate_complete_five_minute(minutes: pd.DataFrame) -> pd.DataFrame:
    """Make causal 5-minute features only from five complete minute records."""
    if minutes.empty:
        return pd.DataFrame()
    frame = minutes.copy()
    clock = frame.time.str[:2].astype(int) * 60 + frame.time.str[2:4].astype(int)
    frame["bar_start"] = (clock // 5) * 5
    grouped = frame.groupby(["code", "date", "bar_start"], sort=True)
    bars = grouped.agg(
        observed_minutes=("time", "nunique"),
        quote_spread_bps_mean=("spread_bps", "mean"),
        quote_spread_bps_last=("spread_bps", "last"),
        quote_imbalance_mean=("imbalance", "mean"),
        quote_imbalance_last=("imbalance", "last"),
        best_bid_qty_last=("best_bid_qty", "last"),
        best_ask_qty_last=("best_ask_qty", "last"),
        bid_depth_10_last=("bid_depth_10", "last"),
        ask_depth_10_last=("ask_depth_10", "last"),
        book_imbalance_10_mean=("book_imbalance_10", "mean"),
        book_imbalance_10_last=("book_imbalance_10", "last"),
        trade_intensity_mean=("trade_intensity", "mean"),
        trade_intensity_last=("trade_intensity", "last"),
        trade_intensity_5m_last=("trade_intensity_5m", "last"),
        trade_intensity_20m_last=("trade_intensity_20m", "last"),
        trade_quantity_sum=("trade_quantity", "sum"),
        available_at=("available_at", "max"),
    ).reset_index()
    bars = bars.loc[bars.observed_minutes.eq(5)].copy()
    # The fifth minute's candle is not final until bar_start + 5 minutes.
    start = pd.to_datetime(bars.date, format="%Y%m%d").dt.tz_localize(SEOUL)
    bar_close = start + pd.to_timedelta(bars.bar_start + 5, unit="m")
    bars["available_at"] = bars[["available_at"]].assign(bar_close=bar_close).max(axis=1)
    return bars.loc[bars.available_at.le(bar_close)].reset_index(drop=True)


def iter_complete_stock_days(start_date: str, end_date: str):
    """Stream joined sessions one date at a time to bound training memory."""
    query = text("""
        SELECT DISTINCT date FROM stock_bid
        WHERE code IN :codes AND date BETWEEN :start_date AND :end_date
          AND extra ? 'received_at'
        ORDER BY date
    """).bindparams(bindparam("codes", expanding=True))
    with psql.get_session() as session:
        dates = session.execute(query, {
            "codes": list(MAJOR_STOCK_CODES),
            "start_date": start_date, "end_date": end_date,
        }).scalars().all()
    for date in dates:
        frame = load_complete_stock_minutes(date, date)
        if not frame.empty:
            yield date, frame


def summarize(start_date: str, end_date: str) -> dict:
    # Keep the audit bounded to one day at a time, even after months of data
    # have accumulated. The raw JSONB snapshots are large.
    query = text("""
        SELECT DISTINCT date FROM stock_minute_candle
        WHERE code IN :codes AND date BETWEEN :start_date AND :end_date
        UNION
        SELECT DISTINCT date FROM stock_bid
        WHERE code IN :codes AND date BETWEEN :start_date AND :end_date
          AND extra ? 'received_at'
        ORDER BY date
    """).bindparams(bindparam("codes", expanding=True))
    with psql.get_session() as session:
        dates = session.execute(query, {
            "codes": list(MAJOR_STOCK_CODES),
            "start_date": start_date, "end_date": end_date,
        }).scalars().all()
    days = []
    for date in dates:
        raw_count = session_count_live_snapshots(date)
        c = load_stock_minute_candles(date, date)
        if not c.empty:
            c = c.loc[c.time.between("0900", "1519")]
        s = load_stock_microstructure(date, date)
        joined = load_complete_stock_minutes(date, date)
        five = aggregate_complete_five_minute(joined)
        days.append({
            "date": date,
            "raw_live_snapshot_rows": raw_count,
            "candle_minutes": int(len(c)),
            "snapshot_source_minutes": int(len(s)),
            "fresh_quote_minutes": int(s.quote_fresh.sum()) if not s.empty else 0,
            "fresh_intensity_minutes": int(s.intensity_fresh.sum()) if not s.empty else 0,
            "complete_minutes": int(len(joined)),
            "complete_five_minute_bars": int(len(five)),
            "stocks_with_complete_minutes": int(joined.code.nunique()) if not joined.empty else 0,
        })
    return {"universe_size": len(MAJOR_STOCK_CODES), "days": days,
            "generated_at": datetime.now(SEOUL).isoformat()}


def session_count_live_snapshots(date: str) -> int:
    query = text("""
        SELECT count(*) FROM stock_bid
        WHERE code IN :codes AND date = :date AND extra ? 'received_at'
    """).bindparams(bindparam("codes", expanding=True))
    with psql.get_session() as session:
        return int(session.execute(query, {
            "codes": list(MAJOR_STOCK_CODES), "date": date,
        }).scalar_one())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="20260923")
    parser.add_argument("--end-date", default="20260928")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = summarize(args.start_date, args.end_date)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        from pathlib import Path
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
