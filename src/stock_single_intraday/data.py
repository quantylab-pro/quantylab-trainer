"""Fixed-universe stock minute data loader with a canonical HHMM clock."""

from __future__ import annotations

from typing import Iterable

import pandas as pd
from sqlalchemy import bindparam, text

from quantylab.common.db import psql
from quantylab.acquisition.kiwoom.stock_intraday_universe import MAJOR_STOCK_CODES


def load_stock_minute_candles(
    start_date: str,
    end_date: str,
    codes: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Read stock candles, excluding ETFs, and normalize HHMMSS to HHMM."""
    target_codes = [str(code).zfill(6) for code in (codes or MAJOR_STOCK_CODES)]
    query = text("""
        SELECT c.code, c.date, left(c.time, 4) AS time, c.time AS source_time,
               c.open, c.high, c.low, c.close, c.volume,
               c.diff, c.diff_ratio, c.amount
        FROM stock_minute_candle c
        JOIN stock_code s ON s.code = c.code
        LEFT JOIN etf_code e ON e.code = c.code
        WHERE e.code IS NULL
          AND c.code IN :codes
          AND c.date BETWEEN :start_date AND :end_date
          AND c.date ~ '^[0-9]{8}$'
          AND c.time ~ '^[0-9]{4,6}$'
          AND left(c.time, 4) BETWEEN '0900' AND '1530'
          AND abs(c.open) > 0 AND abs(c.high) > 0
          AND abs(c.low) > 0 AND abs(c.close) > 0
          AND c.volume >= 0
        ORDER BY c.code, c.date, left(c.time, 4), c.time
    """).bindparams(bindparam("codes", expanding=True))

    with psql.get_session() as session:
        rows = session.execute(query, {
            "codes": target_codes,
            "start_date": start_date,
            "end_date": end_date,
        }).mappings().all()
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    for column in ("open", "high", "low", "close", "volume", "diff", "diff_ratio", "amount"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)
    # Kiwoom uses the price sign to encode direction relative to the previous
    # close. Negative OHLC values are valid prices, not invalid candles.
    for column in ("open", "high", "low", "close"):
        frame[column] = frame[column].abs()
    previous_close = frame.close - frame["diff"]
    frame["diff_ratio"] = (frame["diff"] / previous_close.where(previous_close.gt(0))).fillna(0.0)
    frame["code"] = frame.code.astype(str).str.zfill(6)
    frame["date"] = frame.date.astype(str)
    frame["time"] = frame.time.astype(str).str.zfill(4)
    # Some rows store HHMMSS and others canonical HHMM. Keep the latest
    # source-time update inside each minute before discarding the seconds.
    frame = frame.drop_duplicates(["code", "date", "time"], keep="last")
    frame = frame.drop(columns=["source_time"])
    return frame.sort_values(["code", "date", "time"]).reset_index(drop=True)
