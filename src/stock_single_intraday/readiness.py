"""주식 intraday 학습 데이터의 범위와 종목별 완전성을 확인한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from quantylab.common.db import psql
from quantylab.acquisition.kiwoom.stock_intraday_universe import MAJOR_STOCK_CODES


def summarize(start_date: str = "20260101", end_date: str = "20991231") -> dict:
    """Summarize only the fixed major-stock universe, excluding ETF candles."""
    query = text("""
        SELECT c.code, s.name, c.date, count(*) AS bars,
               count(DISTINCT left(c.time, 4)) AS distinct_minutes,
               min(left(c.time, 4)) AS first_time,
               max(left(c.time, 4)) AS last_time
        FROM stock_minute_candle c
        JOIN stock_code s ON s.code = c.code
        LEFT JOIN etf_code e ON e.code = c.code
        WHERE e.code IS NULL
          AND c.code = ANY(:codes)
          AND c.date BETWEEN :start_date AND :end_date
          AND c.date ~ '^[0-9]{8}$'
          AND c.time ~ '^[0-9]{4,6}$'
          AND left(c.time, 4) BETWEEN '0900' AND '1530'
          AND abs(c.open) > 0 AND abs(c.high) > 0
          AND abs(c.low) > 0 AND abs(c.close) > 0
        GROUP BY c.code, s.name, c.date
        ORDER BY c.code, c.date
    """)
    with psql.get_session() as session:
        rows = session.execute(query, {
            "codes": list(MAJOR_STOCK_CODES),
            "start_date": start_date,
            "end_date": end_date,
        }).mappings().all()

    frame = pd.DataFrame(rows)
    if frame.empty:
        return {
            "universe_size": len(MAJOR_STOCK_CODES), "stock_sessions": 0,
            "date_min": None, "date_max": None, "rows": 0, "stocks": [],
        }

    frame["date"] = frame.date.astype(str)
    stocks = []
    for code, stock in frame.groupby("code", sort=True):
        stocks.append({
            "code": code,
            "name": str(stock["name"].iloc[0]),
            "sessions": int(stock.date.nunique()),
            "bars": int(stock.bars.sum()),
            "median_bars_per_session": float(stock.bars.median()),
            "first_date": str(stock.date.min()),
            "last_date": str(stock.date.max()),
        })
    return {
        "universe_size": len(MAJOR_STOCK_CODES),
        "stocks_with_data": len(stocks),
        "stock_sessions": int(frame.date.nunique()),
        "date_min": str(frame.date.min()),
        "date_max": str(frame.date.max()),
        "rows": int(frame.bars.sum()),
        "stocks": stocks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="20260101")
    parser.add_argument("--end-date", default="20991231")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = summarize(args.start_date, args.end_date)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
