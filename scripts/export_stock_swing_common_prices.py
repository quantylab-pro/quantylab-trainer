"""Export a broad common-share daily OHLCV research snapshot from local DB.

KRX security classification is the current snapshot and does not reconstruct
the historical listed universe. This export must not be called point-in-time.
"""
from __future__ import annotations

import argparse
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path

from quantylab.common.db import psql


QUERY = """
COPY (
 SELECT c.code, c.date, c.open, c.high, c.low, c.close, c.volume,
        c.amount, s.market_name, s.is_active
 FROM stock_day_candle c
 JOIN stock_code s ON s.code = c.code
 JOIN (SELECT DISTINCT code FROM krx_stock_code
       WHERE content->>'증건구분' = '주권'
         AND content->>'주식종류' = '보통주') k ON k.code = c.code
 WHERE s.market_name IN ('kospi', 'kosdaq')
   AND c.date BETWEEN %(start_date)s AND %(end_date)s
 ORDER BY c.code, c.date
) TO STDOUT WITH (FORMAT CSV, HEADER TRUE)
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start-date", default="20180101")
    parser.add_argument("--end-date", default="20260916")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "ohlcv.csv.gz"
    connection = psql.engine.raw_connection()
    try:
        with connection.cursor() as cursor, gzip.open(path, "wb", compresslevel=5) as handle:
            cursor.copy_expert(cursor.mogrify(QUERY, {"start_date": args.start_date,
                                                     "end_date": args.end_date}).decode(), handle)
    finally:
        connection.close()
    metadata = {"source": "stock_day_candle + current stock_code + current krx_stock_code",
                "filter": "current KRX ordinary shares, KOSPI/KOSDAQ",
                "warning": "Current security type and active status are not point-in-time; delisted histories are incomplete.",
                "date_start": args.start_date, "date_end": args.end_date,
                "exported_at_utc": datetime.now(timezone.utc).isoformat(),
                "file": path.name, "size_bytes": path.stat().st_size}
    (args.output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
