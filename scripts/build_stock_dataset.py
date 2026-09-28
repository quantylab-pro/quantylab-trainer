#!/usr/bin/env python3
"""Build a causal stock single-swing dataset from an exported DB snapshot.

The raw input is intentionally a simple CSV export containing one row per
stock/date with a JSON ``content`` column from ``stock_feature`` and OHLCV
columns from ``stock_day_candle``. Scaling statistics are fitted only on the
training period.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


RAW_COLUMNS = {
    "code": "code",
    "date": "date",
    "stock_name": "stock_name",
    "market_name": "market_name",
    "content": "content",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
}

LOG_FEATURES = {
    "open", "high", "low", "close", "volume", "ma20", "lbb", "ubb",
    "marketcap", "bps", "eps", "eps_krx", "bps_krx", "trans_price_exp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="주식 single swing 학습 데이터 빌드")
    parser.add_argument("--input", required=True, help="원격 DB에서 export한 CSV(.gz)")
    parser.add_argument("--output", required=True, help="data/<dataset> 출력 경로")
    parser.add_argument("--train-end-date", default="20241231")
    parser.add_argument("--validation-start-date", default="20250101")
    parser.add_argument("--validation-end-date", default="20251231")
    parser.add_argument("--min-rows", type=int, default=500)
    return parser.parse_args()


def _json_frame(series: pd.Series) -> pd.DataFrame:
    records = []
    for value in series:
        try:
            record = json.loads(value) if isinstance(value, str) else {}
            records.append(record if isinstance(record, dict) else {})
        except (TypeError, json.JSONDecodeError):
            records.append({})
    frame = pd.DataFrame.from_records(records, index=series.index)
    if "name" in frame:
        frame = frame.drop(columns=["name"])
    return frame


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.input, compression="infer", low_memory=False)
    missing = sorted(set(RAW_COLUMNS.values()) - set(raw.columns))
    if missing:
        raise ValueError(f"raw export 필수 컬럼 누락: {missing}")
    raw = raw.rename(columns={v: k for k, v in RAW_COLUMNS.items()})
    raw["code"] = raw["code"].astype(str).str.zfill(6)
    raw["date"] = raw["date"].astype(str).str.zfill(8)
    raw = raw.sort_values(["code", "date"]).reset_index(drop=True)

    counts = raw.groupby("code")["date"].size()
    valid_codes = counts[counts >= args.min_rows].index
    raw = raw[raw["code"].isin(valid_codes)].reset_index(drop=True)
    if raw.empty:
        raise ValueError("min_rows 조건을 만족하는 stock이 없습니다.")

    content = _json_frame(raw["content"])
    numeric = {}
    for column in content.columns:
        values = pd.to_numeric(content[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if column in LOG_FEATURES:
            values = np.sign(values) * np.log1p(values.abs())
        numeric[f"stock_{column}"] = values.to_numpy(dtype=np.float64)
    features = pd.DataFrame(numeric, index=raw.index)
    features["stock_market_kospi"] = (raw["market_name"].eq("kospi")).astype(float)
    features["stock_market_kosdaq"] = (raw["market_name"].eq("kosdaq")).astype(float)
    features = features.replace([np.inf, -np.inf], np.nan)

    train_mask = raw["date"] <= str(args.train_end_date).zfill(8)
    if train_mask.sum() < 1000:
        raise ValueError("학습 구간이 너무 짧습니다.")
    medians = features.loc[train_mask].median().fillna(0.0)
    features = features.fillna(medians).fillna(0.0)
    scaler = StandardScaler()
    scaler.fit(features.loc[train_mask])
    scaled = scaler.transform(features).astype(np.float32)
    scaled = np.clip(scaled, -8.0, 8.0)

    env = raw[["date", "open", "high", "low", "close", "volume", "code", "stock_name", "market_name"]].copy()
    env = env.rename(columns={"code": "stock_code"})
    env.to_csv(output / "environment.csv", index=False)
    pd.DataFrame(scaled, columns=features.columns).to_csv(output / "training_scaled.csv", index=False)
    raw[["code"]].rename(columns={"code": "stock_code"}).to_csv(output / "stock_codes.csv", index=False)
    joblib.dump(
        {"scaler": scaler, "medians": medians.to_dict(), "feature_names": list(features.columns)},
        output / "scaler.pkl",
    )

    dates = raw["date"]
    metadata = {
        "dataset_type": "stock-single-swing",
        "source": "stock_feature + stock_day_candle export",
        "rows": int(len(raw)),
        "stock_count": int(raw["code"].nunique()),
        "stock_codes": sorted(raw["code"].unique().tolist()),
        "feature_dim": int(scaled.shape[1]),
        "feature_names": list(features.columns),
        "date_start": str(dates.min()),
        "date_end": str(dates.max()),
        "train_end_date": str(args.train_end_date),
        "validation_start_date": str(args.validation_start_date),
        "validation_end_date": str(args.validation_end_date),
        "locked_evaluation_start_date": str(
            (pd.to_datetime(args.validation_end_date, format="%Y%m%d") + pd.Timedelta(days=1)).strftime("%Y%m%d")
        ),
        "scaler_fit_period": f"<= {args.train_end_date}",
        "min_rows": int(args.min_rows),
    }
    with open(output / "dataset_meta.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print(f"dataset: {output}")
    print(f"stocks: {metadata['stock_count']:,}, rows: {metadata['rows']:,}, features: {metadata['feature_dim']}")
    print(f"dates: {metadata['date_start']}~{metadata['date_end']}")
    print(f"train scaling: {metadata['scaler_fit_period']}")


if __name__ == "__main__":
    main()
