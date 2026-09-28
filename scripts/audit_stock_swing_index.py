"""Compare a completed stock swing candidate with KOSPI/KOSDAQ index opens."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    run = args.run
    manifest = json.loads((run / "manifest.json").read_text())
    codes = manifest["stock-portfolio-swing"]["universe"]
    env = pd.read_csv(Path(manifest["dataset"]) / "environment.csv",
                      usecols=["date", "stock_code"], dtype=str)
    source = run / "kospi_kosdaq_index_2025_2026.csv"
    index = pd.read_csv(source, dtype={"date": str})
    opens = index.pivot(index="date", columns="code", values="open").sort_index()
    results = {}
    for name, start, end in (("2025", "20250101", "20251231"), ("2026", "20260101", "20260916")):
        eligible = env.loc[env.stock_code.isin(codes) & env.date.between(start, end)]
        counts = eligible.groupby("date").stock_code.nunique()
        dates = counts[counts.eq(len(codes))].index.sort_values().tolist()
        aligned = opens.reindex(dates)[["kospi", "kosdaq"]]
        if aligned.isna().any().any():
            raise ValueError(f"Missing market index open for {name}")
        daily = aligned.iloc[2:].to_numpy(float) / aligned.iloc[1:-1].to_numpy(float) - 1
        series = {"kospi": daily[:, 0], "kosdaq": daily[:, 1],
                  "equal_weight_kospi_kosdaq_daily": daily.mean(axis=1)}
        returns = {key: float(np.prod(1 + value) - 1) for key, value in series.items()}
        selected_k = str(manifest["stock-portfolio-swing"]["top_k"])
        candidate = (manifest["stock-portfolio-swing"]["validation"][selected_k]["return"] if name == "2025"
                     else manifest["stock-portfolio-swing"]["retrospective"]["return"])
        returns.update({"candidate_return": candidate,
                        "excess_vs_50_50_percentage_points": float((candidate - returns["equal_weight_kospi_kosdaq_daily"]) * 100),
                        "days": len(daily), "first_trade_open": dates[1], "last_mark_open": dates[-1]})
        results[name] = returns
    report = {"source": str(source), "method": "Index open-to-open returns matched to candidate trade dates; index is a reference series without ETF trading costs.",
              "results": results}
    (run / "index_benchmark_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    manifest["index_benchmark_audit"] = "index_benchmark_audit.json"
    (run / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
