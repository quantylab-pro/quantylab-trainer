"""Day-clustered diagnostics, not a multiple-testing-adjusted significance test."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args()
    root=Path(args.run)
    t=pd.read_csv(root/'cost18_trades.csv',dtype={'code':str,'date':str})
    d=pd.read_csv(root/'cost18_daily.csv').groupby('date').return_.mean()
    pnl=t.groupby('date').pnl.sum()
    samples=np.random.default_rng(42).choice(d.to_numpy(),(10000,len(d)),replace=True).sum(axis=1)*100
    result=dict(active_days=int(len(pnl)),test_days=len(d),
        pnl_by_day=pnl.to_dict(),net_pnl=float(pnl.sum()),
        net_pnl_excluding_best_day=float(pnl.sum()-pnl.max()),
        approximate_return_95pct_day_bootstrap=np.quantile(samples,[.025,.975]).tolist(),
        warning='Additive equal-ETF daily return bootstrap; small sample, assumes exchangeable days; not adjusted for repeated model selection.')
    (root/'robustness.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
