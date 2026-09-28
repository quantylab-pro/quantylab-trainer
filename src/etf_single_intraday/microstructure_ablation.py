"""Embargoed within-day pilot; not evidence of out-of-day generalization."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args();root=Path(args.run)
    d=pd.read_csv(root/'forward_pairs.csv',dtype={'code':str})
    d['at']=pd.to_datetime(d['at'],utc=True);d['exit_at']=pd.to_datetime(d.exit_at,utc=True)
    d=d.sort_values(['code','at'])
    d['lag_mid_ret']=d.groupby('code').mid.pct_change().fillna(0)*10000
    cutoff=d['at'].quantile(.65)
    tr=d[d.exit_at<cutoff-pd.Timedelta(minutes=1)]
    te=d[d['at']>=cutoff].copy()
    results={}
    for name,cols in [('baseline',['spread_bps','lag_mid_ret']),
        ('plus_depth',['spread_bps','lag_mid_ret','imbalance','depth_imbalance']),
        ('plus_intensity',['spread_bps','lag_mid_ret','imbalance','depth_imbalance','intensity'])]:
        model=make_pipeline(SimpleImputer(),StandardScaler(),Ridge(alpha=100))
        model.fit(tr[cols],tr.mid_return_bps)
        pred=model.predict(te[cols]);chosen=pred>te.spread_bps.to_numpy()+2
        results[name]=dict(test_mse=float(np.mean((pred-te.mid_return_bps)**2)),
            signals=int(chosen.sum()),mean_signal_quoted_net_bps=float(te.loc[chosen,'quoted_net_bps'].mean()) if chosen.any() else None)
    report=dict(train_rows=len(tr),test_rows=len(te),cutoff=cutoff.isoformat(),results=results,
        warning='One-day exploratory ablation; signals overlap, not independent trades. Fee assumption 2bp, no impact or fill guarantee.')
    (root/'ablation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
