"""Delayed quoted-cost, nonoverlapping depth-feature pilot and day validation gate."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from .microstructure_audit import extract
from .priority_features import depth_features,join_nav


def main():
    p=argparse.ArgumentParser();p.add_argument('--roots',nargs='+',required=True)
    p.add_argument('--nav-root',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    rows=[]
    for root in args.roots:
        for path in Path(root).glob('*/*/*.json'):
            r=json.loads(path.read_text())
            if r.get('status')=='captured':rows.append({**extract(r),**depth_features(r['quote'])})
    d=pd.DataFrame(rows).drop_duplicates(['code','at']).sort_values(['code','at'])
    d['at']=pd.to_datetime(d['at'],utc=True)
    d['date']=d['at'].dt.tz_convert('Asia/Seoul').dt.strftime('%Y%m%d')
    g=d.groupby(['code','date'])
    gap=g['at'].diff().dt.total_seconds()
    d['lag_return']=g.mid.pct_change().where(gap.between(0,120),0).fillna(0)*10000
    d['depth_change']=g.imbalance_10.diff().where(gap.between(0,120),0).fillna(0)
    records=[json.loads(p.read_text()) for p in Path(args.nav_root).rglob('nav*.json')]
    records=[r for r in records if r.get('status')=='captured' and r.get('received_at')]
    d=join_nav(d,records)
    d.loc[d.intensity_age>120,'intensity']=np.nan
    d=d[d.fresh&d.bid.gt(0)&d.ask.ge(d.bid)].sort_values('at')
    coverage=int(d.nav_available.sum())
    for name,minutes in [('entry',1),('exit',6)]:
        d['target']=d['at']+pd.Timedelta(minutes=minutes)
        future=d[['code','date','at','ask','bid']].rename(columns={c:f'{name}_{c}' for c in ['at','ask','bid']})
        d=pd.merge_asof(d.sort_values('target'),future.sort_values(f'{name}_at'),by=['code','date'],
            left_on='target',right_on=f'{name}_at',direction='forward',tolerance=pd.Timedelta(seconds=45))
    d=d.dropna(subset=['entry_at','exit_at']).copy()
    d=d[d.exit_at>d.entry_at]
    d['gross_bps']=(d.exit_bid/d.entry_ask-1)*10000
    d.to_csv(out/'samples.csv',index=False)
    dates=sorted(d.date.unique());splits=[]
    if len(dates)>=3:
        for day in dates[2:]:splits.append((day,d[d.date<day],d[d.date==day]))
    else:
        cut=d['at'].quantile(.65)
        splits=[('within_day_only',d[d.exit_at<cut-pd.Timedelta(minutes=1)],d[d['at']>=cut])]
    results=[]
    baseline=['spread_bps','lag_return']
    depth=baseline+['imbalance_1','imbalance_3','imbalance_5','imbalance_10','microprice_gap_bps','depth_change','intensity']
    variants=[('baseline',baseline),('depth',depth)]
    if coverage:variants.append(('nav',depth+['nav_gap_bps','nav_available']))
    for label,tr,te in splits:
        for name,cols in variants:
            model=make_pipeline(SimpleImputer(),StandardScaler(),Ridge(alpha=100))
            model.fit(tr[cols],tr.gross_bps)
            pred=model.predict(te[cols])
            for fee in (2,9,18):
                # Fees added to actual quote spread; no future spread entry filter.
                signals=te.assign(prediction=pred)
                signals=signals[(signals.prediction>fee+2)&(signals.spread_bps<=10)].sort_values('at')
                exits={};trades=[]
                for row in signals.itertuples():
                    if row.code in exits and row.at<=exits[row.code]:continue
                    exits[row.code]=row.exit_at
                    trades.append(dict(code=row.code,at=row.at,exit_at=row.exit_at,net_bps=row.gross_bps-fee))
                frame=pd.DataFrame(trades)
                frame.to_csv(out/f'{label}_{name}_fee{fee}.csv',index=False)
                results.append(dict(split=label,variant=name,fee_bps=fee,trades=len(trades),
                    mean_net_bps=float(frame.net_bps.mean()) if trades else None))
    report=dict(days=dates,rows=len(d),nav_available_rows=coverage,results=results,
        multi_day_validated=len(dates)>=3,nav_ablation_executed=coverage>0,deployment_approved=False,
        warning='One-minute delayed displayed quote prices, no guaranteed fills or depth execution; fixed fees are assumptions. No overlapping positions per ETF. Single-day pilot is not multi-day validation.')
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
