"""Crypto context ablation with closed-bar +60s availability assumption."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from .minute_engine import BASE,EXTRA,simulate
from .minute_experiments import FOLDS


def crypto_features(c):
    c=c.sort_values('bar_close_utc').drop_duplicates('bar_close_utc').copy()
    for n in (1,5,15):
        valid=c.bar_close_utc.diff(n).eq(pd.Timedelta(minutes=n))
        c[f'ret{n}']=c.close.pct_change(n).where(valid)
    c['vol15']=c.ret1.rolling(15,min_periods=15).std()
    c['taker_ratio']=c.taker_quote/c.quote_volume.replace(0,np.nan)
    c['available_at']=c.bar_close_utc+pd.Timedelta(seconds=60)
    return c[['available_at','ret1','ret5','ret15','vol15','taker_ratio']]


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--roots',nargs='+',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    d=pd.read_pickle(args.cache)
    d['at']=pd.to_datetime(d.date.astype(str)+d.time.astype(str).str.zfill(4),format='%Y%m%d%H%M').dt.tz_localize('Asia/Seoul').dt.tz_convert('UTC')
    extra=[];coverage={}
    for symbol in ['BTCUSDT','ETHUSDT']:
        c=pd.concat([pd.read_parquet(p) for root in args.roots for p in Path(root).glob(symbol+'*.parquet')])
        c=crypto_features(c);cols=[x for x in c if x!='available_at']
        c=c.rename(columns={x:f'{symbol}_{x}' for x in cols})
        d=pd.merge_asof(d.sort_values('at'),c.sort_values('available_at'),left_on='at',right_on='available_at',direction='backward',tolerance=pd.Timedelta(seconds=90))
        coverage[symbol]=float(d.available_at.notna().mean());d=d.drop(columns='available_at')
        extra.extend(f'{symbol}_{x}' for x in cols)
    d.to_pickle(out/'samples.pkl');results=[]
    for name,cols in [('baseline',BASE+EXTRA),('crypto',BASE+EXTRA+extra)]:
        metrics=[]
        for end,start,stop in FOLDS:
            tr=d[(d.date<=end)&(d.minute%5==0)&d.y15.notna()&d.minute.between(555,899)]
            va=d[d.date.between(start,stop)]
            model=HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=7,min_samples_leaf=200,l2_regularization=30,early_stopping=False,random_state=42)
            with threadpool_limits(limits=2):
                model.fit(tr[cols],tr.y15.clip(-.03,.03));pred=model.predict(va[cols])
            metrics.append(simulate(va,pred,15,.0018,cost=.0018))
        joblib.dump(dict(model=model,features=cols,training_end='20260731',deployment_approved=False),out/f'{name}.joblib')
        recent=d[d.date>='20260818']
        with threadpool_limits(limits=2):pred=model.predict(recent[cols])
        tests={}
        for cost in (.0018,.0036):
            m,t,days=simulate(recent,pred,15,.0018,cost=cost,detail=True)
            tests[str(cost)]=m;t.to_csv(out/f'{name}_{cost}_trades.csv',index=False)
        results.append(dict(name=name,folds=metrics,recent=tests));print(name,tests,flush=True)
    report=dict(coverage=coverage,experiments=results,deployment_approved=False,
        warning='Repeated retrospective test; archive close time +60sec is assumed availability, not recorded historical arrival. US series not used until publication/vintage validation.')
    (out/'report.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
