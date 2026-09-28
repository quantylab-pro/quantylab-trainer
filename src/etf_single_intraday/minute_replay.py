"""Replay selected minute model using exact feature warmup and real bar dates."""
import argparse
import json
from pathlib import Path
import joblib
import pandas as pd
from threadpoolctl import threadpool_limits
from ..intraday_features import load_minute_candles
from .minute_engine import features,simulate


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',required=True)
    p.add_argument('--date',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    root=Path(args.run);bundle=joblib.load(root/'selected_candidate.joblib')
    if args.date<=bundle['training_end']:raise ValueError('test must follow training')
    history=pd.read_pickle(root/'minute_samples.pkl')
    # Prior 40 calendar days cover the 20-session seasonal lookback.
    start=(pd.Timestamp(args.date)-pd.Timedelta(days=40)).strftime('%Y%m%d')
    history=history[history.date.between(start,args.date,inclusive='left')]
    today=load_minute_candles(args.date,args.date,codes=bundle['codes'])
    if today.empty:raise ValueError('No test candles')
    cols=['code','date','time','open','high','low','close','volume']
    d=features(pd.concat([history[cols],today[cols]],ignore_index=True))
    d=d[d.date==args.date]
    with threadpool_limits(limits=2):pred=bundle['model'].predict(d[bundle['features']])
    if bundle.get('transform')=='risk_scaled':pred=pred*d.vol5.clip(lower=.0005).to_numpy()
    report={}
    for name,cost in [('base',.0009),('double_cost',.0018),('cost_36bp',.0036)]:
        m,t,daily=simulate(d,pred,bundle['horizon'],bundle['threshold'],bundle['dynamic'],cost=cost,detail=True)
        report[name]=m;t.to_csv(out/f'{name}_trades.csv',index=False)
        daily.to_csv(out/f'{name}_daily.csv',index=False)
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
