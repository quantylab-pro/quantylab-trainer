"""Read-only forward simulation of a saved research candidate; no orders."""
import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from ..intraday_features import load_minute_candles
from .research import samples, evaluate


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--start-date',required=True)
    p.add_argument('--end-date',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    # Only load artifacts from trusted local training runs.
    bundle=joblib.load(args.model)
    if args.start_date <= bundle['training_end']:
        raise ValueError('Forward evaluation must start after training_end')
    warmup=(datetime.strptime(args.start_date,'%Y%m%d')-timedelta(days=14)).strftime('%Y%m%d')
    candles=load_minute_candles(warmup,args.end_date,codes=bundle['codes'])
    if candles.empty:
        raise ValueError('No forward candles available')
    data=samples(candles,bundle['horizon']).sort_values(['date','code','time']).reset_index(drop=True)
    data=data[data.date >= args.start_date].reset_index(drop=True)
    if data.empty:
        raise ValueError('No consecutive tradable windows available')
    with threadpool_limits(limits=2):
        prediction=bundle['model'].predict(data[bundle['features']])
    dates=sorted(candles.loc[candles.date >= args.start_date,'date'].unique())
    result={}
    for name,cost in [('base',.0009),('double_cost',.0018)]:
        result[name]=evaluate(data,prediction,bundle['threshold'],bundle['codes'],dates,cost)
    result['cash']=evaluate(data,np.zeros(len(data)),bundle['threshold'],bundle['codes'],dates)
    result['deployment_approved']=False
    with Path(args.output).open('x') as handle:
        json.dump(result,handle,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
