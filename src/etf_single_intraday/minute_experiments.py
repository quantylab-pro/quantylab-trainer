"""Intraday experiment suite: every-minute inference and event simulation."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from ..intraday_features import load_minute_candles
from .minute_engine import features, simulate, BASE, EXTRA

FOLDS=[('20260529','20260601','20260612'),('20260630','20260701','20260715'),
       ('20260731','20260803','20260814')]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--cache')
    p.add_argument('--advanced',action='store_true')
    p.add_argument('--architecture',action='store_true')
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    if args.cache:
        data=pd.read_pickle(args.cache)
        # Upgrade cached labels to the simulator's next-open entry AND exit.
        data=data.sort_values(['code','date','minute']).copy()
        groups=data.groupby(['code','date'])
        for h in (3,5,10,15,30):
            end_minute=groups.minute.shift(-(h+1))
            valid=(end_minute-data.minute).eq(h+1)&end_minute.le(915)
            data[f'y{h}']=(groups.open.shift(-(h+1))/groups.open.shift(-1)-1).where(valid)
    else:
        candles=load_minute_candles('20260129','20260910')
        early=candles[candles.date<='20260529']
        cnt=early.groupby(['code','date']).size().groupby('code').median()
        codes=sorted(cnt[cnt>=330].index)
        print('building minute features',len(codes),flush=True)
        data=features(candles[candles.code.isin(codes)])
    data.to_pickle(out/'minute_samples.pkl')
    print('minute rows',len(data),flush=True)
    rows=[]
    for horizon in ((5,15,30) if args.advanced or args.architecture else (3,5,10,15,30)):
        variants=([('risk_scaled',BASE+EXTRA),('recent_weighted',BASE+EXTRA)] if args.advanced
                  else [('basic',BASE),('seasonal_patterns',BASE+EXTRA)])
        if args.architecture:
            variants=[('linear',BASE+EXTRA),('deep_tree',BASE+EXTRA)]
        for name,cols in variants:
            configs={(dynamic,threshold):[] for dynamic in (False,True) for threshold in (.0009,.0018,.003)}
            for end,start,stop in FOLDS:
                # Subsample training decisions to reduce overlap/compute;
                # validation and trading still evaluate EVERY minute.
                train=data[(data.date<=end)&(data.minute%5==0)&data[f'y{horizon}'].notna()]
                val=data[data.date.between(start,stop)]
                model=HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=7,
                    min_samples_leaf=200,l2_regularization=20,learning_rate=.04,
                    early_stopping=False,random_state=42)
                if name=='linear':
                    model=make_pipeline(StandardScaler(),Ridge(alpha=10000))
                elif name=='deep_tree':
                    model=HistGradientBoostingRegressor(max_iter=150,max_leaf_nodes=15,
                        min_samples_leaf=300,l2_regularization=50,learning_rate=.035,
                        early_stopping=False,random_state=42)
                with threadpool_limits(limits=2):
                    target=train[f'y{horizon}'].clip(-.03,.03)
                    weight=None
                    if name=='risk_scaled':
                        target=(target/train.vol5.clip(lower=.0005)).clip(-20,20)
                    if name=='recent_weighted':
                        age=(pd.Timestamp(end)-pd.to_datetime(train.date)).dt.days
                        weight=np.exp(-np.log(2)*age/45)
                    if name=='linear': model.fit(train[cols],target)
                    else: model.fit(train[cols],target,sample_weight=weight)
                    pred=model.predict(val[cols])
                    if name=='risk_scaled': pred=pred*val.vol5.clip(lower=.0005).to_numpy()
                for (dynamic,threshold),metrics in configs.items():
                    metrics.append(simulate(val,pred,horizon,threshold,dynamic))
            joblib.dump(dict(model=model,features=cols,horizon=horizon,training_end='20260731',
                transform=name,codes=sorted(data.code.unique()),deployment_approved=False),out/f'{name}_{horizon}.joblib')
            for (dynamic,threshold),metrics in configs.items():
                returns=[m['mean_return_pct'] for m in metrics]
                rows.append(dict(name=name,horizon=horizon,dynamic=dynamic,threshold=threshold,
                    folds=metrics,score=float(np.mean(returns)-.5*np.std(returns)),
                    eligible=all(m['trades']>=20 and m['truncated_exits']==0 for m in metrics)))
            print('completed',name,horizon,flush=True)
            (out/'progress.json').write_text(json.dumps(rows,indent=2))
    eligible=[r for r in rows if r['eligible']]
    best=max(eligible,key=lambda r:r['score']) if eligible else max(rows,key=lambda r:r['score'])
    bundle=joblib.load(out/f"{best['name']}_{best['horizon']}.joblib")
    bundle.update(threshold=best['threshold'],dynamic=best['dynamic'])
    joblib.dump(bundle,out/'selected_candidate.joblib')
    recent=data[data.date>='20260818']
    with threadpool_limits(limits=2): pred=bundle['model'].predict(recent[bundle['features']])
    if best['name']=='risk_scaled': pred=pred*recent.vol5.clip(lower=.0005).to_numpy()
    tests={}
    for label,cost,delay in [('base',.0009,1),('double_cost',.0018,1),('delay_2min',.0018,2),('cost_36bp',.0036,1)]:
        metric,trades,daily=simulate(recent,pred,bundle['horizon'],bundle['threshold'],
            bundle['dynamic'],cost=cost,delay=delay,detail=True)
        tests[label]=metric
        trades.to_csv(out/f'{label}_trades.csv',index=False)
        daily.to_csv(out/f'{label}_daily.csv',index=False)
    report=dict(selected=best,experiments=rows,recent=tests,feature_columns=BASE+EXTRA,
        warning='Recent dates are retrospective audit; no independent fresh holdout claim.',
        deployment_approved=False)
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'selected':best,'recent':tests},indent=2),flush=True)


if __name__=='__main__': main()
