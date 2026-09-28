"""Opportunity diagnosis, market-factor decomposition, causal event selection."""
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
from .selective_experiments import enrich


def event_mask(d):
    # Fixed before evaluation; no future target or future daily rank.
    return (d.vwap_cross.abs()>0)&(d.same_time_volume>=1.5)&(d.seasonal_available>0)


def regressor():
    return HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=7,min_samples_leaf=200,
        l2_regularization=30,early_stopping=False,random_state=42)


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    d=enrich(pd.read_pickle(args.cache));cols=BASE+EXTRA+['relative_strength','risk_ret','risk_vwap','breadth','dispersion']
    market_cols=['market_ret5','breadth','dispersion','elapsed']
    d['event']=event_mask(d)
    diagnostic=[];results=[]
    for h in (5,15,30):
        ycol=f'y{h}'
        for name in ('raw','factor','event'):
            fold_metrics=[]
            for end,start,stop in FOLDS:
                tr=d[(d.date<=end)&(d.minute%5==0)&d[ycol].notna()&d.minute.between(555,899)]
                va=d[d.date.between(start,stop)]
                with threadpool_limits(limits=2):
                    target=tr[ycol].clip(-.03,.03)
                    if name=='factor':
                        # Future market average is a TRAINING label only.
                        market=tr.groupby(['date','time'])[ycol].transform('mean')
                        residual=regressor().fit(tr[cols],target-market)
                        market_data=tr.assign(market_target=market).drop_duplicates(['date','time'])
                        common=regressor().fit(market_data[market_cols],market_data.market_target)
                        model={'residual':residual,'common':common}
                        pred=residual.predict(va[cols])+common.predict(va[market_cols])
                    else:
                        if name=='event':tr=tr[tr.event];target=tr[ycol].clip(-.03,.03)
                        model=regressor().fit(tr[cols],target)
                        pred=model.predict(va[cols])
                        if name=='event':pred=np.where(va.event,pred,-1)
                active=va[ycol].notna()&va.minute.between(555,899)
                if name=='event':active=active&va.event
                observed=va.loc[active,ycol];forecast=pred[active.to_numpy()]
                diagnostic.append(dict(name=name,horizon=h,period=start,rows=len(observed),
                    actual_above18bp=float((observed>.0018).mean()),
                    predicted_above18bp=float((forecast>.0018).mean()),
                    forecast_std=float(np.std(forecast)),target_std=float(observed.std()),
                    correlation=float(pd.Series(forecast).corr(observed.reset_index(drop=True)))))
                fold_metrics.append(simulate(va,pred,h,.0018,cost=.0018))
            bundle=dict(model=model,name=name,horizon=h,features=cols,market_features=market_cols,
                threshold=.0018,training_end='20260731',deployment_approved=False)
            joblib.dump(bundle,out/f'{name}_{h}.joblib')
            returns=np.array([m['mean_return_pct'] for m in fold_metrics])
            results.append(dict(name=name,horizon=h,folds=fold_metrics,
                score=float(returns.mean()-.5*returns.std()),
                eligible=all(m['trades']>=20 and m['truncated_exits']==0 for m in fold_metrics)))
            print('completed',name,h,flush=True)
    pd.DataFrame(diagnostic).to_csv(out/'diagnostics.csv',index=False)
    candidates=[r for r in results if r['eligible']]
    best=max(candidates or results,key=lambda x:x['score'])
    b=joblib.load(out/f"{best['name']}_{best['horizon']}.joblib");recent=d[d.date>='20260818']
    with threadpool_limits(limits=2):
        if b['name']=='factor':pred=b['model']['residual'].predict(recent[cols])+b['model']['common'].predict(recent[market_cols])
        else:
            pred=b['model'].predict(recent[cols])
            if b['name']=='event':pred=np.where(recent.event,pred,-1)
    stress={}
    for cost in (.0009,.0018,.0036):
        metric,trades,daily=simulate(recent,pred,b['horizon'],.0018,cost=cost,detail=True)
        key=f'cost{round(cost*10000)}';stress[key]=metric
        trades.to_csv(out/f'{key}_trades.csv',index=False);daily.to_csv(out/f'{key}_daily.csv',index=False)
    report=dict(selected=best,experiments=results,recent=stress,deployment_approved=False,
        warning='Repeated retrospective evaluation. Event uses bar VWAP/volume, NOT NAV or depth. Factor is contemporaneous ETF-average return, not an external index hedge.')
    (out/'report.json').write_text(json.dumps(report,indent=2));joblib.dump(b,out/'selected_candidate.joblib')
    print(json.dumps({'selected':best,'recent':stress},indent=2))


if __name__=='__main__':main()
