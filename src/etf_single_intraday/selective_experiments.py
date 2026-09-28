"""Cost-aware classification and intraday regime experts; retrospective research."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from .minute_engine import BASE, EXTRA, simulate
from .minute_experiments import FOLDS


def enrich(data):
    d=data.copy()
    d['relative_strength']=d.ret5-d.market_ret5
    d['risk_ret']=d.ret5/d.vol5.clip(lower=.0005)
    d['risk_vwap']=d.vwap_gap/d.vol15.clip(lower=.0005)
    d['breadth']=d.groupby(['date','time']).ret5.transform(lambda x:(x>0).mean())
    d['dispersion']=d.groupby(['date','time']).ret5.transform('std').fillna(0)
    return d


def payoff_predict(model, x):
    probability=model['probability'].predict_proba(x)[:,1]
    return probability*model['gain'].predict(x)+(1-probability)*model['loss'].predict(x)


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--output',required=True)
    p.add_argument('--payoff',action='store_true')
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    d=enrich(pd.read_pickle(args.cache))
    cols=BASE+EXTRA+['relative_strength','risk_ret','risk_vwap','breadth','dispersion']
    rows=[]
    for horizon in (5,15,30):
        for kind in (('payoff',) if args.payoff else ('cost_classifier','session_experts')):
            thresholds=(.52,.58,.64) if kind=='cost_classifier' else (.0018,.003)
            metrics={t:[] for t in thresholds}
            for end,start,stop in FOLDS:
                tr=d[(d.date<=end)&(d.minute%5==0)&d[f'y{horizon}'].notna()&d.minute.between(555,899)]
                va=d[d.date.between(start,stop)]
                with threadpool_limits(limits=2):
                    params=dict(max_iter=100,max_leaf_nodes=7,min_samples_leaf=300,
                        l2_regularization=30,early_stopping=False,random_state=42)
                    if kind=='cost_classifier':
                        model=HistGradientBoostingClassifier(**params)
                        model.fit(tr[cols],tr[f'y{horizon}']>.0018)
                        pred=model.predict_proba(va[cols])[:,1]
                    elif kind=='payoff':
                        y=tr[f'y{horizon}'].clip(-.03,.03)
                        positive=y>.0018
                        model={'probability':HistGradientBoostingClassifier(**params).fit(tr[cols],positive),
                            'gain':HistGradientBoostingRegressor(**params).fit(tr.loc[positive,cols],y[positive]),
                            'loss':HistGradientBoostingRegressor(**params).fit(tr.loc[~positive,cols],y[~positive])}
                        pred=payoff_predict(model,va[cols])
                    else:
                        model={};pred=np.zeros(len(va))
                        for morning in (False,True):
                            subset=tr[(tr.minute<690)==morning]
                            mask=((va.minute<690)==morning).to_numpy()
                            model[morning]=HistGradientBoostingRegressor(**params).fit(subset[cols],subset[f'y{horizon}'].clip(-.03,.03))
                            pred[mask]=model[morning].predict(va.loc[mask,cols])
                for t in thresholds:
                    metrics[t].append(simulate(va,pred,horizon,t,cost=.0018))
            joblib.dump(dict(model=model,features=cols,horizon=horizon,kind=kind,training_end='20260731',deployment_approved=False),out/f'{kind}_{horizon}.joblib')
            for t,ms in metrics.items():
                r=np.array([m['mean_return_pct'] for m in ms])
                rows.append(dict(kind=kind,horizon=horizon,threshold=t,folds=ms,
                    score=float(r.mean()-.5*r.std()),eligible=all(m['trades']>=20 and m['truncated_exits']==0 for m in ms),
                    positive_all_folds=bool((r>0).all())))
            (out/'progress.json').write_text(json.dumps(rows,indent=2))
            print('completed',kind,horizon,flush=True)
    eligible=[r for r in rows if r['eligible']]
    best=max(eligible or rows,key=lambda r:r['score'])
    bundle=joblib.load(out/f"{best['kind']}_{best['horizon']}.joblib")
    bundle['threshold']=best['threshold'];joblib.dump(bundle,out/'selected_candidate.joblib')
    recent=d[d.date>='20260818'];model=bundle['model']
    with threadpool_limits(limits=2):
        if best['kind']=='cost_classifier': pred=model.predict_proba(recent[cols])[:,1]
        elif best['kind']=='payoff': pred=payoff_predict(model,recent[cols])
        else:
            pred=np.zeros(len(recent))
            for morning in (False,True):
                mask=((recent.minute<690)==morning).to_numpy()
                pred[mask]=model[morning].predict(recent.loc[mask,cols])
    tests={}
    for label,cost,delay in [('cost9',.0009,1),('cost18',.0018,1),('cost36',.0036,1),('delay2',.0018,2)]:
        m,t,days=simulate(recent,pred,best['horizon'],best['threshold'],cost=cost,delay=delay,detail=True)
        tests[label]=m;t.to_csv(out/f'{label}_trades.csv',index=False);days.to_csv(out/f'{label}_daily.csv',index=False)
    report=dict(selected=best,experiments=rows,recent=tests,deployment_approved=False,
        warning='Repeated retrospective audit, not independent holdout. Probability of positive net return is not expected profit.')
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report['recent']),flush=True)


if __name__=='__main__':main()
