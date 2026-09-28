"""Snapshot quality and exploratory forward-quote associations, not fill backtests."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def number(x):
    try:return abs(float(str(x).replace(',','')))
    except (ValueError,TypeError):return np.nan


def extract(record):
    q=record.get('quote',{})
    at=max(pd.Timestamp(record['quote_received_at']),pd.Timestamp(record.get('intensity_received_at',record['quote_received_at'])))
    ask,bid=number(q.get('sel_fpr_bid')),number(q.get('buy_fpr_bid'))
    aq,bq=number(q.get('tot_sel_req')),number(q.get('tot_buy_req'))
    intensity=[]
    for row in record.get('intensity',[]):
        clock=str(row.get('체결시간','')).zfill(6)
        if len(clock)==6 and clock.isdigit():
            seconds=int(clock[:2])*3600+int(clock[2:4])*60+int(clock[4:])
            if seconds<=at.hour*3600+at.minute*60+at.second:
                intensity.append((seconds,row))
    latest=max(intensity,key=lambda x:x[0]) if intensity else None
    return dict(code=record['code'],at=at,ask=ask,bid=bid,mid=(ask+bid)/2,
        spread_bps=record.get('spread_bps'),imbalance=record.get('imbalance'),
        depth_imbalance=(bq-aq)/(bq+aq) if bq+aq>0 else np.nan,
        fresh=record.get('quote_fresh',False),
        intensity=number(latest[1].get('체결강도')) if latest else np.nan,
        intensity_age=(at.hour*3600+at.minute*60+at.second-latest[0]) if latest else np.nan)


def main():
    p=argparse.ArgumentParser();p.add_argument('--roots',nargs='+',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    rows=[];errors=0
    for root in args.roots:
        for path in Path(root).glob('*/*/*.json'):
            try:
                r=json.loads(path.read_text())
                if r.get('status')=='captured':rows.append(extract(r))
                else:errors+=1
            except (ValueError,KeyError):errors+=1
    d=pd.DataFrame(rows).sort_values('at').drop_duplicates(['code','at'])
    d.to_csv(out/'snapshots.csv',index=False)
    report=dict(snapshots=len(d),errors=errors,dates=sorted(d['at'].dt.strftime('%Y%m%d').unique()),
        codes=int(d.code.nunique()),fresh_fraction=float(d.fresh.mean()),
        spread_bps=d.spread_bps.quantile([.1,.5,.9,.99]).to_dict(),
        intensity_age_seconds=d.intensity_age.quantile([.5,.9,.99]).to_dict())
    valid=d[d.fresh & d.ask.ge(d.bid)&d.bid.gt(0)].copy()
    valid['target_at']=valid['at']+pd.Timedelta(minutes=5)
    future=valid[['code','at','bid','mid']].rename(columns={'at':'exit_at','bid':'exit_bid','mid':'exit_mid'})
    paired=pd.merge_asof(valid.sort_values('target_at'),future.sort_values('exit_at'),
        left_on='target_at',right_on='exit_at',by='code',direction='forward',tolerance=pd.Timedelta(seconds=90))
    paired=paired[paired['at'].dt.date.eq(paired.exit_at.dt.date)].copy()
    paired['mid_return_bps']=(paired.exit_mid/paired.mid-1)*10000
    paired['quoted_net_bps']=(paired.exit_bid/paired.ask-1)*10000-2
    paired.loc[paired.intensity_age>120,'intensity']=np.nan
    report['forward_pairs']=len(paired)
    report['exploratory_correlations']={c:float(paired[c].corr(paired.mid_return_bps,method='spearman'))
        for c in ['imbalance','depth_imbalance','intensity','spread_bps']}
    report['mean_quoted_net_bps']=float(paired.quoted_net_bps.mean())
    report['warning']='Overlapping single-day samples; no independence or predictive benefit claim. Quotes are not guaranteed fills; 2bp fee is assumed, no additional impact.'
    paired.to_csv(out/'forward_pairs.csv',index=False)
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
