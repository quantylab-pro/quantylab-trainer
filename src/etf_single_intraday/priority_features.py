"""Depth and receipt-time NAV features; no historical backdating."""
import numpy as np
import pandas as pd
from .microstructure_audit import number


def depth_features(quote):
    result={};asks=[];bids=[]
    for level in range(1,11):
        asks.append(number(quote.get('sel_fpr_req' if level==1 else f'sel_{level}th_pre_req')))
        bids.append(number(quote.get('buy_fpr_req' if level==1 else f'buy_{level}th_pre_req')))
        if level in (1,3,5,10):
            a,b=np.array(asks),np.array(bids)
            valid=np.isfinite(a).all() and np.isfinite(b).all() and a.sum()+b.sum()>0
            result[f'imbalance_{level}']=(b.sum()-a.sum())/(b.sum()+a.sum()) if valid else np.nan
    ask,bid=number(quote.get('sel_fpr_bid')),number(quote.get('buy_fpr_bid'))
    total=asks[0]+bids[0]
    micro=(ask*bids[0]+bid*asks[0])/total if total>0 else np.nan
    result['microprice_gap_bps']=(micro/((ask+bid)/2)-1)*10000 if ask+bid>0 else np.nan
    return result


def join_nav(decisions,records):
    left=decisions.copy();left['_order']=range(len(left))
    left['at']=pd.to_datetime(left['at'],utc=True)
    rows=[]
    for record in records:
        received=pd.Timestamp(record['received_at'])
        for row in record.get('rows',[]):
            clock=str(row.get('tm','')).zfill(6)
            if len(clock)!=6 or not clock.isdigit():continue
            h,m,s=int(clock[:2]),int(clock[2:4]),int(clock[4:])
            if h>23 or m>59 or s>59:continue
            source=received.replace(hour=h,minute=m,second=s,microsecond=0)
            if not 0<=(received-source).total_seconds()<=120:continue
            nav=number(row.get('nav'));price=number(row.get('close_pric'))
            if not nav>0:continue
            rows.append(dict(code=record['request']['stk_cd'],nav_received=received,
                nav_gap_bps=(price/nav-1)*10000,source=source))
    if rows:
        right=pd.DataFrame(rows).sort_values('source').drop_duplicates(['code','nav_received'],keep='last')
        right['nav_received']=pd.to_datetime(right.nav_received,utc=True)
        left=pd.merge_asof(left.sort_values('at'),right.sort_values('nav_received'),
            left_on='at',right_on='nav_received',by='code',direction='backward',tolerance=pd.Timedelta(seconds=120))
        left['nav_available']=left.nav_received.notna()
    else:
        left['nav_gap_bps']=np.nan;left['nav_available']=False
    return left.sort_values('_order').drop(columns='_order')
