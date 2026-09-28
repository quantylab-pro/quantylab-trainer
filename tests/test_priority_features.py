import pandas as pd
from quantylab.trainer.etf_single_intraday.priority_features import depth_features,join_nav


def test_depth_levels_and_microprice():
    q={'sel_fpr_bid':101,'buy_fpr_bid':99,'sel_fpr_req':10,'buy_fpr_req':30}
    for n in range(2,11):
        q[f'sel_{n}th_pre_req']=10;q[f'buy_{n}th_pre_req']=30
    r=depth_features(q)
    assert all(r[f'imbalance_{n}']==.5 for n in (1,3,5,10))
    assert abs(r['microprice_gap_bps']-50)<1e-8


def test_nav_never_backdated_or_special_time_used():
    d=pd.DataFrame({'code':['102110']*3,'at':[
        '2026-09-14T09:00:00+09:00','2026-09-14T09:01:00+09:00','2026-09-14T09:04:00+09:00']})
    records=[dict(request={'stk_cd':'102110'},received_at='2026-09-14T09:00:30+09:00',
        rows=[{'tm':'888888','nav':1,'close_pric':100},{'tm':'090000','nav':100,'close_pric':101}])]
    r=join_nav(d,records)
    assert r.nav_available.tolist()==[False,True,False]
    assert abs(r.nav_gap_bps.iloc[1]-100)<1e-8


def test_after_close_nav_is_not_intraday_feature():
    d=pd.DataFrame({'code':['102110'],'at':['2026-09-14T15:00:00+09:00']})
    r=join_nav(d,[dict(request={'stk_cd':'102110'},received_at='2026-09-14T16:10:00+09:00',
        rows=[{'tm':'153000','nav':100,'close_pric':101}])])
    assert not r.nav_available.any()
