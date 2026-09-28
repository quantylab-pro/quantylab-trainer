import unittest
import numpy as np
import pandas as pd
from quantylab.trainer.etf_single_intraday.minute_engine import features,simulate,BASE,EXTRA


def bars(days=7):
    rows=[]
    for day in range(1,days+1):
        for minute in range(540,576):
            rows.append(dict(code='102110',date=f'202606{day:02d}',time=f'{minute//60:02d}{minute%60:02d}',
                open=100.,high=100.,low=100.,close=100.,volume=10000.))
    return pd.DataFrame(rows)


class MinuteTests(unittest.TestCase):
    def test_label_matches_next_open_exit(self):
        raw=bars(1)
        raw.loc[19,'open']=102.
        d=features(raw)
        self.assertAlmostEqual(d.loc[15,'y3'],.02)

    def test_future_bars_do_not_change_current_features(self):
        raw=bars()
        full=features(raw)
        prefix=features(raw.iloc[:-10])
        keys=['code','date','time']
        joined=prefix[keys+BASE+EXTRA].merge(full[keys+BASE+EXTRA],on=keys,suffixes=('_p','_f'))
        for c in BASE+EXTRA:
            np.testing.assert_allclose(joined[c+'_p'],joined[c+'_f'])
        self.assertEqual(full.iloc[-1].same_time_volume,1.)

    def test_next_open_cost_and_single_position(self):
        d=features(bars(1)); pred=np.zeros(len(d)); pred[15]=1
        result,t,_=simulate(d,pred,3,.001,cost=.002,detail=True)
        self.assertEqual(len(t),1)
        self.assertEqual(t.entry_minute.iloc[0],556)
        self.assertEqual(t.exit_minute.iloc[0],559)
        self.assertAlmostEqual(t.pnl.iloc[0],-20.)
        self.assertEqual(result['truncated_exits'],0)

    def test_intraday_drawdown_sees_unrealized_loss(self):
        d=features(bars(1)); d.loc[17,'close']=90
        pred=np.zeros(len(d));pred[15]=1
        result,t,_=simulate(d,pred,5,.001,cost=0,detail=True)
        self.assertEqual(t.pnl.iloc[0],0)
        self.assertLess(result['worst_intraday_drawdown_pct'],0)

    def test_session_deadline_overrides_signal_delay(self):
        for delay in (1, 2):
            for missing_deadline in (False, True):
                with self.subTest(delay=delay, missing=missing_deadline):
                    minutes=np.arange(880,921)
                    if missing_deadline:
                        minutes=minutes[minutes != 915]
                    d=pd.DataFrame(dict(code='102110',date='20260914',
                        minute=minutes,open=100.,close=100.,capacity=100.,
                        vol5=.001,vwap_gap=0.))
                    pred=np.zeros(len(d));pred[15]=1
                    result,t,_=simulate(d,pred,30,.001,delay=delay,detail=True)
                    self.assertEqual(len(t),1)
                    self.assertEqual(t.exit_minute.iloc[0],916 if missing_deadline else 915)
                    self.assertEqual(t.reason.iloc[0],'session')
                    self.assertEqual(result['truncated_exits'],0)

    def test_session_deadline_overrides_pending_exit(self):
        minutes=np.arange(880,921)
        d=pd.DataFrame(dict(code='102110',date='20260914',minute=minutes,
            open=100.,close=100.,capacity=100.,vol5=.001,vwap_gap=0.))
        pred=np.zeros(len(d));pred[15]=1
        _,t,_=simulate(d,pred,18,.001,delay=2,detail=True)
        self.assertEqual(t.exit_minute.iloc[0],915)
        self.assertEqual(t.reason.iloc[0],'session')


if __name__=='__main__': unittest.main()
