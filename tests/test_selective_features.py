import pandas as pd
from quantylab.trainer.etf_single_intraday.selective_experiments import enrich, payoff_predict
import numpy as np


def test_enrichment_does_not_use_future_minutes():
    d=pd.DataFrame(dict(code=['a','b','a','b'],date=['20260601']*4,
        time=['0900','0900','0901','0901'],ret5=[.01,-.01,.9,.8],
        market_ret5=[0,0,.85,.85],vol5=[.01]*4,vol15=[.01]*4,vwap_gap=[.01]*4))
    full=enrich(d);prefix=enrich(d.iloc[:2])
    pd.testing.assert_frame_equal(full.iloc[:2],prefix)
    assert prefix.breadth.tolist()==[.5,.5]
    assert prefix.relative_strength.tolist()==[.01,-.01]


def test_payoff_accounts_for_loss_size():
    class Probability:
        def predict_proba(self,x):return np.array([[.2,.8]])
    class Constant:
        def __init__(self,v):self.v=v
        def predict(self,x):return np.array([self.v])
    model={'probability':Probability(),'gain':Constant(.002),'loss':Constant(-.02)}
    assert payoff_predict(model,None)[0]<0
