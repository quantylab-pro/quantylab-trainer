import pandas as pd
from quantylab.trainer.etf_single_intraday.factor_event_experiments import event_mask


def test_event_uses_only_current_bar_features():
    d=pd.DataFrame({'vwap_cross':[1,0,-1,1],'same_time_volume':[2,2,1,2],
        'seasonal_available':[1,1,1,0],'y5':[999]*4})
    expected=[True,False,False,False]
    assert event_mask(d).tolist()==expected
    d['y5']=-999
    assert event_mask(d).tolist()==expected
