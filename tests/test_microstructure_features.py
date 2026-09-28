import pandas as pd
from quantylab.trainer.etf_single_intraday.microstructure_features import join_quotes


def test_only_received_nonstale_quotes_are_available():
    decisions=pd.DataFrame(dict(code=['102110']*3, decision_at=[
        '2026-09-14T09:00:00+09:00','2026-09-14T09:01:00+09:00','2026-09-14T09:04:00+09:00']))
    snapshots=pd.DataFrame([dict(code='102110',quote_received_at='2026-09-14T09:00:30+09:00',
        quote_fresh=True,spread_bps=2.,imbalance=.2)])
    result=join_quotes(decisions,snapshots)
    assert result.quote_available.tolist()==[False,True,False]
