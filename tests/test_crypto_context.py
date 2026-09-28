import pandas as pd
from quantylab.trainer.etf_single_intraday.global_context_experiments import crypto_features


def test_crypto_features_are_closed_and_delayed_and_prefix_invariant():
    raw=pd.DataFrame({'bar_close_utc':pd.date_range('2026-01-01',periods=20,freq='min',tz='UTC'),
        'close':range(100,120),'taker_quote':[50]*20,'quote_volume':[100]*20})
    full=crypto_features(raw);prefix=crypto_features(raw.iloc[:18])
    pd.testing.assert_frame_equal(full.iloc[:18],prefix)
    assert (full.available_at-raw.bar_close_utc).eq(pd.Timedelta(seconds=60)).all()


def test_gap_does_not_become_one_minute_return():
    raw=pd.DataFrame({'bar_close_utc':pd.to_datetime(['2026-01-01T00:00Z','2026-01-01T00:10Z']),
        'close':[100,200],'taker_quote':[1,1],'quote_volume':[2,2]})
    assert pd.isna(crypto_features(raw).ret1.iloc[-1])
