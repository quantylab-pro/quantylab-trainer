from datetime import datetime
from quantylab.trainer.etf_single_intraday.auto_collect import market_window


def test_market_window():
    assert market_window(datetime(2026,9,14,9,0))
    assert market_window(datetime(2026,9,14,15,29))
    assert not market_window(datetime(2026,9,14,15,30))
    assert not market_window(datetime(2026,9,13,10,0))
    assert not market_window(datetime(2026,9,14,8,59))
