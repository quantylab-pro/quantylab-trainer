"""Causality checks for the stock swing research training path."""

import numpy as np
import pandas as pd

from scripts.train_stock_swing_alpha_candidates import make_target, portfolio_eval


def test_market_relative_target_uses_future_days_only():
    frame = pd.DataFrame({
        "date": ["20240101", "20240102", "20240103", "20240104"] * 2,
        "stock_code": ["000001"] * 4 + ["000002"] * 4,
        "market_name": ["kospi"] * 8,
        "close": [100, 110, 121, 133.1, 200, 200, 200, 200],
    })
    target, future_date = make_target(frame, 2)
    assert np.isclose(target[0], np.log(1.21) / 2)
    assert future_date[0] == "20240103"
    assert np.isnan(target[2])


def test_portfolio_fills_after_signal_close():
    dates = ["20240101", "20240102", "20240103", "20240104"]
    frame = pd.DataFrame({
        "date": dates * 2,
        "stock_code": ["000001"] * 4 + ["000002"] * 4,
        "open": [100, 200, 220, 242, 100, 100, 100, 100],
    })
    result = portfolio_eval(frame, np.array([1.0] * 4 + [-1.0] * 4),
                            ["000001", "000002"], dates[0], dates[-1], 1)
    # The opening jump from 100 to 200 occurs before the first fill.
    assert 0.15 < result["return"] < 0.22
    assert result["days"] == 2
