"""Point-in-time checks for the single-stock swing retraining path."""

import numpy as np
import pandas as pd

from scripts.train_stock_single_swing_regime import evaluate, forward_target, market_features


def test_market_context_uses_current_close_and_target_uses_future_close():
    dates = pd.date_range("2024-01-01", periods=23).strftime("%Y%m%d").tolist()
    index = pd.DataFrame({"market": ["kospi"] * len(dates) + ["kosdaq"] * len(dates),
                          "date": dates * 2, "close": list(range(100, 123)) + [100] * len(dates)})
    stock = pd.DataFrame({"date": dates, "stock_code": ["000001"] * len(dates),
                          "market_name": ["kospi"] * len(dates), "close": list(range(100, 123))})
    features, _, _ = market_features(stock, index)
    assert np.isclose(features[20, 1], 120 / 100 - 1)
    target, future_date = forward_target(stock, 2)
    assert np.isclose(target[0], np.log(102 / 100))
    assert future_date[0] == dates[2]
    assert np.isnan(target[-1])


def test_single_account_ignores_opening_jump_before_fill():
    dates = pd.date_range("2024-01-01", periods=32).strftime("%Y%m%d").tolist()
    frame = pd.DataFrame({"date": dates, "stock_code": ["000001"] * 32,
                          "market_name": ["kospi"] * 32,
                          "open": [100] + [200] * 31})
    result = evaluate(frame, np.ones(32), np.zeros(32), dates[0], dates[-1], "binary")
    assert result["mean_return"] < 0
    assert result["mean_return"] > -0.01
