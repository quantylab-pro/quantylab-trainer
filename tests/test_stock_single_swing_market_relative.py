"""Checks for causal features and next-open alpha labels."""

import numpy as np
import pandas as pd

from scripts.train_stock_single_swing_market_relative import make_features, make_target, evaluate


def _sample():
    dates = pd.date_range("2024-01-01", periods=145).strftime("%Y%m%d").tolist()
    close = np.linspace(100, 160, len(dates))
    stock = pd.DataFrame({"date": dates, "stock_code": ["000001"] * len(dates),
                          "market_name": ["kospi"] * len(dates),
                          "open": close, "close": close, "high": close * 1.01,
                          "low": close * 0.99, "volume": [1000] * len(dates)})
    index = pd.DataFrame({"date": dates, "market": ["kospi"] * len(dates),
                          "open": np.linspace(200, 220, len(dates)),
                          "close": np.linspace(200, 220, len(dates))})
    return stock, index


def test_features_do_not_change_when_future_rows_are_added():
    stock, index = _sample()
    early, names = make_features(stock.iloc[:135].copy(), index.iloc[:135].copy())
    full, full_names = make_features(stock, index)
    assert names == full_names
    np.testing.assert_allclose(early, full[:135], equal_nan=True)


def test_target_starts_at_next_open_and_uses_matched_index():
    stock, index = _sample()
    target, end_dates = make_target(stock, index, 5)
    expected = np.log(stock.open.iloc[6] / stock.open.iloc[1]) - np.log(index.open.iloc[6] / index.open.iloc[1])
    assert np.isclose(target[0], expected)
    assert end_dates[0] == stock.date.iloc[6]
    assert np.isnan(target[-1])


def test_market_fallback_is_index_return_without_stock_signal():
    stock, index = _sample()
    result = evaluate(stock, np.full(len(stock), -1.0), index,
                      stock.date.iloc[0], stock.date.iloc[-1])
    assert np.isclose(result["mean_excess_percentage_points"], 0.0, atol=1e-8)
    assert result["mean_stock_weight"] == 0.0
