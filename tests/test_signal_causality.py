import unittest
import numpy as np
import pandas as pd
from quantylab.trainer.etf_single_intraday.robust_experiments import ranked_prediction
from quantylab.trainer.intraday_features import _build_code_features


class CausalityTests(unittest.TestCase):
    def test_completed_session_returns_use_full_lags(self):
        prices = [100., 110., 120., 130., 140., 150., 900.]
        frame = pd.DataFrame({'code': ['102110'] * 7,
            'date': [f'202609{i:02d}' for i in range(1, 8)], 'time': ['0900'] * 7,
            'open': prices, 'high': prices, 'low': prices, 'close': prices,
            'volume': [100] * 7})
        result = _build_code_features(frame)
        self.assertAlmostEqual(result.prior_day_ret_3.iloc[-1], 150 / 120 - 1)
        self.assertAlmostEqual(result.prior_day_ret_5.iloc[-1], 150 / 100 - 1)
        changed = frame.copy()
        changed.loc[6, 'close'] = 1.
        other = _build_code_features(changed)
        self.assertEqual(result.prior_day_ret_3.iloc[-1], other.prior_day_ret_3.iloc[-1])

    def test_later_signal_cannot_remove_morning_selection(self):
        frame = pd.DataFrame({'date': ['20260901'] * 3,
                              'time': ['0930', '0930', '1400']})
        prediction = np.array([.01, .02, .5])
        before = ranked_prediction(frame.iloc[:2], prediction[:2], 1)
        after = ranked_prediction(frame, prediction, 1)
        np.testing.assert_array_equal(before, after[:2])
        self.assertEqual(after[1], .02)


if __name__ == '__main__':
    unittest.main()
