import unittest
import numpy as np
import pandas as pd
from quantylab.trainer.etf_single_intraday.environment import IntradayTradingEnvironment
from quantylab.trainer.intraday_features import build_intraday_features
from quantylab.trainer.etf_single_intraday.research import evaluate


class AccountingTests(unittest.TestCase):
    def test_execution_cost_and_cash_baseline(self):
        data=pd.DataFrame([dict(code='102110',date='20260901',entry=100.,exit_price=100.,capacity=1000)])
        cash=evaluate(data,np.array([0.]),.001,['102110'],['20260901'])
        trade=evaluate(data,np.array([1.]),.001,['102110'],['20260901'])
        self.assertEqual(cash['mean_return_pct'],0.)
        self.assertEqual(trade['trades'],1)
        self.assertAlmostEqual(trade['mean_return_pct'],-.0009)

    def test_gap_pnl_and_session_liquidation(self):
        bars = pd.DataFrame(dict(etf_code=['102110']*4, date=['20260901']*4,
                    time=['0900','0901','0902','0903'], open=[100,100,110,110],
                    close=[100,100,110,110], high=[100,100,110,110], low=[100,100,110,110],
                    volume=[10000]*4))
        env = IntradayTradingEnvironment(bars, np.zeros((4,2)), trading_fee=0.,
                    slippage=0., max_position_change=1., hold_threshold=.01,
                    rebalance_interval_bars=1, drawdown_penalty_scale=0., reward_scale=1.)
        env.step(1.)
        _, reward, _, _ = env.step(1.)
        self.assertAlmostEqual(reward, .1)
        env.step(1.)
        self.assertEqual(env.num_shares, 0)
        self.assertAlmostEqual(np.prod(1+np.array(env.daily_returns)), env.portfolio_value/env.initial_balance)

    def test_monotonic_rsi_and_future_independence(self):
        n=40
        bars=pd.DataFrame(dict(code=['102110']*n,date=['20260901']*n,
            time=[f'09{i:02d}' for i in range(n)],open=np.arange(n)+100.,
            close=np.arange(n)+100.,high=np.arange(n)+100.,low=np.arange(n)+100.,volume=[100]*n))
        full=build_intraday_features(bars)
        partial=build_intraday_features(bars.iloc[:20])
        self.assertEqual(full.rsi_14.iloc[19],100.)
        np.testing.assert_allclose(full.rsi_14.iloc[:20],partial.rsi_14)

    def test_ai_market_features_are_previous_day_only(self):
        bars = pd.DataFrame(dict(code=['102110']*4, date=['20260910']*4,
            time=['0900','0901','0902','0903'], open=[100.]*4, close=[100.]*4,
            high=[100.]*4, low=[100.]*4, volume=[100.]*4))
        features = build_intraday_features(bars)
        self.assertEqual(float(features.market_available.iloc[0]), 1.0)
        # 20260910's market analysis must not be visible on 20260910 bars.
        self.assertGreaterEqual(float(features.market_age_days.iloc[0]), 1.0 / 30.0)


if __name__ == '__main__':
    unittest.main()
