"""분봉 단일 ETF 트레이딩 환경.

행동은 현재 ETF의 목표 포지션 비율 하나이며, 여러 ETF의 비중 벡터를
출력하지 않는다. 기존 단일 ETF PPO 환경의 거래/보상 로직을 재사용하되
분봉 데이터셋에 필요한 스키마 검증과 명확한 이름을 제공한다.
"""

import pandas as pd
import numpy as np

from ..etf_single_swing.environment import SwingTradingEnvironment


class IntradayTradingEnvironment(SwingTradingEnvironment):
    """분봉 OHLCV를 한 ETF씩 순차 처리하는 환경.

    ``training_data[t - 1]``을 관찰하고 ``t``번째 분봉의 시가에
    목표 포지션을 조정한 뒤 종가로 평가하므로, 기존 학습 루프와 동일하게
    미래 분봉을 참조하지 않는다.
    """

    DATA_FREQUENCY = '1min'

    REQUIRED_COLUMNS = {
        'etf_code', 'date', 'time',
        'open', 'high', 'low', 'close', 'volume',
    }

    def __init__(self, env_data: pd.DataFrame, training_data, *args, **kwargs):
        missing = self.REQUIRED_COLUMNS.difference(env_data.columns)
        if missing:
            raise ValueError(
                '분봉 환경 데이터에 필요한 컬럼이 없습니다: '
                + ', '.join(sorted(missing))
            )

        # 분봉은 시계열 순서와 ETF/session 경계가 보상 계산의 전제다.
        # 조용히 정렬하거나 중복을 허용하면 look-ahead와 잘못된 장마감
        # 청산이 섞일 수 있으므로 데이터셋 오류를 학습 전에 드러낸다.
        ordered = env_data[['etf_code', 'date', 'time']].copy().reset_index(drop=True)
        ordered['etf_code'] = ordered['etf_code'].astype(str).str.zfill(6)
        ordered['date'] = ordered['date'].astype(str).str.zfill(8)
        ordered['time'] = ordered['time'].astype(str).str.zfill(4)
        if ordered.duplicated().any():
            raise ValueError('분봉 환경 데이터에 중복 ETF/날짜/시간 행이 있습니다.')
        expected_order = ordered.sort_values(
            ['etf_code', 'date', 'time'], kind='stable'
        ).index.to_numpy()
        if not np.array_equal(expected_order, np.arange(len(ordered))):
            raise ValueError(
                '분봉 환경 데이터는 etf_code/date/time 오름차순으로 정렬되어야 합니다.'
            )
        if ((ordered['time'].astype(int) < 900) |
                (ordered['time'].astype(int) > 1530)).any():
            raise ValueError('분봉 환경 데이터에 정규 장외 시간 행이 포함되어 있습니다.')

        # 1분봉의 매 바 리밸런싱은 수수료 누적으로 학습이 붕괴하므로
        # intraday 전용 기본값을 사용한다. CLI에서 명시한 값은 보존한다.
        kwargs.setdefault('rebalance_interval_bars', 15)
        kwargs.setdefault('max_position_change', 0.25)
        kwargs.setdefault('turnover_penalty_scale', 0.5)
        kwargs.setdefault('flat_at_session_end', True)

        for key, default in (
            ('rebalance_interval_bars', 15),
            ('max_position_change', 0.25),
            ('turnover_penalty_scale', 0.5),
            ('flat_at_session_end', True),
        ):
            if kwargs.get(key) is None:
                kwargs[key] = default

        super().__init__(env_data=env_data, training_data=training_data,
                         *args, **kwargs)

    def step(self, target_ratio):
        """Reward reconciles with marked equity, including close-to-next-open gaps.

        Costs already reduce equity; do not charge them a second time in reward.
        The daily environment's terminal CAGR treats bars as days, so it is not
        used for this intraday objective.
        """
        boundary = self.tick in self._etf_boundaries
        previous_value = self.initial_balance if boundary else self.portfolio_value
        previous_peak = self.initial_balance if boundary else self.peak_portfolio_value
        prior_wins = 0 if boundary else self.consecutive_wins
        prior_losses = 0 if boundary else self.consecutive_losses
        previous_dd = max(0., 1. - previous_value / max(previous_peak, 1e-8))
        if boundary:
            # Previous ETF's features cannot drive an entry in the new ETF.
            target_ratio = 0.
        state, _, done, info = super().step(target_ratio)
        net_return = self.portfolio_value / max(previous_value, 1e-8) - 1.
        reward = net_return * self.reward_scale
        if net_return < 0:
            reward *= self.loss_aversion
        current_dd = max(0., 1. - self.portfolio_value / max(self.peak_portfolio_value, 1e-8))
        incremental_dd = max(0., current_dd - max(previous_dd, self.drawdown_penalty_threshold))
        reward -= self.drawdown_penalty_scale * incremental_dd
        reward = float(np.clip(reward, -self.reward_clip, self.reward_clip))
        self.daily_returns[-1] = net_return
        if net_return > 0:
            self.consecutive_wins, self.consecutive_losses = prior_wins + 1, 0
        elif net_return < 0:
            self.consecutive_wins, self.consecutive_losses = 0, prior_losses + 1
        else:
            self.consecutive_wins, self.consecutive_losses = prior_wins, prior_losses
        self.history[-1].update(reward=reward, prev_portfolio_value=previous_value)
        info['net_return'] = net_return
        if not done:
            state = self._get_state()
        return state, reward, done, info

    def get_stats(self):
        """분봉 intraday 기준으로 세션 수익률과 B&H를 계산한다."""
        stats = super().get_stats()

        if 'date' not in self.env_data.columns:
            return stats

        group_columns = ['date']
        if 'etf_code' in self.env_data.columns:
            group_columns = ['etf_code', 'date']

        sessions = self.env_data.groupby(group_columns, sort=False).agg(
            session_open=('open', 'first'),
            session_close=('close', 'last'),
        )
        session_bnh = (
            sessions['session_close'] / sessions['session_open'].clip(lower=1e-8) - 1.0
        ).to_numpy(dtype=float)

        if session_bnh.size:
            if 'etf_code' in self.env_data.columns and self.env_data['etf_code'].nunique() > 1:
                bnh_by_code = pd.Series(session_bnh, index=sessions.index).groupby(level=0).apply(
                    lambda values: np.prod(1.0 + values.to_numpy()) - 1.0
                )
                intraday_bnh = float(bnh_by_code.mean() * 100.0)
            else:
                intraday_bnh = float((np.prod(1.0 + session_bnh) - 1.0) * 100.0)
        else:
            intraday_bnh = 0.0

        actual_sessions = np.asarray(self.session_returns, dtype=float)
        if actual_sessions.size > 1:
            win_rate = float((actual_sessions > 0).mean() * 100.0)
            wins = actual_sessions[actual_sessions > 0]
            losses = actual_sessions[actual_sessions < 0]
            avg_win = float(wins.mean() * 100.0) if wins.size else 0.0
            avg_loss = float(losses.mean() * 100.0) if losses.size else 0.0
            sharpe = float(actual_sessions.mean() / (actual_sessions.std() + 1e-8) * np.sqrt(252.0))
        else:
            win_rate = 0.0
            avg_win = 0.0
            avg_loss = 0.0
            sharpe = 0.0

        stats.update({
            'bnh_return': intraday_bnh,
            'intraday_bnh_return': intraday_bnh,
            'excess_bnh': float(stats['profit_rate'] - intraday_bnh),
            'num_sessions': int(len(sessions)),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'sharpe_ratio': sharpe,
        })
        return stats


# 기존 단일 ETF 코드와의 명시적 호환 별칭.
SingleIntradayEnvironment = IntradayTradingEnvironment

__all__ = ['IntradayTradingEnvironment', 'SingleIntradayEnvironment']
