"""Stock-specific swing environments.

The PPO implementation is shared with the ETF single-swing model, but the
asset identity and transaction assumptions are stock-specific here.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from ..etf_single_swing.environment import SwingTradingEnvironment


class StockSwingTradingEnvironment(SwingTradingEnvironment):
    """One stock over a contiguous daily history.

    The legacy environment uses ``etf_code`` internally for segment boundaries.
    We preserve the original ``stock_code`` column and provide that internal
    alias so the accounting code remains identical while the public product
    contract is stock-specific.
    """

    def __init__(self, env_data: pd.DataFrame, training_data: np.ndarray, **kwargs):
        data = env_data.copy().reset_index(drop=True)
        if "stock_code" not in data.columns:
            raise ValueError("stock 데이터에 stock_code 컬럼이 필요합니다.")
        if "etf_code" not in data.columns:
            data["etf_code"] = data["stock_code"].astype(str).str.zfill(6)
        super().__init__(data, training_data, **kwargs)

    @property
    def stock_code(self) -> str:
        return str(self.env_data["stock_code"].iloc[0]).zfill(6)

    def get_stats(self) -> dict:
        stats = super().get_stats()
        stats["stock_code"] = self.stock_code
        stats["asset_type"] = "stock"
        return stats


class RandomStockSwingEnvironment(StockSwingTradingEnvironment):
    """Sample a stock and a random contiguous window at every reset.

    A shared single-stock policy should see many assets during one run. This
    environment avoids running hundreds of independent PPO trainers while
    keeping each rollout causal and contiguous in calendar time.
    """

    def __init__(
        self,
        sequences: Iterable[tuple[str, pd.DataFrame, np.ndarray]],
        window_days: int = 756,
        randomize: bool = True,
        seed: int = 42,
        **kwargs,
    ):
        self.sequences = list(sequences)
        if not self.sequences:
            raise ValueError("학습 가능한 stock sequence가 없습니다.")
        self.window_days = max(int(window_days), 0)
        self.randomize = bool(randomize)
        self.rng = np.random.default_rng(seed)
        self._sequence_cursor = 0
        self._active_code = ""

        _, first_env, first_features = self._pick_sequence()
        super().__init__(first_env, first_features, **kwargs)

    def _pick_sequence(self):
        if self.randomize:
            index = int(self.rng.integers(0, len(self.sequences)))
        else:
            index = self._sequence_cursor % len(self.sequences)
            self._sequence_cursor += 1

        code, env_data, training_data = self.sequences[index]
        env_data = env_data.reset_index(drop=True)
        training_data = np.asarray(training_data, dtype=np.float32)

        # Keep at least one previous observation plus one trade day.
        max_window = max(len(env_data), 2)
        window = self.window_days if self.window_days > 0 else max_window
        window = min(max(window, 2), max_window)
        if len(env_data) > window:
            start = int(self.rng.integers(0, len(env_data) - window + 1)) if self.randomize else 0
            env_data = env_data.iloc[start:start + window].reset_index(drop=True)
            training_data = training_data[start:start + window]

        self._active_code = str(code).zfill(6)
        return self._active_code, env_data, training_data

    def reset(self) -> np.ndarray:
        # SwingTradingEnvironment.__init__ calls reset(), so this method also
        # handles initial construction after the sequence state is available.
        if hasattr(self, "sequences"):
            _, env_data, training_data = self._pick_sequence()
            self.env_data = env_data
            self.training_data = training_data
            self.total_ticks = len(env_data)
            self.num_steps = self.total_ticks - 1
            self.num_features = training_data.shape[1] + self.PORTFOLIO_FEATURE_NUM
            self._etf_boundaries = set()
        return super().reset()
