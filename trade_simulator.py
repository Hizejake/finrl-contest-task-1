"""CryptoTradingEnv extracted from the Kaggle DAgger notebook."""

from __future__ import annotations

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class CryptoTradingEnv(gym.Env):
    """A simple BTC trading environment with a lookback window."""

    def __init__(self, data: pd.DataFrame, initial_balance: float = 10000.0, transaction_cost: float = 0.001, lookback_window: int = 60):
        super().__init__()
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.lookback_window = lookback_window
        # feature columns are expected to be provided globally as ``all_feature_columns``
        global all_feature_columns
        self.feature_columns = all_feature_columns

        self.action_space = spaces.Discrete(3)  # 0 = hold, 1 = buy, 2 = sell
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.lookback_window * len(self.feature_columns),),
            dtype=np.float32,
        )
        self.reset()

    def reset(self, seed: int | None = None):
        super().reset(seed=seed)
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.btc_held = 0.0
        self.net_worth_history = [self.initial_balance]
        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        if self.current_step >= len(self.data):
            self.current_step = len(self.data) - 1
        frame = self.data.iloc[self.current_step - self.lookback_window : self.current_step]
        obs = frame[self.feature_columns].values.flatten()
        return obs.astype(np.float32)

    def step(self, action: int):
        if self.current_step >= len(self.data) - 1:
            return self._get_observation(), 0.0, True, True, {}

        current_price = self.data.loc[self.current_step, "unscaled_midpoint"]

        if action == 1 and self.balance > 0:
            btc_to_buy = (self.balance / current_price) * (1 - self.transaction_cost)
            self.btc_held += btc_to_buy
            self.balance = 0
        elif action == 2 and self.btc_held > 0:
            self.balance += self.btc_held * current_price * (1 - self.transaction_cost)
            self.btc_held = 0

        self.current_step += 1
        next_price = self.data.loc[self.current_step, "unscaled_midpoint"]
        portfolio_value = self.balance + self.btc_held * next_price

        reward = (portfolio_value - self.net_worth_history[-1]) / self.initial_balance
        self.net_worth_history.append(portfolio_value)

        done = self.current_step >= len(self.data) - 1
        return self._get_observation(), reward, done, False, {"portfolio_value": portfolio_value}
