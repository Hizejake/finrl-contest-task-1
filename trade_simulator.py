"""Simple cryptocurrency trading environment for RL and DAgger."""

from __future__ import annotations

import numpy as np
import pandas as pd
import gymnasium as gym


class CryptoTradingEnv(gym.Env):
    """A minimal trading environment using OHLCV data and sentiment factor."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, data: pd.DataFrame, window_size: int = 60, fee: float = 0.001, initial_cash: float = 10000.0):
        super().__init__()
        self.data = data.reset_index(drop=True)
        self.window_size = window_size
        self.fee = fee
        self.initial_cash = initial_cash

        self.action_space = gym.spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell
        obs_shape = (window_size, self.data.shape[1] - 1)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)

        self._reset_internal_state()

    def _reset_internal_state(self):
        self.current_step = self.window_size
        self.cash = self.initial_cash
        self.coins = 0.0

    def _get_observation(self):
        obs = self.data.iloc[self.current_step - self.window_size : self.current_step]
        obs = obs.drop(columns=["timestamp"]).to_numpy(dtype=np.float32)
        return obs

    def reset(self, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self._reset_internal_state()
        observation = self._get_observation()
        return observation, {}

    def step(self, action: int):
        price = float(self.data.loc[self.current_step, "close"])
        done = self.current_step >= len(self.data) - 1

        if action == 1 and self.cash > 0:  # buy
            self.coins = (self.cash / price) * (1 - self.fee)
            self.cash = 0
        elif action == 2 and self.coins > 0:  # sell
            self.cash = (self.coins * price) * (1 - self.fee)
            self.coins = 0

        self.current_step += 1
        next_obs = self._get_observation()

        portfolio_value = self.cash + self.coins * price
        reward = portfolio_value - self.initial_cash
        info = {"portfolio_value": portfolio_value}

        return next_obs, reward, done, False, info

    def render(self):
        price = float(self.data.loc[self.current_step - 1, "close"])
        print(f"Step {self.current_step}: Price={price:.2f} Cash={self.cash:.2f} Coins={self.coins:.6f}")
