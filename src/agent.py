"""Agent, expert, and environment definitions for the trading pipeline."""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    DAGGER_DATASET_LIMIT,
    DAGGER_ITERATIONS,
    DAGGER_TRAJECTORY_LENGTH,
    INITIAL_BALANCE,
    LOOKBACK_WINDOW,
    NEUTRAL_RISK_SCORE,
    NEUTRAL_SENTIMENT_SCORE,
    TRANSACTION_COST,
)


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def set_seeds(seed_value: int) -> None:
    """Set Python, NumPy, and Torch seeds for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class RIGate:
    """Implements a Rational Inattention Gate over sentiment and risk scores."""

    def __init__(self, threshold: float = 1.0, neutral_score: int = NEUTRAL_SENTIMENT_SCORE):
        self.threshold = threshold
        self.neutral = neutral_score

    def should_pay_attention(self, sentiment_score: float, risk_score: float) -> bool:
        deviation = abs(sentiment_score - self.neutral) + abs(risk_score - self.neutral)
        return deviation >= self.threshold


# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------


class CryptoTradingEnv(gym.Env):
    """Limit order book trading environment with discrete actions."""

    metadata = {"render.modes": []}

    def __init__(
        self,
        data: pd.DataFrame,
        feature_columns: Sequence[str],
        initial_balance: float = INITIAL_BALANCE,
        transaction_cost: float = TRANSACTION_COST,
        lookback_window: int = LOOKBACK_WINDOW,
    ) -> None:
        super().__init__()
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.lookback_window = lookback_window
        self.feature_columns = list(feature_columns)
        self.features_per_step = len(self.feature_columns)

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.lookback_window * self.features_per_step,),
            dtype=np.float32,
        )
        self.reset()

    def reset(self, *, seed: int | None = None, options=None):  # type: ignore[override]
        super().reset(seed=seed)
        self.current_step = self.lookback_window
        self.balance = float(self.initial_balance)
        self.btc_held = 0.0
        self.net_worth_history = [self.initial_balance]
        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        if self.current_step >= len(self.data):
            self.current_step = len(self.data) - 1
        window = self.data.iloc[self.current_step - self.lookback_window : self.current_step]
        return window[self.feature_columns].values.flatten().astype(np.float32)

    def step(self, action: int):  # type: ignore[override]
        if self.current_step >= len(self.data) - 1:
            return self._get_observation(), 0.0, True, True, {}

        price = self.data.loc[self.current_step, "unscaled_midpoint"]
        if price <= 0:
            self.current_step += 1
            next_price = self.data.loc[self.current_step, "unscaled_midpoint"]
            portfolio_value = self.balance + self.btc_held * next_price
            return self._get_observation(), 0.0, False, False, {"portfolio_value": portfolio_value}

        if action == 1 and self.balance > 0:
            btc_bought = (self.balance / price) * (1 - self.transaction_cost)
            self.btc_held += btc_bought
            self.balance = 0.0
        elif action == 2 and self.btc_held > 0:
            self.balance += self.btc_held * price * (1 - self.transaction_cost)
            self.btc_held = 0.0

        self.current_step += 1
        next_price = self.data.loc[self.current_step, "unscaled_midpoint"]
        portfolio_value = self.balance + self.btc_held * next_price
        reward = (portfolio_value - self.net_worth_history[-1]) / self.initial_balance
        self.net_worth_history.append(portfolio_value)
        done = self.current_step >= len(self.data) - 1
        return self._get_observation(), reward, done, False, {"portfolio_value": portfolio_value}


# -----------------------------------------------------------------------------
# Mixed expert ensemble
# -----------------------------------------------------------------------------


class LSTMExtractor(BaseFeaturesExtractor):
    """Custom LSTM feature extractor tailored to LOB sequences."""

    def __init__(self, observation_space: spaces.Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        num_features = observation_space.shape[0] // LOOKBACK_WINDOW
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=features_dim,
            num_layers=2,
            batch_first=True,
        )
        self.lookback_window = LOOKBACK_WINDOW
        self.num_features = num_features

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        reshaped = observations.view(batch_size, self.lookback_window, self.num_features)
        lstm_output, _ = self.lstm(reshaped)
        return lstm_output[:, -1, :]


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            features_extractor_class=LSTMExtractor,
            features_extractor_kwargs=dict(features_dim=128),
        )


class MixedEnsembleExpert:
    """Wrapper combining DQN, PPO, and A2C experts."""

    def __init__(self) -> None:
        self.models = {}
        self.trained = False

    def create_mixed_ensemble(self, vec_env, seed: int | None = None) -> None:
        self.models = {
            "DQN": DQN("MlpPolicy", vec_env, verbose=0, learning_rate=1e-4, buffer_size=50_000, seed=seed),
            "PPO": PPO(CustomActorCriticPolicy, vec_env, verbose=0, learning_rate=3e-4, n_steps=2048, seed=seed),
            "A2C": A2C(CustomActorCriticPolicy, vec_env, verbose=0, learning_rate=7e-4, n_steps=5, seed=seed),
        }

    def train_mixed_ensemble(self, timesteps: int) -> None:
        if not self.models:
            raise RuntimeError("Call create_mixed_ensemble before training.")
        steps = timesteps // len(self.models)
        for name, model in self.models.items():
            print(f"Training {name} expert for {steps} timesteps...")
            model.learn(total_timesteps=steps)
        self.trained = True

    def get_ensemble_action(self, state: np.ndarray) -> int:
        if not self.trained:
            return int(np.random.choice([0, 1, 2]))
        votes = [int(model.predict(state, deterministic=True)[0]) for model in self.models.values()]
        return Counter(votes).most_common(1)[0][0]

    def get_ensemble_confidence(self, state: np.ndarray) -> float:
        if not self.trained:
            return 1.0 / max(len(self.models), 1)
        votes = [int(model.predict(state, deterministic=True)[0]) for model in self.models.values()]
        return Counter(votes).most_common(1)[0][1] / len(votes)


# -----------------------------------------------------------------------------
# LLM-driven expert
# -----------------------------------------------------------------------------


class LLMExpert:
    """Maps sentiment and risk scores to discrete trading actions."""

    def get_llm_action(self, sentiment_score: float, risk_score: float) -> int:
        if sentiment_score >= 4 and risk_score <= 2:
            return 1  # Buy
        if sentiment_score <= 2 or risk_score >= 4:
            return 2  # Sell
        return 0  # Hold


# -----------------------------------------------------------------------------
# DAgger student
# -----------------------------------------------------------------------------


class DAggerAgent(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 512) -> None:
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.policy(x)

    def get_action(self, state: np.ndarray, deterministic: bool = True) -> int:
        with torch.no_grad():
            logits = self.forward(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)).numpy()[0]
        if deterministic:
            return int(np.argmax(logits))
        return int(np.random.choice(len(logits), p=logits))


@dataclass
class DAggerTrainer:
    agent: DAggerAgent
    mixed_ensemble_expert: MixedEnsembleExpert
    llm_expert: LLMExpert
    env: CryptoTradingEnv
    rigate: RIGate
    optimizer: optim.Optimizer = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.optimizer is None:
            self.optimizer = optim.Adam(self.agent.parameters(), lr=5e-4)
        self.dataset: List[Tuple[np.ndarray, int]] = []
        self.loss_history: List[float] = []

    def get_combined_expert_action(self, observation: np.ndarray, row: pd.Series) -> int:  # type: ignore[name-defined]
        ensemble_action = self.mixed_ensemble_expert.get_ensemble_action(observation)
        if {"sentiment_score", "risk_score"}.issubset(row.index):
            sentiment = row.get("sentiment_score", NEUTRAL_SENTIMENT_SCORE)
            risk = row.get("risk_score", NEUTRAL_RISK_SCORE)
            if self.rigate.should_pay_attention(sentiment, risk):
                llm_action = self.llm_expert.get_llm_action(sentiment, risk)
                if llm_action == ensemble_action:
                    return ensemble_action
                confidence = self.mixed_ensemble_expert.get_ensemble_confidence(observation)
                if confidence > 0.66:
                    return ensemble_action
                return llm_action
        return ensemble_action

    def collect_mixed_trajectory(self, beta: float, max_steps: int) -> Tuple[List[np.ndarray], List[int]]:
        states: List[np.ndarray] = []
        actions: List[int] = []
        observation, _ = self.env.reset()
        for _ in range(max_steps):
            if self.env.current_step >= len(self.env.data) - 1:
                break
            row = self.env.data.iloc[self.env.current_step]
            expert_action = self.get_combined_expert_action(observation, row)
            action = expert_action if random.random() < beta else self.agent.get_action(observation, deterministic=False)
            states.append(observation.copy())
            actions.append(expert_action)
            observation, _, done, _, _ = self.env.step(action)
            if done:
                break
        return states, actions

    def train_on_data(self, states: Sequence[np.ndarray], actions: Sequence[int], epochs: int = 10, batch_size: int = 256) -> float:
        if not states:
            return 0.0
        dataset = TensorDataset(
            torch.as_tensor(np.array(states), dtype=torch.float32),
            torch.as_tensor(np.array(actions), dtype=torch.long),
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        total_loss = 0.0
        self.agent.train()
        for _ in range(epochs):
            epoch_loss = 0.0
            for state_batch, action_batch in loader:
                self.optimizer.zero_grad()
                predictions = self.agent(state_batch)
                loss = nn.CrossEntropyLoss()(predictions, action_batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            total_loss += epoch_loss / max(len(loader), 1)
        return total_loss / epochs

    def run_dagger(self, iterations: int = DAGGER_ITERATIONS, trajectory_length: int = DAGGER_TRAJECTORY_LENGTH) -> None:
        betas = np.linspace(1.0, 0.1, iterations)
        for idx, beta in enumerate(betas, start=1):
            print(f"Iteration {idx}/{iterations} with beta={beta:.2f}")
            states, actions = self.collect_mixed_trajectory(beta=beta, max_steps=trajectory_length)
            self.dataset.extend(zip(states, actions))
            if len(self.dataset) > DAGGER_DATASET_LIMIT:
                self.dataset = self.dataset[-DAGGER_DATASET_LIMIT:]
            if self.dataset:
                dataset_states, dataset_actions = zip(*self.dataset)
                loss = self.train_on_data(dataset_states, dataset_actions)
                self.loss_history.append(loss)
                print(f"  Loss: {loss:.4f} | Dataset size: {len(self.dataset)}")
        print("DAgger training complete.")

