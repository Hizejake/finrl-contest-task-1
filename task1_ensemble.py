"""Expert policies and DAgger agent from the Kaggle notebook."""

from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv


class MixedEnsembleExpert:
    """Ensemble of DQN, PPO and A2C experts."""

    def __init__(self) -> None:
        self.models: dict[str, any] = {}
        self.trained = False

    def create_mixed_ensemble(self, env, seed: Optional[int] = None) -> None:
        print("Creating mixed RL ensemble (DQN + PPO + A2C) with LSTM Policies...")
        policy_type = "MlpPolicy"
        self.models["DQN"] = DQN(policy_type, env, verbose=0, learning_rate=0.0001, buffer_size=50000, seed=seed)
        self.models["PPO"] = PPO(policy_type, env, verbose=0, learning_rate=0.0003, n_steps=2048, seed=seed)
        self.models["A2C"] = A2C(policy_type, env, verbose=0, learning_rate=0.0007, n_steps=5, seed=seed)

    def train_mixed_ensemble(self, timesteps: int = 60000) -> None:
        print("Training mixed ensemble...")
        steps_per_model = timesteps // len(self.models)
        for name, model in self.models.items():
            print(f"Training {name} ({steps_per_model} timesteps)...")
            model.learn(total_timesteps=steps_per_model)
        self.trained = True
        print("✅ Mixed ensemble training completed!")

    def get_ensemble_action(self, state: np.ndarray) -> int:
        if not self.trained:
            return int(np.random.choice([0, 1, 2]))
        votes = [int(model.predict(state, deterministic=True)[0]) for model in self.models.values()]
        return Counter(votes).most_common(1)[0][0]

    def get_ensemble_confidence(self, state: np.ndarray) -> float:
        if not self.trained:
            return 0.33
        votes = [int(model.predict(state, deterministic=True)[0]) for model in self.models.values()]
        return Counter(votes).most_common(1)[0][1] / len(votes)


class LLMExpert:
    """Simple rule-based expert using sentiment and risk scores."""

    def get_llm_action(self, sentiment_score: float, risk_score: float) -> int:
        if sentiment_score >= 4 and risk_score <= 2:
            return 1
        if sentiment_score <= 2 or risk_score >= 4:
            return 2
        return 0


class DAggerAgent(nn.Module):
    """Neural network policy used by DAgger."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.network = nn.Sequential(
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.network(x)

    def get_action(self, state: np.ndarray, deterministic: bool = True) -> int:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs = self.forward(state_tensor).numpy()[0]
            if deterministic:
                return int(np.argmax(action_probs))
            return int(np.random.choice(len(action_probs), p=action_probs))
