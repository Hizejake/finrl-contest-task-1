"""Full DAgger training script adapted from the Kaggle notebook."""

import argparse
import os
import random
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from stable_baselines3.common.vec_env import DummyVecEnv

from task1_ensemble import MixedEnsembleExpert, LLMExpert, DAggerAgent
from trade_simulator import CryptoTradingEnv
from task1_eval import evaluate_agent, visualize_results


def set_seeds(seed_value: int) -> None:
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✅ All random seeds set to: {seed_value}")


class DAggerTrainer:
    """Trainer implementing the DAgger algorithm."""

    def __init__(self, agent: DAggerAgent, mixed_expert: MixedEnsembleExpert, llm_expert: LLMExpert, env: CryptoTradingEnv):
        self.agent = agent
        self.mixed_expert = mixed_expert
        self.llm_expert = llm_expert
        self.env = env
        self.optimizer = torch.optim.Adam(agent.parameters(), lr=0.0005)
        self.dataset: list[Tuple[np.ndarray, int]] = []
        self.loss_history: list[float] = []

    def get_combined_expert_action(self, obs: np.ndarray, current_row: pd.Series) -> int:
        ensemble_action = self.mixed_expert.get_ensemble_action(obs)
        llm_action = self.llm_expert.get_llm_action(current_row['sentiment_score'], current_row['risk_score'])
        ensemble_confidence = self.mixed_expert.get_ensemble_confidence(obs)
        if ensemble_action == llm_action:
            return ensemble_action
        if ensemble_confidence > 0.66:
            return ensemble_action
        return llm_action

    def collect_mixed_trajectory(self, beta: float = 0.5, max_steps: int = 1000) -> Tuple[list[np.ndarray], list[int]]:
        states: list[np.ndarray] = []
        expert_actions: list[int] = []
        obs, _ = self.env.reset()
        for _ in range(max_steps):
            if self.env.current_step >= len(self.env.data) - 1:
                break
            current_row = self.env.data.iloc[self.env.current_step]
            expert_action = self.get_combined_expert_action(obs, current_row)
            action_to_take = expert_action if np.random.random() < beta else self.agent.get_action(obs, deterministic=False)
            states.append(obs.copy())
            expert_actions.append(expert_action)
            obs, _, done, _, _ = self.env.step(action_to_take)
            if done:
                break
        return states, expert_actions

    def train_on_data(self, states: list[np.ndarray], actions: list[int], epochs: int = 10, batch_size: int = 256) -> float:
        if not states:
            return 0.0
        dataset = torch.utils.data.TensorDataset(torch.FloatTensor(np.array(states)), torch.LongTensor(np.array(actions)))
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.agent.train()
        epoch_loss = 0.0
        for _ in range(epochs):
            batch_loss = 0.0
            for state_batch, action_batch in loader:
                self.optimizer.zero_grad()
                action_probs = self.agent(state_batch)
                loss = torch.nn.CrossEntropyLoss()(action_probs, action_batch)
                loss.backward()
                self.optimizer.step()
                batch_loss += float(loss.item())
            epoch_loss += batch_loss / len(loader)
        return epoch_loss / epochs

    def run_dagger(self, iterations: int = 10, trajectory_length: int = 1000) -> None:
        print(f"\nRunning DAgger for {iterations} iterations...")
        beta_schedule = np.linspace(1.0, 0.1, iterations)
        for i in range(iterations):
            print(f"\nDAgger Iteration {i + 1}/{iterations} (β={beta_schedule[i]:.2f})")
            states, expert_actions = self.collect_mixed_trajectory(beta=beta_schedule[i], max_steps=trajectory_length)
            self.dataset.extend(list(zip(states, expert_actions)))
            if len(self.dataset) > 20000:
                self.dataset = self.dataset[-20000:]
            if self.dataset:
                d_states, d_actions = zip(*self.dataset)
                loss = self.train_on_data(list(d_states), list(d_actions))
                self.loss_history.append(loss)
                print(f"Training loss: {loss:.4f}, Dataset size: {len(self.dataset)}")
        print("✅ DAgger training completed!")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Path to pre-generated dataset CSV")
    parser.add_argument(
        "--model",
        choices=["gemma", "llama3", "deepseek", "none"],
        help="Convenience flag to use datasets/dataset_<model>.csv",
    )
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--lookback", type=int, default=60)
    args = parser.parse_args()

    if args.dataset is None:
        chosen = args.model if args.model else "gemma"
        args.dataset = os.path.join("datasets", f"dataset_{chosen}.csv")

    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    df = pd.read_csv(args.dataset)
    print(f"✅ Successfully loaded data from {args.dataset}")
    df["system_time"] = pd.to_datetime(df["system_time"])
    df = df.sort_values("system_time").reset_index(drop=True)
    df = df.fillna(method="ffill").fillna(0)

    non_feature_cols = ["system_time", "news_time"] if "news_time" in df.columns else ["system_time"]
    global all_feature_columns
    all_feature_columns = [c for c in df.columns if c not in non_feature_cols]
    print(f"✅ Detected {len(all_feature_columns)} feature columns.")

    train_size = int(0.8 * len(df))
    train_data_df = df[:train_size].copy()
    test_data_df = df[train_size:].copy()

    if "midpoint" not in df.columns and "bids_price_0" in df.columns:
        df["midpoint"] = (df["bids_price_0"] + df["asks_price_0"]) / 2
    unscaled_train_midpoint = train_data_df["midpoint"].copy()
    unscaled_test_midpoint = test_data_df["midpoint"].copy()

    scaler = StandardScaler()
    train_data_df[all_feature_columns] = scaler.fit_transform(train_data_df[all_feature_columns])
    test_data_df[all_feature_columns] = scaler.transform(test_data_df[all_feature_columns])
    train_data_df["unscaled_midpoint"] = unscaled_train_midpoint
    test_data_df["unscaled_midpoint"] = unscaled_test_midpoint
    print("✅ Data scaling complete.")

    set_seeds(args.seed)

    train_env = CryptoTradingEnv(train_data_df, lookback_window=args.lookback)
    test_env = CryptoTradingEnv(test_data_df, lookback_window=args.lookback)
    vec_train_env = DummyVecEnv([lambda: train_env])

    mixed_expert = MixedEnsembleExpert()
    llm_expert = LLMExpert()
    state_dim = train_env.observation_space.shape[0]
    agent = DAggerAgent(state_dim, 3, hidden_dim=512)
    print(f"Agent state dimension: {state_dim}")

    mixed_expert.create_mixed_ensemble(vec_train_env, seed=args.seed)
    mixed_expert.train_mixed_ensemble(timesteps=60000)

    trainer = DAggerTrainer(agent, mixed_expert, llm_expert, train_env)
    trainer.run_dagger(iterations=10, trajectory_length=1000)

    roi, portfolio_history, actions_taken = evaluate_agent(agent, test_env)
    visualize_results(portfolio_history, actions_taken, trainer.loss_history)

    os.makedirs("trained_models", exist_ok=True)
    model_path = os.path.join("trained_models", "dagger_final.pt")
    torch.save(agent.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()
