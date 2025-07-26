"""Evaluation utilities for trained agents."""

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

from trade_simulator import CryptoTradingEnv
from task1_ensemble import DAggerAgent, MixedEnsembleExpert


def evaluate_agent(model_path: str, dataset: pd.DataFrame) -> List[float]:
    env = CryptoTradingEnv(dataset)
    expert = MixedEnsembleExpert()
    agent = DAggerAgent(env, expert)
    agent.load(model_path)

    obs, _ = env.reset()
    done = False
    values = []
    while not done:
        action, _ = agent.model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(int(action))
        values.append(info["portfolio_value"])
    return values


def visualize_results(results: Dict[str, List[float]], btc_prices: List[float]):
    plt.figure(figsize=(10, 5))
    for name, vals in results.items():
        plt.plot(vals, label=name)
    plt.plot(btc_prices[: len(vals)], label="BTC", linestyle="--")
    plt.legend()
    plt.xlabel("Step")
    plt.ylabel("Portfolio Value")
    plt.title("Agent Performance")
    plt.tight_layout()
    plt.savefig("comparison.png")
    plt.close()


def main():
    btc = pd.read_csv("BTC_1min.csv")
    btc_prices = btc["close"].tolist()

    results = {}
    for model in ["gemma", "llama3", "deepseek", "none"]:
        dataset = pd.read_csv(os.path.join("datasets", f"dataset_{model}.csv"))
        model_path = os.path.join("trained_models", f"dagger_{model}.zip")
        if not os.path.exists(model_path):
            continue
        pv = evaluate_agent(model_path, dataset)
        results[model] = pv
        pd.DataFrame({"portfolio": pv}).to_csv(f"results_{model}.csv", index=False)

    visualize_results(results, btc_prices)


if __name__ == "__main__":
    main()
