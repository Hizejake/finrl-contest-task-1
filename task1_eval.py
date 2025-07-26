"""Evaluation helpers extracted from the Kaggle notebook."""

from collections import Counter
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from trade_simulator import CryptoTradingEnv
from task1_ensemble import DAggerAgent, MixedEnsembleExpert, LLMExpert


def evaluate_agent(agent: DAggerAgent, env: CryptoTradingEnv) -> Tuple[float, List[float], List[int]]:
    """Run the trained agent on the environment and report ROI."""
    print("\nEvaluating trained agent...")
    obs, _ = env.reset()
    actions_taken: List[int] = []
    done = False
    while not done:
        action = agent.get_action(obs, deterministic=True)
        actions_taken.append(int(action))
        obs, _, done, _, _ = env.step(int(action))

    final_portfolio = env.net_worth_history[-1]
    roi = (final_portfolio - env.initial_balance) / env.initial_balance * 100

    print(f"  Final Portfolio Value: ${final_portfolio:.2f}")
    print(f"  Return on Investment (ROI): {roi:.2f}%")
    print(
        f"  Actions - Hold: {actions_taken.count(0)}, Buy: {actions_taken.count(1)}, Sell: {actions_taken.count(2)}"
    )

    return roi, env.net_worth_history, actions_taken


def visualize_results(portfolio_history: List[float], actions_taken: List[int], training_losses: List[float]) -> None:
    """Plot evaluation results and training loss."""
    print("\n--- \U0001F4CA Generating Visualizations ---")
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle("DAgger Agent Performance and Training Summary", fontsize=16)

    axes[0].plot(portfolio_history, color="g", lw=2)
    axes[0].set_title("Portfolio Value During Evaluation")
    axes[0].set_xlabel("Time Steps")
    axes[0].set_ylabel("Portfolio Value ($)")
    axes[0].grid(True)

    action_counts = Counter(actions_taken)
    action_labels = ["Hold", "Buy", "Sell"]
    sizes = [action_counts.get(i, 0) for i in range(3)]
    colors = ["#ff9999", "#66b3ff", "#99ff99"]
    axes[1].pie(sizes, labels=action_labels, autopct="%1.1f%%", startangle=90, colors=colors)
    axes[1].set_title("Action Distribution in Evaluation")
    axes[1].axis("equal")

    axes[2].plot(training_losses, marker="o", linestyle="--", color="r")
    axes[2].set_title("DAgger Training Loss per Iteration")
    axes[2].set_xlabel("DAgger Iteration")
    axes[2].set_ylabel("Cross-Entropy Loss")
    axes[2].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
