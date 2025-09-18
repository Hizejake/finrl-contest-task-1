"""Evaluate trained agents across all expert configurations."""
from __future__ import annotations

import json
import os
from collections import Counter
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src.agent import DAggerAgent, CryptoTradingEnv, set_seeds
from src.config import (
    ANNUALIZATION_FACTOR,
    DATA_DIR,
    FINAL_DATASET_TEMPLATE,
    LOSS_HISTORY_TEMPLATE,
    MODEL_CHECKPOINT_TEMPLATE,
    MODELS_DIR,
    MODELS_TO_TEST,
    LOOKBACK_WINDOW,
    SEED_VALUE,
    TRAIN_TEST_SPLIT,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def load_processed_dataset(model_name: str) -> pd.DataFrame:
    dataset_path = DATA_DIR / FINAL_DATASET_TEMPLATE.format(model=model_name)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing dataset for model '{model_name}' at {dataset_path}.")
    df = pd.read_csv(dataset_path)
    if "system_time" in df:
        df["system_time"] = pd.to_datetime(df["system_time"], utc=True, errors="coerce")
        df = df.sort_values("system_time")
    df = df.reset_index(drop=True)
    df = df.fillna(method="ffill").fillna(0)
    if "unscaled_midpoint" not in df.columns:
        if "midpoint" in df.columns:
            df["unscaled_midpoint"] = df["midpoint"]
        elif {"bids_price_0", "asks_price_0"}.issubset(df.columns):
            df["unscaled_midpoint"] = (df["bids_price_0"] + df["asks_price_0"]) / 2
        else:
            raise ValueError("Dataset must contain price columns to derive the midpoint.")
    return df


def extract_feature_columns(df: pd.DataFrame) -> List[str]:
    excluded = {"system_time", "unscaled_midpoint", "news_time"}
    features = [
        column
        for column in df.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(df[column])
    ]
    if not features:
        raise ValueError("No numeric feature columns available for evaluation.")
    return features


def split_and_scale(df: pd.DataFrame, feature_columns: List[str]):
    split_index = int(len(df) * TRAIN_TEST_SPLIT)
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("Dataset is too small to split into train and test segments.")
    scaler = StandardScaler()
    train_df[feature_columns] = scaler.fit_transform(train_df[feature_columns])
    test_df[feature_columns] = scaler.transform(test_df[feature_columns])
    return train_df, test_df


def load_loss_history(model_name: str) -> List[float]:
    loss_path = MODELS_DIR / LOSS_HISTORY_TEMPLATE.format(model=model_name)
    if loss_path.exists():
        with loss_path.open("r", encoding="utf-8") as fp:
            try:
                return json.load(fp)
            except json.JSONDecodeError:
                return []
    return []


def evaluate_agent(agent: DAggerAgent, env: CryptoTradingEnv) -> Dict[str, object]:
    agent.eval()
    observation, _ = env.reset()
    terminated = False
    truncated = False
    actions = []
    while not (terminated or truncated):
        action = agent.get_action(observation)
        actions.append(int(action))
        observation, _, terminated, truncated, _ = env.step(action)
    portfolio_history = pd.Series(env.net_worth_history)
    returns = portfolio_history.pct_change().dropna()
    sharpe_ratio = 0.0
    if not returns.empty and returns.std() != 0:
        sharpe_ratio = (returns.mean() / returns.std()) * ANNUALIZATION_FACTOR
    running_max = portfolio_history.cummax()
    drawdown = (portfolio_history - running_max) / running_max
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0
    final_value = portfolio_history.iloc[-1]
    roi = (final_value - env.initial_balance) / env.initial_balance * 100
    return {
        "roi": float(roi),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": max_drawdown,
        "portfolio_history": env.net_worth_history,
        "actions": actions,
        "final_value": float(final_value),
    }


def evaluate_model(model_name: str):
    checkpoint_path = MODELS_DIR / MODEL_CHECKPOINT_TEMPLATE.format(model=model_name)
    if not checkpoint_path.exists():
        print(f"Skipping {model_name}: checkpoint not found at {checkpoint_path}")
        return None

    df = load_processed_dataset(model_name)
    feature_columns = extract_feature_columns(df)
    _, test_df = split_and_scale(df, feature_columns)

    test_env = CryptoTradingEnv(test_df, feature_columns, lookback_window=LOOKBACK_WINDOW)
    agent = DAggerAgent(test_env.observation_space.shape[0], test_env.action_space.n)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    agent.load_state_dict(state_dict)

    metrics = evaluate_agent(agent, test_env)
    metrics["loss_history"] = load_loss_history(model_name)
    return metrics


def main() -> None:
    set_seeds(SEED_VALUE)
    results = {}
    for model_name in MODELS_TO_TEST:
        print(f"\nEvaluating model: {model_name}")
        try:
            metrics = evaluate_model(model_name)
        except FileNotFoundError as err:
            print(f"  {err}")
            continue
        if metrics is None:
            continue
        results[model_name] = metrics

    if not results:
        print("No models evaluated. Ensure training has been completed.")
        return

    summary_records = []
    for model_name, metrics in results.items():
        action_counts = Counter(metrics["actions"])
        total_trades = action_counts.get(1, 0) + action_counts.get(2, 0)
        summary_records.append(
            {
                "model": model_name,
                "roi": metrics["roi"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "max_drawdown": metrics["max_drawdown"],
                "total_trades": total_trades,
            }
        )

    summary_df = pd.DataFrame(summary_records).set_index("model")
    print("\n" + "=" * 80)
    print("FINAL COMPARATIVE ANALYSIS RESULTS")
    print("=" * 80)
    display_df = summary_df.copy()
    display_df["roi"] = display_df["roi"].map("{:.2f}%".format)
    display_df["sharpe_ratio"] = display_df["sharpe_ratio"].map("{:.2f}".format)
    display_df["max_drawdown"] = display_df["max_drawdown"].map("{:.2%}".format)
    print(display_df)

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    fig.suptitle("Comparative Analysis of LLM-Enhanced Trading Agents", fontsize=18)

    # ROI bar chart
    axes[0, 0].bar(summary_df.index, summary_df["roi"], color="steelblue")
    axes[0, 0].set_title("ROI Comparison")
    axes[0, 0].set_ylabel("ROI (%)")
    axes[0, 0].grid(axis="y", linestyle="--", alpha=0.5)

    # Portfolio trajectories
    for model_name, metrics in results.items():
        axes[0, 1].plot(metrics["portfolio_history"], label=model_name.upper())
    axes[0, 1].set_title("Portfolio Value Over Time")
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle="--", alpha=0.4)

    # Action distribution
    action_data = {
        model_name: [Counter(metrics["actions"]).get(action, 0) for action in range(3)]
        for model_name, metrics in results.items()
    }
    action_df = pd.DataFrame(action_data, index=["Hold", "Buy", "Sell"])
    action_df.plot(kind="bar", ax=axes[1, 0], rot=0)
    axes[1, 0].set_title("Action Distribution")
    axes[1, 0].set_ylabel("Count")

    # DAgger loss curves
    for model_name, metrics in results.items():
        loss_history = metrics.get("loss_history", [])
        if loss_history:
            axes[1, 1].plot(loss_history, marker="o", linestyle="--", label=model_name.upper())
    axes[1, 1].set_title("DAgger Training Loss")
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    main()

