"""Train the DAgger policy using processed datasets."""
from __future__ import annotations

import argparse
import json
import os
from typing import List

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from stable_baselines3.common.vec_env import DummyVecEnv

from src.agent import (
    DAggerAgent,
    DAggerTrainer,
    LLMExpert,
    MixedEnsembleExpert,
    RIGate,
    CryptoTradingEnv,
    set_seeds,
)
from src.config import (
    DATA_DIR,
    DAGGER_ITERATIONS,
    DAGGER_TRAJECTORY_LENGTH,
    DEFAULT_MODEL,
    FINAL_DATASET_TEMPLATE,
    MODELS_DIR,
    MODEL_CHECKPOINT_TEMPLATE,
    LOSS_HISTORY_TEMPLATE,
    MODELS_TO_TEST,
    NEUTRAL_SENTIMENT_SCORE,
    RI_GATE_THRESHOLD,
    SEED_VALUE,
    TOTAL_TRAINING_TIMESTEPS,
    TRAIN_TEST_SPLIT,
    LOOKBACK_WINDOW,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def load_processed_dataset(model_name: str) -> pd.DataFrame:
    dataset_path = DATA_DIR / FINAL_DATASET_TEMPLATE.format(model=model_name)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found for model '{model_name}'. Expected at {dataset_path}. "
            "Run `python src/data_processing.py` first."
        )

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
        raise ValueError("No numeric feature columns available for training.")
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

    return train_df, test_df, scaler


def build_vector_env(train_df: pd.DataFrame, feature_columns: List[str]):
    def _make_env():
        return CryptoTradingEnv(train_df, feature_columns, lookback_window=LOOKBACK_WINDOW)

    return DummyVecEnv([_make_env])


def train_agent(model_name: str) -> None:
    print(f"Training DAgger agent for model: {model_name}")
    set_seeds(SEED_VALUE)

    df = load_processed_dataset(model_name)
    feature_columns = extract_feature_columns(df)
    train_df, _, _ = split_and_scale(df, feature_columns)

    train_env = CryptoTradingEnv(train_df, feature_columns, lookback_window=LOOKBACK_WINDOW)

    vec_env = build_vector_env(train_df, feature_columns)
    mixed_expert = MixedEnsembleExpert()
    mixed_expert.create_mixed_ensemble(vec_env, seed=SEED_VALUE)
    mixed_expert.train_mixed_ensemble(TOTAL_TRAINING_TIMESTEPS)

    llm_expert = LLMExpert()
    agent = DAggerAgent(train_env.observation_space.shape[0], train_env.action_space.n)
    ri_gate = RIGate(threshold=RI_GATE_THRESHOLD, neutral_score=NEUTRAL_SENTIMENT_SCORE)
    trainer = DAggerTrainer(agent, mixed_expert, llm_expert, train_env, ri_gate)
    trainer.run_dagger(iterations=DAGGER_ITERATIONS, trajectory_length=DAGGER_TRAJECTORY_LENGTH)

    MODELS_DIR.mkdir(exist_ok=True)
    checkpoint_path = MODELS_DIR / MODEL_CHECKPOINT_TEMPLATE.format(model=model_name)
    torch.save(agent.state_dict(), checkpoint_path)
    loss_path = MODELS_DIR / LOSS_HISTORY_TEMPLATE.format(model=model_name)
    with loss_path.open("w", encoding="utf-8") as fp:
        json.dump(trainer.loss_history, fp)
    print(f"Saved trained agent to {checkpoint_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the DAgger imitation learner.")
    parser.add_argument(
        "--model",
        choices=MODELS_TO_TEST,
        default=DEFAULT_MODEL,
        help="Name of the LLM configuration whose dataset should be used for training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_agent(args.model)


if __name__ == "__main__":
    main()
