"""Run DAgger training for the selected dataset."""

import argparse
import os

import pandas as pd

from trade_simulator import CryptoTradingEnv
from task1_ensemble import MixedEnsembleExpert, DAggerAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["gemma", "llama3", "deepseek", "none"], default="gemma")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--timesteps", type=int, default=2000)
    args = parser.parse_args()

    dataset_path = os.path.join("datasets", f"dataset_{args.model}.csv")
    data = pd.read_csv(dataset_path)

    env = CryptoTradingEnv(data)
    expert = MixedEnsembleExpert(llm_model=None if args.model == "none" else args.model)
    agent = DAggerAgent(env, expert, iterations=args.iterations, timesteps=args.timesteps)
    agent.train()

    out_path = os.path.join("trained_models", f"dagger_{args.model}.zip")
    agent.save(out_path)
    print(f"Model saved to {out_path}")


if __name__ == "__main__":
    main()
