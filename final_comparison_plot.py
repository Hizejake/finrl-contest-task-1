"""Plot portfolio values of all agents against BTC price."""

import os
import matplotlib.pyplot as plt
import pandas as pd


def main():
    btc = pd.read_csv("BTC_1min.csv")
    plt.figure(figsize=(10, 5))
    plt.plot(btc["close"].tolist(), label="BTC", linestyle="--")

    for model in ["gemma", "llama3", "deepseek", "none"]:
        results_path = f"results_{model}.csv"
        if not os.path.exists(results_path):
            continue
        vals = pd.read_csv(results_path)["portfolio"].tolist()
        plt.plot(vals, label=model)

    plt.legend()
    plt.xlabel("Step")
    plt.ylabel("Portfolio Value")
    plt.title("Portfolio Value Comparison")
    plt.tight_layout()
    plt.savefig("final_comparison.png")
    plt.show()


if __name__ == "__main__":
    main()
