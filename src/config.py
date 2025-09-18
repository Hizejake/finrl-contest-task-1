"""Global configuration for the FinAI Contest 2025 crypto trading pipeline."""
from __future__ import annotations

from pathlib import Path

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"

NEWS_CSV_PATH = DATA_DIR / "news_train.csv"
LOB_DATA_PATH = DATA_DIR / "BTC_1sec.csv"

FINAL_DATASET_TEMPLATE = "final_dataset_{model}.csv"
MODEL_CHECKPOINT_TEMPLATE = "dagger_agent_{model}.pth"
LOSS_HISTORY_TEMPLATE = "dagger_agent_{model}_loss.json"

# -----------------------------------------------------------------------------
# Experiment configuration
# -----------------------------------------------------------------------------
MODELS_TO_TEST = ["gemma", "llama3", "deepseek", "no_llm"]
MODELS_CONFIG = {
    "gemma": {"model_id": "google/gemma-7b-it", "prompt_type": "gemma"},
    "llama3": {"model_id": "meta-llama/Meta-Llama-3-8B-Instruct", "prompt_type": "llama3"},
    "deepseek": {"model_id": "deepseek-ai/deepseek-coder-6.7b-instruct", "prompt_type": "deepseek"},
    "no_llm": {"model_id": None, "prompt_type": "none"},
}

NEWS_INFLUENCE_MINUTES = 60
NEUTRAL_SENTIMENT_SCORE = 3
NEUTRAL_RISK_SCORE = 3
RI_GATE_THRESHOLD = 1.5

LOOKBACK_WINDOW = 60
INITIAL_BALANCE = 10_000
TRANSACTION_COST = 0.001

TOTAL_TRAINING_TIMESTEPS = 60_000
DAGGER_ITERATIONS = 10
DAGGER_TRAJECTORY_LENGTH = 1_000
DAGGER_DATASET_LIMIT = 20_000

TRAIN_TEST_SPLIT = 0.8
SEED_VALUE = 123

# -----------------------------------------------------------------------------
# Evaluation constants
# -----------------------------------------------------------------------------
ANNUALIZATION_FACTOR = (252 * 24 * 60 * 60) ** 0.5

# -----------------------------------------------------------------------------
# Miscellaneous
# -----------------------------------------------------------------------------
DEFAULT_MODEL = "gemma"
TOKENIZERS_PARALLELISM = "false"
HF_TOKEN_ENV_KEY = "HF_TOKEN"

