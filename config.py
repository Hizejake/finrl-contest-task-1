# config.py
"""Configuration settings for FinAI Contest 2025 Task 1."""

# Model configurations
MODELS_TO_TEST = ["gemma", "llama3", "deepseek", "no_llm"]
MODELS_CONFIG = {
    "gemma": {"model_id": "google/gemma-7b-it", "prompt_type": "gemma"},
    "llama3": {"model_id": "meta-llama/Meta-Llama-3-8B-Instruct", "prompt_type": "llama3"},
    "deepseek": {"model_id": "deepseek-ai/deepseek-coder-6.7b-instruct", "prompt_type": "deepseek"},
    "no_llm": {"model_id": None, "prompt_type": "none"}
}

# Data paths (adjust these for your actual data location)
NEWS_CSV_PATH = "data/news_train.csv"
LOB_DATA_PATH = "data/BTC_1min.csv"

# Training parameters
NEWS_INFLUENCE_MINUTES = 60
NEUTRAL_SENTIMENT_SCORE = 3
NEUTRAL_RISK_SCORE = 3
SEED_VALUE = 20
LOOKBACK_WINDOW = 60
TOTAL_TRAINING_TIMESTEPS = 60000
DAGGER_ITERATIONS = 10

# Environment parameters
INITIAL_BALANCE = 10000
TRANSACTION_COST = 0.001

# DAgger parameters
DAGGER_LEARNING_RATE = 5e-4
DAGGER_EPOCHS = 10
DAGGER_BATCH_SIZE = 256
HIDDEN_DIM = 512