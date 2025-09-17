# utils.py
"""Utility functions for FinAI Contest 2025 Task 1."""

import random
import numpy as np
import torch
import os
import warnings
from config import SEED_VALUE

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def set_seeds(seed_value=SEED_VALUE):
    """Set all random seeds for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✅ All random seeds set to: {seed_value}")

def authenticate_huggingface():
    """Authenticate with Hugging Face Hub."""
    try:
        from huggingface_hub import login
        from kaggle_secrets import UserSecretsClient
        secs = UserSecretsClient()
        token = secs.get_secret("HF_TOKEN")
        login(token=token)
        print("✅ Authenticated with HF.")
        return True
    except Exception as e:
        print(f"❌ HF_TOKEN missing or authentication failed: {e}")
        return False

def cleanup_memory():
    """Clean up GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("🧹 GPU memory cleaned up.")

def evaluate_agent(agent, env):
    """Evaluate a trained agent on the given environment."""
    print("\nEvaluating trained agent...")
    obs, _ = env.reset()
    done = False
    acts = []
    
    while not done:
        act = agent.get_action(obs)
        acts.append(act)
        obs, _, done, _, _ = env.step(act)
    
    final = env.net_worth_history[-1]
    roi = (final - env.initial_balance) / env.initial_balance * 100
    
    print(f"Final: ${final:.2f}, ROI: {roi:.2f}%")
    print(f"Actions H/B/S: {acts.count(0)}/{acts.count(1)}/{acts.count(2)}")
    
    return roi, env.net_worth_history, acts