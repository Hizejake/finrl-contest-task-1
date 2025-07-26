"""DAgger agent and expert definitions used for Task 1."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class LLMExpert:
    """Expert policy that consults an instruction-tuned LLM."""

    MODEL_MAP = {
        "gemma": "google/gemma-1.1-2b-it",
        "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
        "deepseek": "deepseek-ai/deepseek-llm-7b-instruct",
    }

    def __init__(self, model_key: str):
        if model_key not in self.MODEL_MAP:
            raise ValueError(f"Unknown model {model_key}")
        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_MAP[model_key])
        model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_MAP[model_key],
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_4bit=True,
        )
        self.pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    def predict(self, observation: np.ndarray) -> int:
        prompt = (
            "Given the following market observation, return an action among \"hold\"",
            ", \"buy\" or \"sell\". Only output the word.\nObservation:\n"
        ) + str(observation.tolist()) + "\nAction:"
        out = self.pipe(prompt, max_new_tokens=1)[0]["generated_text"]
        result = out.split(prompt)[-1].strip().lower()
        if "buy" in result:
            return 1
        if "sell" in result:
            return 2
        return 0


class MixedEnsembleExpert:
    """Combines heuristics with an LLM expert."""

    def __init__(self, llm_model: Optional[str] = None):
        self.llm = None if llm_model is None else LLMExpert(llm_model)

    def predict(self, observation: np.ndarray) -> int:
        ma_short = observation[-1, 3]  # close price
        ma_long = observation[:, 3].mean()
        rule_action = 1 if ma_short > ma_long else 2
        if self.llm is None:
            return rule_action
        llm_action = self.llm.predict(observation)
        return llm_action if np.random.rand() < 0.5 else rule_action


class DAggerAgent:
    """Simplified DAgger trainer around a PPO policy."""

    def __init__(self, env, expert: MixedEnsembleExpert, iterations: int = 5, timesteps: int = 2000):
        self.env = DummyVecEnv([lambda: env])
        self.expert = expert
        self.iterations = iterations
        self.timesteps = timesteps
        self.model = PPO("MlpPolicy", self.env, verbose=0)

    def train(self):
        for _ in range(self.iterations):
            obs = self.env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=False)
                expert_action = self.expert.predict(obs[0])
                obs, _, done, _ = self.env.step([expert_action])
            self.model.learn(total_timesteps=self.timesteps)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)

    def load(self, path: str):
        self.model = PPO.load(path, env=self.env)
