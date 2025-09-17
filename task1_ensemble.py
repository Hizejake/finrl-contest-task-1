# task1_ensemble.py
"""Ensemble training pipeline for FinAI Contest 2025 Task 1."""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import Counter
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv

from config import *
from utils import set_seeds, authenticate_huggingface, cleanup_memory
from trade_simulator import CryptoTradingEnv
from factor_mining import RIGate, load_llm_model, process_news_data, merge_data, prepare_features

class MixedEnsembleExpert:
    """Ensemble of RL experts using majority voting."""
    
    def __init__(self):
        self.models = {}
        self.trained = False
    
    def create_mixed_ensemble(self, env, seed=None):
        """Create ensemble of DQN, PPO, and A2C models."""
        self.models['DQN'] = DQN(
            "MlpPolicy", env, verbose=0, 
            learning_rate=1e-4, buffer_size=50000, seed=seed
        )
        self.models['PPO'] = PPO(
            "MlpPolicy", env, verbose=0, 
            learning_rate=3e-4, n_steps=2048, seed=seed
        )
        self.models['A2C'] = A2C(
            "MlpPolicy", env, verbose=0, 
            learning_rate=7e-4, n_steps=5, seed=seed
        )
    
    def train_mixed_ensemble(self, timesteps=TOTAL_TRAINING_TIMESTEPS):
        """Train all models in the ensemble."""
        steps_per_model = timesteps // len(self.models)
        print(f"Training ensemble with {steps_per_model} steps per model...")
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.learn(total_timesteps=steps_per_model)
        
        self.trained = True
        print("✅ Ensemble training completed!")
    
    def get_ensemble_action(self, state):
        """Get action using majority voting."""
        if not self.trained:
            return np.random.choice([0, 1, 2])
        
        votes = [int(model.predict(state, deterministic=True)[0]) for model in self.models.values()]
        return Counter(votes).most_common(1)[0][0]
    
    def get_ensemble_confidence(self, state):
        """Get confidence level of ensemble decision."""
        if not self.trained:
            return 1/3
        
        votes = [int(model.predict(state, deterministic=True)[0]) for model in self.models.values()]
        return Counter(votes).most_common(1)[0][1] / len(votes)
    
    def save_models(self, save_dir="trained_models"):
        """Save all trained models."""
        os.makedirs(save_dir, exist_ok=True)
        for name, model in self.models.items():
            model.save(os.path.join(save_dir, f"{name}_model"))
        print(f"✅ Models saved to {save_dir}")
    
    def load_models(self, save_dir="trained_models"):
        """Load trained models."""
        for name in ['DQN', 'PPO', 'A2C']:
            model_path = os.path.join(save_dir, f"{name}_model")
            if os.path.exists(model_path + ".zip"):
                if name == 'DQN':
                    self.models[name] = DQN.load(model_path)
                elif name == 'PPO':
                    self.models[name] = PPO.load(model_path)
                elif name == 'A2C':
                    self.models[name] = A2C.load(model_path)
        self.trained = len(self.models) > 0
        print(f"✅ Loaded {len(self.models)} models from {save_dir}")

class LLMExpert:
    """Expert that makes trading decisions based on LLM sentiment and risk scores."""
    
    def get_llm_action(self, sentiment_score, risk_score):
        """
        Get trading action based on sentiment and risk scores.
        
        Args:
            sentiment_score: 1-5 sentiment score
            risk_score: 1-5 risk score
            
        Returns:
            int: Action (0=Hold, 1=Buy, 2=Sell)
        """
        # Buy if very positive sentiment and low risk
        if sentiment_score >= 4 and risk_score <= 2:
            return 1
        
        # Sell if very negative sentiment or high risk
        if sentiment_score <= 2 or risk_score >= 4:
            return 2
        
        # Hold otherwise
        return 0

class DAggerAgent(nn.Module):
    """DAgger agent that learns from expert demonstrations."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.net(x)
    
    def get_action(self, state, deterministic=True):
        """Get action from the agent."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            probs = self.forward(state_tensor).numpy()[0]
        
        if deterministic:
            return np.argmax(probs)
        else:
            return np.random.choice(len(probs), p=probs)
    
    def save_model(self, path="trained_models/dagger_agent.pth"):
        """Save the DAgger agent."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"✅ DAgger agent saved to {path}")
    
    def load_model(self, path="trained_models/dagger_agent.pth"):
        """Load the DAgger agent."""
        if os.path.exists(path):
            self.load_state_dict(torch.load(path))
            print(f"✅ DAgger agent loaded from {path}")
            return True
        return False

class DAggerTrainer:
    """DAgger trainer with rational inattention."""
    
    def __init__(self, agent, mixed_ensemble_expert, llm_expert, env, rigate=None):
        self.agent = agent
        self.mixed_ensemble_expert = mixed_ensemble_expert
        self.llm_expert = llm_expert
        self.env = env
        self.optimizer = optim.Adam(agent.parameters(), lr=DAGGER_LEARNING_RATE)
        self.dataset = []
        self.loss_history = []
        self.rigate = rigate or RIGate()
    
    def get_combined_expert_action(self, obs, row):
        """Get combined action from ensemble and LLM experts using rational inattention."""
        ensemble_action = self.mixed_ensemble_expert.get_ensemble_action(obs)
        
        # Check if LLM signals are available and should be used
        if 'sentiment_score' in row.index and 'risk_score' in row.index:
            sentiment = row['sentiment_score']
            risk = row['risk_score']
            
            # Use rational inattention gate
            if self.rigate.should_pay_attention(sentiment, risk):
                llm_action = self.llm_expert.get_llm_action(sentiment, risk)
                
                # If LLM and ensemble agree, use the action
                if llm_action == ensemble_action:
                    return ensemble_action
                
                # If ensemble is very confident, trust it over LLM
                if self.mixed_ensemble_expert.get_ensemble_confidence(obs) > 0.66:
                    return ensemble_action
                
                # Otherwise, use LLM action
                return llm_action
        
        # Default to ensemble action
        return ensemble_action
    
    def collect_mixed_trajectory(self, beta=0.5, max_steps=1000):
        """Collect trajectory with mixture of expert and agent actions."""
        states, actions = [], []
        obs, _ = self.env.reset()
        
        for _ in range(max_steps):
            if self.env.current_step >= len(self.env.data) - 1:
                break
            
            # Get current row for LLM signals
            row = self.env.data.iloc[self.env.current_step]
            
            # Get expert action
            expert_action = self.get_combined_expert_action(obs, row)
            
            # Mix expert and agent actions based on beta
            if random.random() < beta:
                action = expert_action
            else:
                action = self.agent.get_action(obs, deterministic=False)
            
            # Store state and expert action (not the mixed action)
            states.append(obs.copy())
            actions.append(expert_action)
            
            # Take step in environment
            obs, _, done, _, _ = self.env.step(action)
            
            if done:
                break
        
        return states, actions
    
    def train_on_data(self, states, actions, epochs=DAGGER_EPOCHS, batch_size=DAGGER_BATCH_SIZE):
        """Train the agent on collected data."""
        if not states:
            return 0
        
        # Create dataset
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(states),
            torch.LongTensor(actions)
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.agent.train()
        total_loss = 0
        
        for _ in range(epochs):
            epoch_loss = 0
            for state_batch, action_batch in loader:
                self.optimizer.zero_grad()
                
                predictions = self.agent(state_batch)
                loss = nn.CrossEntropyLoss()(predictions, action_batch)
                
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            total_loss += epoch_loss / len(loader)
        
        return total_loss / epochs
    
    def run_dagger(self, iterations=DAGGER_ITERATIONS, trajectory_length=1000):
        """Run DAgger training algorithm."""
        print(f"\nRunning DAgger for {iterations} iterations...")
        
        # Decrease beta over time
        betas = np.linspace(1.0, 0.1, iterations)
        
        for i, beta in enumerate(betas):
            print(f"Iteration {i+1}/{iterations} (β={beta:.2f})")
            
            # Collect trajectory
            states, actions = self.collect_mixed_trajectory(
                beta=beta, max_steps=trajectory_length
            )
            
            # Add to dataset
            self.dataset.extend(zip(states, actions))
            
            # Limit dataset size
            if len(self.dataset) > 20000:
                self.dataset = self.dataset[-20000:]
            
            # Train on dataset
            if self.dataset:
                dataset_states, dataset_actions = zip(*self.dataset)
                loss = self.train_on_data(list(dataset_states), list(dataset_actions))
                self.loss_history.append(loss)
                
                print(f"Loss: {loss:.4f}, Dataset size: {len(self.dataset)}")
        
        print("✅ DAgger completed!")

def main():
    """Main training pipeline."""
    print("=== FinAI Contest 2025 Task 1 Training Pipeline ===")
    
    # Set seeds
    set_seeds(SEED_VALUE)
    
    # Authenticate HuggingFace if needed
    authenticated = authenticate_huggingface()
    
    # Select models to train
    models_to_train = MODELS_TO_TEST if authenticated else ["no_llm"]
    
    all_results = {}
    
    for model_name in models_to_train:
        print(f"\n=== TRAINING {model_name.upper()} ===")
        
        try:
            config = MODELS_CONFIG[model_name]
            
            # Load LLM model if needed
            tokenizer, model, device = None, None, None
            if config['model_id'] is not None:
                tokenizer, model, device = load_llm_model(config)
                if tokenizer is None:
                    print(f"Failed to load {model_name}, skipping...")
                    continue
            
            # Process data
            news_data = process_news_data(NEWS_CSV_PATH, config, tokenizer, model, device)
            merged_data = merge_data(LOB_DATA_PATH, news_data, model_name)
            final_data, feature_columns = prepare_features(merged_data, model_name)
            
            # Save processed data
            output_path = f"data/processed_{model_name}.csv"
            os.makedirs("data", exist_ok=True)
            final_data.to_csv(output_path, index=False)
            print(f"Processed data saved to {output_path}")
            
            # Prepare training data
            final_data = final_data.fillna(method='ffill').fillna(0)
            
            # Split data
            split_idx = int(0.8 * len(final_data))
            train_data = final_data[:split_idx]
            test_data = final_data[split_idx:]
            
            # Scale features
            scaler = StandardScaler()
            unscaled_train_midpoint = train_data['midpoint'].copy()
            unscaled_test_midpoint = test_data['midpoint'].copy()
            
            train_data[feature_columns] = scaler.fit_transform(train_data[feature_columns])
            test_data[feature_columns] = scaler.transform(test_data[feature_columns])
            
            train_data['unscaled_midpoint'] = unscaled_train_midpoint
            test_data['unscaled_midpoint'] = unscaled_test_midpoint
            
            # Create environments
            train_env = CryptoTradingEnv(train_data, feature_columns, lookback_window=LOOKBACK_WINDOW)
            test_env = CryptoTradingEnv(test_data, feature_columns, lookback_window=LOOKBACK_WINDOW)
            vec_env = DummyVecEnv([lambda: train_env])
            
            # Create experts
            ensemble_expert = MixedEnsembleExpert()
            llm_expert = LLMExpert()
            
            # Create and train ensemble
            ensemble_expert.create_mixed_ensemble(vec_env, seed=SEED_VALUE)
            ensemble_expert.train_mixed_ensemble(timesteps=TOTAL_TRAINING_TIMESTEPS)
            
            # Save ensemble models
            ensemble_expert.save_models(f"trained_models/{model_name}_ensemble")
            
            # Create DAgger agent
            agent = DAggerAgent(train_env.observation_space.shape[0], 3)
            
            # Create rational inattention gate
            ri_gate = RIGate(threshold=1.5)
            
            # Train with DAgger
            trainer = DAggerTrainer(agent, ensemble_expert, llm_expert, train_env, rigate=ri_gate)
            trainer.run_dagger(iterations=DAGGER_ITERATIONS)
            
            # Save DAgger agent
            agent.save_model(f"trained_models/{model_name}_dagger.pth")
            
            # Store results
            all_results[model_name] = {
                'loss_history': trainer.loss_history,
                'feature_columns': feature_columns,
                'scaler': scaler
            }
            
            print(f"✅ {model_name} training completed!")
            
        except Exception as e:
            print(f"❌ Error training {model_name}: {e}")
        
        finally:
            # Cleanup
            if model is not None:
                del model, tokenizer
            cleanup_memory()
    
    print("\n=== TRAINING PIPELINE COMPLETED ===")
    return all_results

if __name__ == "__main__":
    results = main()