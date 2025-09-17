# trade_simulator.py
"""Trading environment for FinAI Contest 2025 Task 1."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from config import INITIAL_BALANCE, TRANSACTION_COST, LOOKBACK_WINDOW

class CryptoTradingEnv(gym.Env):
    """
    Custom Gym environment for cryptocurrency trading.
    
    Actions:
    - 0: Hold
    - 1: Buy
    - 2: Sell
    """
    
    def __init__(self, data, all_feature_columns, 
                 initial_balance=INITIAL_BALANCE, 
                 transaction_cost=TRANSACTION_COST, 
                 lookback_window=LOOKBACK_WINDOW):
        super().__init__()
        
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.lookback_window = lookback_window
        self.feature_columns = all_feature_columns
        
        # Action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: flattened feature vectors for lookback window
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf,
            shape=(self.lookback_window * len(self.feature_columns),), 
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.btc_held = 0
        self.net_worth_history = [self.initial_balance]
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Get the current observation (flattened feature vectors for lookback window)."""
        if self.current_step >= len(self.data):
            self.current_step = len(self.data) - 1
            
        # Get the last lookback_window rows
        start_idx = self.current_step - self.lookback_window
        end_idx = self.current_step
        
        frame = self.data.iloc[start_idx:end_idx]
        
        # Return flattened feature vectors
        return frame[self.feature_columns].values.flatten().astype(np.float32)
    
    def step(self, action):
        """Execute one trading step."""
        # Check if we've reached the end of data
        if self.current_step >= len(self.data) - 1:
            return self._get_observation(), 0, True, True, {}
        
        # Get current price
        current_price = self.data.loc[self.current_step, 'unscaled_midpoint']
        
        # Execute trading action
        if action == 1 and self.balance > 0:  # Buy
            btc_bought = (self.balance / current_price) * (1 - self.transaction_cost)
            self.btc_held += btc_bought
            self.balance = 0
            
        elif action == 2 and self.btc_held > 0:  # Sell
            cash_received = self.btc_held * current_price * (1 - self.transaction_cost)
            self.balance += cash_received
            self.btc_held = 0
        
        # Move to next step
        self.current_step += 1
        
        # Calculate portfolio value at new step
        next_price = self.data.loc[self.current_step, 'unscaled_midpoint']
        portfolio_value = self.balance + self.btc_held * next_price
        
        # Calculate reward as percentage change in portfolio value
        reward = (portfolio_value - self.net_worth_history[-1]) / self.initial_balance
        
        # Update history
        self.net_worth_history.append(portfolio_value)
        
        # Check if episode is done
        done = self.current_step >= len(self.data) - 1
        
        info = {'portfolio_value': portfolio_value}
        
        return self._get_observation(), reward, done, False, info