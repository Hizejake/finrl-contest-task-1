# task1_eval.py
"""Evaluation and ablation study for FinAI Contest 2025 Task 1."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import ttest_rel
from sklearn.preprocessing import StandardScaler

from config import *
from utils import set_seeds, evaluate_agent
from trade_simulator import CryptoTradingEnv
from task1_ensemble import MixedEnsembleExpert, LLMExpert, DAggerAgent, DAggerTrainer
from factor_mining import RIGate

def load_trained_models(model_name="no_llm"):
    """Load trained models for evaluation."""
    print(f"Loading trained models for {model_name}...")
    
    # Load processed data
    data_path = f"data/processed_{model_name}.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed data not found: {data_path}")
    
    df = pd.read_csv(data_path)
    df['system_time'] = pd.to_datetime(df['system_time'])
    df = df.sort_values('system_time').reset_index(drop=True)
    df = df.fillna(method='ffill').fillna(0)
    
    # Prepare features
    non_feature_columns = ['system_time', 'news_time']
    if model_name == 'no_llm':
        non_feature_columns.extend(['sentiment_score', 'risk_score'])
    
    feature_columns = [col for col in df.columns if col not in non_feature_columns]
    
    # Create midpoint if not exists
    if 'midpoint' not in df.columns:
        df['midpoint'] = (df['bids_price_0'] + df['asks_price_0']) / 2
        if 'midpoint' not in feature_columns:
            feature_columns.append('midpoint')
    
    # Split data
    split_idx = int(0.8 * len(df))
    train_data = df[:split_idx]
    test_data = df[split_idx:]
    
    # Scale features
    scaler = StandardScaler()
    unscaled_train_midpoint = train_data['midpoint'].copy()
    unscaled_test_midpoint = test_data['midpoint'].copy()
    
    train_data[feature_columns] = scaler.fit_transform(train_data[feature_columns])
    test_data[feature_columns] = scaler.transform(test_data[feature_columns])
    
    train_data['unscaled_midpoint'] = unscaled_train_midpoint
    test_data['unscaled_midpoint'] = unscaled_test_midpoint
    
    # Create environments
    test_env = CryptoTradingEnv(test_data, feature_columns, lookback_window=LOOKBACK_WINDOW)
    
    # Load DAgger agent
    agent = DAggerAgent(test_env.observation_space.shape[0], 3)
    agent_path = f"trained_models/{model_name}_dagger.pth"
    
    if not agent.load_model(agent_path):
        raise FileNotFoundError(f"DAgger agent not found: {agent_path}")
    
    return agent, test_env, train_data, test_data, feature_columns, scaler

def evaluate_model(model_name="no_llm"):
    """Evaluate a single trained model."""
    print(f"\n=== EVALUATING {model_name.upper()} ===")
    
    try:
        agent, test_env, train_data, test_data, feature_columns, scaler = load_trained_models(model_name)
        
        # Evaluate agent
        roi, portfolio_history, actions = evaluate_agent(agent, test_env)
        
        return {
            'model_name': model_name,
            'roi': roi,
            'portfolio_history': portfolio_history,
            'actions': actions,
            'final_value': portfolio_history[-1],
            'action_counts': {
                'hold': actions.count(0),
                'buy': actions.count(1),
                'sell': actions.count(2)
            }
        }
        
    except Exception as e:
        print(f"❌ Error evaluating {model_name}: {e}")
        return None

def run_experiment(use_ri: bool, seed: int, model_name: str = "no_llm"):
    """Run single experiment for ablation study."""
    try:
        set_seeds(seed)
        
        # Load data
        data_path = f"data/processed_{model_name}.csv"
        df = pd.read_csv(data_path)
        df['system_time'] = pd.to_datetime(df['system_time'])
        df = df.sort_values('system_time').reset_index(drop=True)
        df = df.fillna(method='ffill').fillna(0)
        
        # Prepare features
        non_feature_columns = ['system_time', 'news_time']
        if model_name == 'no_llm':
            non_feature_columns.extend(['sentiment_score', 'risk_score'])
        
        feature_columns = [col for col in df.columns if col not in non_feature_columns]
        
        if 'midpoint' not in df.columns:
            df['midpoint'] = (df['bids_price_0'] + df['asks_price_0']) / 2
            if 'midpoint' not in feature_columns:
                feature_columns.append('midpoint')
        
        # Split and scale data
        split_idx = int(0.8 * len(df))
        train_data, test_data = df[:split_idx], df[split_idx:]
        
        unscaled_train, unscaled_test = train_data['midpoint'].copy(), test_data['midpoint'].copy()
        
        scaler = StandardScaler()
        train_data[feature_columns] = scaler.fit_transform(train_data[feature_columns])
        test_data[feature_columns] = scaler.transform(test_data[feature_columns])
        
        train_data['unscaled_midpoint'] = unscaled_train
        test_data['unscaled_midpoint'] = unscaled_test
        
        # Create environments
        train_env = CryptoTradingEnv(train_data, feature_columns, lookback_window=LOOKBACK_WINDOW)
        test_env = CryptoTradingEnv(test_data, feature_columns, lookback_window=LOOKBACK_WINDOW)
        
        from stable_baselines3.common.vec_env import DummyVecEnv
        vec_env = DummyVecEnv([lambda: train_env])
        
        # Create experts and agent
        ensemble_expert = MixedEnsembleExpert()
        llm_expert = LLMExpert()
        agent = DAggerAgent(train_env.observation_space.shape[0], 3)
        
        # Train ensemble
        ensemble_expert.create_mixed_ensemble(vec_env, seed=seed)
        ensemble_expert.train_mixed_ensemble(timesteps=TOTAL_TRAINING_TIMESTEPS)
        
        # Set RI threshold based on experiment condition
        threshold = 1.5 if use_ri else 9999  # Very high threshold disables RI
        ri_gate = RIGate(threshold=threshold)
        
        # Train with DAgger
        trainer = DAggerTrainer(agent, ensemble_expert, llm_expert, train_env, rigate=ri_gate)
        trainer.run_dagger(iterations=DAGGER_ITERATIONS)
        
        # Evaluate
        roi, _, _ = evaluate_agent(agent, test_env)
        return roi
        
    except Exception as e:
        print(f"[SEED {seed}] Failed: {e}")
        return None

def run_ablation_study(model_name="no_llm", seeds=[10, 20, 30, 40, 50]):
    """Run ablation study comparing with and without rational inattention."""
    print(f"\n=== ABLATION STUDY FOR {model_name.upper()} ===")
    
    results = {"baseline": [], "with_ri": []}
    
    for seed in seeds:
        print(f"\n--- Running seed {seed} without RI ---")
        roi_baseline = run_experiment(use_ri=False, seed=seed, model_name=model_name)
        results["baseline"].append(roi_baseline)
        
        print(f"\n--- Running seed {seed} with RI ---")
        roi_with_ri = run_experiment(use_ri=True, seed=seed, model_name=model_name)
        results["with_ri"].append(roi_with_ri)
    
    # Analyze results
    print("\n--- Ablation Study Results ---")
    print(f"Baseline ROIs: {results['baseline']}")
    print(f"With RI ROIs: {results['with_ri']}")
    
    baseline = np.array([x for x in results['baseline'] if x is not None])
    ri_enabled = np.array([x for x in results['with_ri'] if x is not None])
    
    if len(baseline) > 0 and len(ri_enabled) > 0:
        print(f"\nMean ROI - Baseline: {baseline.mean():.2f} ± {baseline.std():.2f}")
        print(f"Mean ROI - With RI: {ri_enabled.mean():.2f} ± {ri_enabled.std():.2f}")
        
        # Statistical significance test
        if len(baseline) == len(ri_enabled):
            t_stat, p_val = ttest_rel(ri_enabled, baseline)
            print(f"\nPaired t-test: t = {t_stat:.3f}, p = {p_val:.4f}")
            
            if p_val < 0.05:
                print("✅ Difference is statistically significant!")
            else:
                print("⚠️  Difference may not be statistically significant.")
        else:
            print("⚠️  Mismatch in trial counts; cannot perform paired t-test.")
    
    return results

def visualize_results(results_list):
    """Visualize evaluation results."""
    if not results_list:
        print("No results to visualize.")
        return
    
    # Filter out None results
    results_list = [r for r in results_list if r is not None]
    
    if not results_list:
        print("No valid results to visualize.")
        return
    
    model_names = [r['model_name'] for r in results_list]
    rois = [r['roi'] for r in results_list]
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
    
    # ROI comparison
    axes[0, 0].bar(model_names, rois, color=colors)
    axes[0, 0].set_title('ROI Comparison')
    axes[0, 0].set_ylabel('ROI (%)')
    axes[0, 0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Portfolio evolution
    for i, result in enumerate(results_list):
        axes[0, 1].plot(result['portfolio_history'], 
                       label=result['model_name'].upper(), 
                       color=colors[i], linewidth=2)
    axes[0, 1].set_title('Portfolio Value Over Time')
    axes[0, 1].set_ylabel('Portfolio Value ($)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Action distribution
    action_data = {}
    for result in results_list:
        action_data[result['model_name']] = [
            result['action_counts']['hold'],
            result['action_counts']['buy'],
            result['action_counts']['sell']
        ]
    
    action_df = pd.DataFrame(action_data, index=['Hold', 'Buy', 'Sell'])
    action_df.plot(kind='bar', ax=axes[1, 0], color=colors, rot=0)
    axes[1, 0].set_title('Action Distribution')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].legend(title='Models')
    
    # Summary statistics
    axes[1, 1].axis('off')
    summary_text = "Summary Statistics:\n\n"
    for result in results_list:
        summary_text += f"{result['model_name'].upper()}:\n"
        summary_text += f"  ROI: {result['roi']:.2f}%\n"
        summary_text += f"  Final Value: ${result['final_value']:.2f}\n"
        summary_text += f"  Total Actions: {sum(result['action_counts'].values())}\n\n"
    
    axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('evaluation_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Visualization saved as 'evaluation_results.png'")

def main():
    """Main evaluation pipeline."""
    print("=== FinAI Contest 2025 Task 1 Evaluation ===")
    
    # Evaluate trained models
    models_to_evaluate = ["no_llm"]  # Add other models if available
    
    # Check for available models
    available_models = []
    for model_name in MODELS_TO_TEST:
        data_path = f"data/processed_{model_name}.csv"
        agent_path = f"trained_models/{model_name}_dagger.pth"
        
        if os.path.exists(data_path) and os.path.exists(agent_path):
            available_models.append(model_name)
    
    if not available_models:
        print("❌ No trained models found. Please run task1_ensemble.py first.")
        return
    
    print(f"Found trained models: {available_models}")
    
    # Evaluate all available models
    results = []
    for model_name in available_models:
        result = evaluate_model(model_name)
        if result:
            results.append(result)
    
    # Visualize results
    if results:
        visualize_results(results)
    
    # Run ablation study on the first available model
    if available_models:
        model_for_ablation = available_models[0]
        print(f"\nRunning ablation study on {model_for_ablation}...")
        ablation_results = run_ablation_study(model_for_ablation)
    
    print("\n=== EVALUATION COMPLETED ===")
    return results

if __name__ == "__main__":
    results = main()