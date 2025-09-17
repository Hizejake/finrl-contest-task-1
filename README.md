# FinAI Contest 2025 - Task 1: FinRL-DeepSeek Crypto Trading

This repository contains our solution for **Task 1** of the FinAI Contest 2025, implementing a crypto trading agent that integrates LLM-generated signals with reinforcement learning using a novel **Rational Inattention** mechanism.

## 🚀 Key Features

- **Multi-Modal LLM Integration**: Supports Gemma, LLaMA-3, and DeepSeek models for sentiment and risk analysis
- **Rational Inattention Gate**: Novel mechanism to filter LLM signals based on deviation from neutral sentiment
- **Ensemble RL Experts**: Combines DQN, PPO, and A2C using majority voting
- **DAgger Training**: Imitation learning from expert demonstrations with decreasing beta schedule
- **Comprehensive Evaluation**: Includes ablation studies and statistical significance testing

## 📁 Repository Structure

```
finrl-contest-task-1/
├── trained_models/           # Trained model weights
├── factor_mining/            # LLM signal extraction and processing
│   ├── __init__.py
│   ├── llm_scoring.py       # LLM inference and Rational Inattention Gate
│   └── data_processing.py   # Data merging and feature preparation
├── data/                    # Processed datasets (created during training)
├── config.py               # Configuration parameters
├── utils.py                # Utility functions
├── trade_simulator.py      # Custom trading environment
├── task1_ensemble.py       # Main training pipeline
├── task1_eval.py          # Evaluation and ablation studies
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🔧 Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd finrl-contest-task-1
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Prepare data**:
   - Place your `news_train.csv` file in the `data/` directory
   - Place your `BTC_1min.csv` file in the `data/` directory
   - Update paths in `config.py` if needed

4. **Set up Hugging Face authentication** (if using LLM models):
   - Obtain a Hugging Face token
   - Set it as `HF_TOKEN` in your environment or Kaggle secrets

## 🏃‍♂️ Quick Start

### Training

Run the complete training pipeline:

```bash
python task1_ensemble.py
```

This will:
1. Process news data with LLM models (if available)
2. Merge with LOB data and apply temporal filtering
3. Train ensemble RL experts (DQN, PPO, A2C)
4. Train DAgger agent with Rational Inattention
5. Save all trained models to `trained_models/`

### Evaluation

Evaluate trained models and run ablation studies:

```bash
python task1_eval.py
```

This will:
1. Load trained models
2. Evaluate on test data
3. Generate performance visualizations
4. Run ablation study comparing with/without Rational Inattention
5. Perform statistical significance testing

## 🧠 Methodology

### 1. Factor Mining (LLM Signals)

Our approach extracts sentiment and risk scores (1-5 scale) from financial news using state-of-the-art LLMs:

- **Gemma-7B-IT**: Google's instruction-tuned model
- **LLaMA-3-8B-Instruct**: Meta's instruction-following model  
- **DeepSeek-Coder-6.7B**: DeepSeek's code-instruction model

**Prompt Engineering**: We use model-specific chat templates with structured JSON output requirements.

### 2. Rational Inattention Gate

Novel mechanism inspired by economic theory that filters LLM signals based on information value:

```python
deviation = |sentiment - 3| + |risk - 3|
should_attend = deviation >= threshold
```

This prevents the agent from paying attention to neutral or low-information signals, improving computational efficiency and decision quality.

### 3. Expert Ensemble

**Mixed Ensemble Expert**: Combines three RL algorithms using majority voting:
- **DQN**: Deep Q-Network for value-based learning
- **PPO**: Proximal Policy Optimization for stable policy gradients  
- **A2C**: Advantage Actor-Critic for immediate policy updates

**LLM Expert**: Rule-based expert that makes trading decisions from sentiment/risk scores:
- Buy: sentiment ≥ 4 AND risk ≤ 2
- Sell: sentiment ≤ 2 OR risk ≥ 4
- Hold: otherwise

### 4. DAgger Training

**Dataset Aggregation (DAgger)** with rational inattention:

1. **Expert Combination**: Merges ensemble and LLM experts using rational inattention gate
2. **Beta Schedule**: Decreasing mixture coefficient from 1.0 → 0.1 over iterations
3. **Experience Replay**: Maintains dataset of expert demonstrations with size limiting

### 5. Trading Environment

Custom Gym environment with:
- **Actions**: Hold (0), Buy (1), Sell (2)
- **Observations**: Flattened LOB features over lookback window
- **Rewards**: Portfolio value changes normalized by initial balance
- **Transaction Costs**: 0.1% per trade

## ⚙️ Configuration

Key parameters in `config.py`:

```python
# Model selection
MODELS_TO_TEST = ["gemma", "llama3", "deepseek", "no_llm"]

# Training parameters
TOTAL_TRAINING_TIMESTEPS = 60000
DAGGER_ITERATIONS = 10
LOOKBACK_WINDOW = 60

# Rational Inattention
NEWS_INFLUENCE_MINUTES = 60  # Temporal filtering window
NEUTRAL_SENTIMENT_SCORE = 3  # Neutral point for RI calculation

# Environment
INITIAL_BALANCE = 10000
TRANSACTION_COST = 0.001
```

## 📊 Results

### Performance Metrics

Our approach demonstrates:
- **Consistent positive ROI** across different model configurations
- **Statistical significance** of Rational Inattention mechanism
- **Robust performance** with ensemble voting reducing variance

### Ablation Study

The evaluation includes comprehensive ablation testing:
- **Baseline**: Ensemble expert only
- **With RI**: Ensemble + LLM + Rational Inattention
- **Statistical Testing**: Paired t-tests across multiple random seeds

## 🔬 Technical Details

### Data Processing Pipeline

1. **News Processing**: Extract LLM scores with progress tracking
2. **Temporal Alignment**: Backward-fill merge with LOB data
3. **Influence Filtering**: Apply 60-minute news influence window
4. **Feature Scaling**: StandardScaler normalization
5. **Train/Test Split**: 80/20 temporal split

### Model Architecture

**DAgger Agent**:
- Input: Flattened LOB features (lookback_window × n_features)
- Hidden: 512 → 512 → 256 neurons with ReLU + Dropout(0.4)
- Output: 3-class softmax for action probabilities

### Memory Management

- **4-bit Quantization**: BitsAndBytesConfig for LLM efficiency
- **GPU Cleanup**: Automatic memory management between models
- **Dataset Limiting**: Maximum 20K experiences in replay buffer

## 🎯 Competition Compliance

This submission meets all FinAI Contest 2025 requirements:

- ✅ **File Structure**: Follows specified directory layout
- ✅ **Trained Models**: All weights saved in `trained_models/`
- ✅ **Evaluation Code**: Complete evaluation pipeline in `task1_eval.py`
- ✅ **Environment**: Custom environment in `trade_simulator.py`
- ✅ **Documentation**: Comprehensive README and code comments
- ✅ **Reproducibility**: Fixed seeds and detailed configuration

## 🚧 Troubleshooting

### Common Issues

1. **HuggingFace Authentication**:
   ```bash
   # Set token manually
   export HF_TOKEN="your_token_here"
   ```

2. **Memory Issues**:
   - Reduce batch size in `config.py`
   - Use CPU-only mode by modifying device detection

3. **Missing Data**:
   - Ensure data files are in correct locations
   - Update paths in `config.py`

### Hardware Requirements

- **Minimum**: 8GB RAM, 4GB VRAM
- **Recommended**: 16GB RAM, 8GB VRAM
- **Training Time**: ~2-4 hours depending on hardware

## 📖 References

1. Ross, S., Gordon, G., & Bagnell, D. (2011). A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning. *AISTATS*.

2. Sims, C. A. (2003). Implications of rational inattention. *Journal of Monetary Economics*, 50(3), 665-690.

3. Liu, X. Y., et al. (2020). FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading. *NeurIPS Deep RL Workshop*.

## 👥 Team

This solution was developed for the FinAI Contest 2025, implementing novel combinations of LLM-based factor extraction with reinforcement learning for cryptocurrency trading.

---

For questions or issues, please refer to the contest documentation or create an issue in this repository.