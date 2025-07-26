# 🧠 FinAI Contest 2025 - Task 1 Submission

This repository contains our solution for Task 1 of the FinAI Contest 2025.

## 📁 Folder Structure
- `factor_mining/`: LLM-based signal extraction + dataset generation
- `trained_models/`: Saved RL and DAgger models
- `task1_ensemble.py`: DAgger + Ensemble logic
- `task1_eval.py`: Evaluation code
- `trade_simulator.py`: Custom Gym environment
- `run_dagger_training.py`: Run DAgger for selected dataset
- `final_comparison_plot.py`: Portfolio comparison plot
- `llm_configs/*.sh`: Scripts to run all model versions
- `datasets/`: Generated CSVs

## 🚀 Instructions

1. Generate all datasets:
```bash
bash llm_configs/generate_gemma.sh
bash llm_configs/generate_llama3.sh
bash llm_configs/generate_deepseek.sh
bash llm_configs/generate_none.sh
```

2. Train the DAgger agent:
```bash
python run_dagger_training.py --model gemma
# Repeat for each model
```

3. Evaluate and compare:
```bash
python task1_eval.py
python final_comparison_plot.py
```

## 🛠️ Requirements
```bash
pip install -r requirements.txt
```
