# FinAI Contest 2025: DRL and LLM Agent for Crypto Trading

## Overview
This repository packages a reproducible research workflow for Task I of the FinAI Contest 2025. The solution fuses a **mixed Deep Reinforcement Learning (DRL) expert ensemble** with an **LLM-based news analyst** that is filtered through a Rational Inattention Gate and distilled into a unified **DAgger imitation learner**. The pipeline processes high-frequency limit order book (LOB) data alongside financial news sentiment to generate actionable trading signals for Bitcoin.

## Features
- **Mixed DRL Experts:** A custom ensemble of DQN, PPO, and A2C policies with recurrent feature extractors tailored to LOB time-series structure.
- **Rational Inattention Gate:** Dynamically decides when the LLM sentiment signal should override the DRL experts based on confidence-weighted deviations from neutrality.
- **DAgger Imitation Learning:** Aggregates demonstrations from DRL and LLM experts into a student policy that iteratively refines under distribution shift.
- **Advanced Financial Metrics:** Comprehensive evaluation reporting ROI, annualized Sharpe ratio, maximum drawdown, and trade statistics with comparative plots.

## Repository Structure

- **`data/`** – Storage for raw contest datasets (LOB ticks, news sentiment) and generated merged datasets (one per LLM configuration).

- **`models/`** – Checkpoints produced by the DAgger agent training routine.
- **`src/`** – Modular Python package housing configuration, preprocessing, agent definitions, training, and evaluation scripts.
- **`notebooks/`** – Jupyter notebooks for exploratory analysis and visualization of experiment outputs.

## Setup Instructions
1. **Clone the repository**
   ```bash
   git clone <your_fork_url>
   cd finrl-contest-task-1
   ```
2. **Download datasets**

   Manually obtain `news_train.csv` and `BTC_1sec.csv` from the official contest distribution. You may download the LOB file directly from Google Drive (`https://drive.google.com/file/d/1toXRwp4IbrIZ8PnILQVFuHy9aqlZUKJ8/view?usp=drive_link`). After downloading, copy both files into the `data/` directory.

3. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\\Scripts\\activate
   ```
4. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
5. **Configure Hugging Face access**
   Ensure your Hugging Face account has access to both Gemma and Llama 3 models. Create a `.env` file in the project root with:
   ```text
   HF_TOKEN=your_token_here
   ```

## Usage Instructions
1. **Data Processing**
   Generate the merged LOB + sentiment datasets for every configured LLM expert:
   ```bash
   python src/data_processing.py
   ```
   Each run produces files such as `data/final_dataset_gemma.csv` and `data/final_dataset_no_llm.csv`.

2. **Training**
   Train the DAgger agent using one of the processed datasets (defaults to `gemma`). You can specify a different expert via `--model`:
   ```bash
   python src/train.py --model gemma
   ```
   The trained policy weights are saved to `models/dagger_agent_gemma.pth`.

3. **Evaluation**
   Evaluate all available trained agents and visualize comparative performance:
   ```bash
   python src/evaluate.py
   ```
   The script prints a summary table and renders ROI, portfolio, action distribution, and loss curves for quick diagnosis.

