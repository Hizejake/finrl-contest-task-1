# =============================================================================
# Multi-Model Data Generator with LOB Data & News Influence Windows (v23 - Final Fix)
# =============================================================================
"""Generate continuous datasets using multiple instruction-tuned LLMs.

This script mirrors the Kaggle notebook used for the FinAI Contest. It loads
`news_train.csv` and `BTC_1min.csv`, scores each news headline with an LLM and
merges the results with limit order book (LOB) data. Four-bit quantised models
are used for inference. The resulting merged dataset is saved to the
``datasets`` folder.
"""

# --- 0. Install necessary libraries ---
# The following commands were run in the original Kaggle notebook. They are left
# as comments for reference.
# !pip uninstall -y transformers
# !pip install pandas tqdm transformers torch accelerate bitsandbytes huggingface_hub

import argparse
import os
import json
from datetime import timedelta
from typing import Optional, Dict

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
from kaggle_secrets import UserSecretsClient
import pkg_resources

os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("--- ✅ All libraries installed and imported. ---")

# --- Self-Diagnostic Version Check ---
try:
    required_version = pkg_resources.parse_version("4.41.0")
    import transformers
    installed_version = pkg_resources.parse_version(transformers.__version__)
    if installed_version < required_version:
        print(f"❌ ERROR: Transformers version {transformers.__version__} too old.")
        print("➡️ Please restart the session and install >= 4.41.0.")
        raise SystemExit
    else:
        print(f"✅ Transformers version check passed ({transformers.__version__}).")
except Exception as e:
    print(f"Could not check transformers version: {e}")


# --- 1. Authenticate with Hugging Face ---
print("\n--- 1. Authenticating with Hugging Face ---")
try:
    user_secrets = UserSecretsClient()
    hf_token = user_secrets.get_secret("HF_TOKEN")
    login(token=hf_token)
    print("✅ Successfully authenticated with Hugging Face.")
except Exception:
    print("❌ HF_TOKEN secret not found. Please add your Hugging Face token as a secret.")
    raise SystemExit

# --- 2. Configuration & Model Selection ---
print("\n--- 2. Configuring the data generation pipeline ---")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    default="gemma",
    choices=["gemma", "llama3", "deepseek", "none"],
    help="Which LLM to use for scoring (or 'none' for baseline)",
)
parser.add_argument("--news_csv", default="news_train.csv")
parser.add_argument("--lob_csv", default="BTC_1min.csv")
args = parser.parse_args()

SELECTED_MODEL = args.model

MODELS_CONFIG: Dict[str, Dict[str, str]] = {
    "gemma": {"model_id": "google/gemma-7b-it", "prompt_type": "gemma"},
    "llama3": {"model_id": "meta-llama/Meta-Llama-3-8B-Instruct", "prompt_type": "llama3"},
    "deepseek": {"model_id": "deepseek-ai/deepseek-coder-6.7b-instruct", "prompt_type": "deepseek"},
    "none": {"model_id": None, "prompt_type": "none"},
}

MODEL_NAME = MODELS_CONFIG[SELECTED_MODEL]["model_id"]
PROMPT_TYPE = MODELS_CONFIG[SELECTED_MODEL]["prompt_type"]

NEWS_CSV_PATH = args.news_csv
LOB_DATA_PATH = args.lob_csv
OUTPUT_DIR = "datasets"
OUTPUT_CSV_PATH = os.path.join(OUTPUT_DIR, f"dataset_{SELECTED_MODEL}.csv")

NEWS_INFLUENCE_MINUTES = 60
NEUTRAL_SENTIMENT_SCORE = 3
NEUTRAL_RISK_SCORE = 3

model: Optional[AutoModelForCausalLM] = None
tokenizer: Optional[AutoTokenizer] = None
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {device}")
print(f"✅ Selected model: {SELECTED_MODEL}")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

if MODEL_NAME is not None:
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quantization_config,
            device_map="auto",
        )
        print(f"✅ LLM model '{MODEL_NAME}' loaded with 4-bit quantization.")
    except Exception as e:
        print(f"❌ Failed to load LLM model: {e}")
        model = None
        tokenizer = None
else:
    print("⚠️ No LLM selected; generating baseline dataset.")


def get_llm_scores(article_text: str) -> Optional[Dict[str, int]]:
    """Query the LLM to obtain sentiment and risk scores."""
    if (
        not isinstance(article_text, str)
        or not article_text.strip()
        or tokenizer is None
        or model is None
    ):
        return None

    base_instruction = (
        "Analyze the following financial news article regarding Bitcoin. "
        "Provide your answer ONLY in a valid JSON format with two keys: "
        "'sentiment_score' and 'risk_score'. The scores must be an integer from 1 "
        "(very negative/high risk) to 5 (very positive/low risk).\n\nArticle: "
        f"\"{article_text[:2000]}\"\n\nJSON Output:"
    )

    if PROMPT_TYPE == "gemma":
        chat = [{"role": "user", "content": base_instruction}]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    elif PROMPT_TYPE == "llama3":
        messages = [
            {"role": "system", "content": "You are a helpful financial analyst providing structured JSON output."},
            {"role": "user", "content": base_instruction},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    elif PROMPT_TYPE == "deepseek":
        prompt = f"### Instruction:\n{base_instruction}\n### Response:\n"
    else:
        prompt = base_instruction

    try:
        inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)
        terminators = [tokenizer.eos_token_id]
        if PROMPT_TYPE == "llama3":
            terminators.append(tokenizer.convert_tokens_to_ids("<|eot_id|>"))

        response = model.generate(input_ids=inputs, max_new_tokens=100, eos_token_id=terminators)
        output_text = tokenizer.decode(response[0][inputs.shape[-1]:], skip_special_tokens=True)
        json_part = output_text[output_text.find('{'):output_text.rfind('}') + 1]
        if not json_part:
            return None
        scores = json.loads(json_part)
        if 'sentiment_score' in scores and 'risk_score' in scores:
            return {'sentiment_score': int(scores['sentiment_score']), 'risk_score': int(scores['risk_score'])}
    except Exception:
        return None
    return None


print("\n--- 3. Loading and preparing local datasets ---")

if not os.path.exists(NEWS_CSV_PATH):
    print(f"❌ News file '{NEWS_CSV_PATH}' not found.")
    raise SystemExit
news_df = pd.read_csv(NEWS_CSV_PATH)
news_df['system_time'] = pd.to_datetime(news_df['date_time']).dt.tz_convert('UTC')
news_df = news_df.sort_values('system_time').reset_index(drop=True)
print(f"✅ News dataset loaded with {len(news_df)} articles.")

if not os.path.exists(LOB_DATA_PATH):
    print(f"❌ LOB data file '{LOB_DATA_PATH}' not found.")
    raise SystemExit
lob_df = pd.read_csv(LOB_DATA_PATH)

if 'system_time' not in lob_df.columns:
    print("❌ 'system_time' column not found in the LOB data file.")
    raise SystemExit
lob_df['system_time'] = pd.to_datetime(lob_df['system_time'])
if lob_df['system_time'].dt.tz is not None:
    lob_df['system_time'] = lob_df['system_time'].dt.tz_convert('UTC')
else:
    lob_df['system_time'] = lob_df['system_time'].dt.tz_localize('UTC')

lob_df = lob_df.sort_values('system_time')
print(f"✅ LOB market data loaded with {len(lob_df)} rows.")

if model is not None and tokenizer is not None:
    print("\n--- 4. Generating LLM scores for all news articles ---")
    tqdm.pandas(desc=f"Generating Scores with {SELECTED_MODEL}")
    news_df['scores'] = news_df['article_text'].progress_apply(get_llm_scores)
    news_df = news_df.dropna(subset=['scores'])
    if news_df.empty:
        print("⚠️ LLM failed to generate valid scores. Proceeding with neutral scores only.")
        news_for_merge = pd.DataFrame({'system_time': news_df['system_time'],
                                      'sentiment_score': NEUTRAL_SENTIMENT_SCORE,
                                      'risk_score': NEUTRAL_RISK_SCORE})
    else:
        scores_df = pd.DataFrame(news_df['scores'].tolist(), index=news_df.index)
        if 'sentiment_score' in scores_df.columns and 'risk_score' in scores_df.columns:
            news_df['sentiment_score'] = scores_df['sentiment_score']
            news_df['risk_score'] = scores_df['risk_score']
            news_for_merge = news_df[['system_time', 'sentiment_score', 'risk_score']].copy()
            news_for_merge = news_for_merge.groupby('system_time').mean().reset_index()
        else:
            print("⚠️ LLM output missing expected columns. Using neutral scores only.")
            news_for_merge = pd.DataFrame({'system_time': news_df['system_time'],
                                          'sentiment_score': NEUTRAL_SENTIMENT_SCORE,
                                          'risk_score': NEUTRAL_RISK_SCORE})
elif SELECTED_MODEL == "none":
    print("\n--- 4. Using neutral scores only (no LLM) ---")
    news_for_merge = news_df[['system_time']].copy()
    news_for_merge['sentiment_score'] = NEUTRAL_SENTIMENT_SCORE
    news_for_merge['risk_score'] = NEUTRAL_RISK_SCORE
else:
    print("\nSkipping data generation because the LLM failed to load.")
    raise SystemExit

print("\n--- 5. Merging LOB data with most recent news ---")
merged_df = pd.merge_asof(lob_df, news_for_merge, on='system_time', direction='backward')

print("\n--- 6. Applying news influence window ---")
news_time_map = news_for_merge.set_index('system_time')
merged_df['last_news_time'] = news_time_map.index.to_series().reindex(merged_df['system_time'], method='ffill')
merged_df['time_since_news'] = merged_df['system_time'] - merged_df['last_news_time']
influence_window = timedelta(minutes=NEWS_INFLUENCE_MINUTES)
neutral_mask = merged_df['time_since_news'] > influence_window
merged_df.loc[neutral_mask, 'sentiment_score'] = NEUTRAL_SENTIMENT_SCORE
merged_df.loc[neutral_mask, 'risk_score'] = NEUTRAL_RISK_SCORE
merged_df['sentiment_score'] = merged_df['sentiment_score'].fillna(NEUTRAL_SENTIMENT_SCORE)
merged_df['risk_score'] = merged_df['risk_score'].fillna(NEUTRAL_RISK_SCORE)

print("\n--- 7. Assembling final dataset ---")
final_df = merged_df.drop(columns=['time_since_news', 'last_news_time'])
final_df = final_df[[col for col in final_df.columns if col != 'Unnamed: 0']]

os.makedirs(OUTPUT_DIR, exist_ok=True)
if final_df.empty:
    print("❌ No data was generated. Ensure news and LOB data overlap in time.")
else:
    final_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"✅ Generated dataset with {len(final_df)} rows -> {OUTPUT_CSV_PATH}")
    print("\nFinal Dataset Preview:")
    print(final_df.head())

print("\n--- 8. Cleaning up GPU memory ---")
if 'model' in locals() and model is not None:
    del model
if 'tokenizer' in locals() and tokenizer is not None:
    del tokenizer
torch.cuda.empty_cache()
print("✅ Memory cleanup complete.")
