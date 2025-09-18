"""Generate merged datasets for each configured LLM expert."""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import timedelta
from typing import Dict, Optional

import pandas as pd
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.config import (
    DATA_DIR,
    FINAL_DATASET_TEMPLATE,
    HF_TOKEN_ENV_KEY,
    LOB_DATA_PATH,
    MODELS_CONFIG,
    MODELS_TO_TEST,
    NEUTRAL_RISK_SCORE,
    NEUTRAL_SENTIMENT_SCORE,
    NEWS_CSV_PATH,
    NEWS_INFLUENCE_MINUTES,
    TOKENIZERS_PARALLELISM,
)

os.environ["TOKENIZERS_PARALLELISM"] = TOKENIZERS_PARALLELISM


def load_hf_token() -> Optional[str]:
    """Load the Hugging Face token from a `.env` file or the environment."""
    load_dotenv()
    token = os.getenv(HF_TOKEN_ENV_KEY)
    return token


@contextmanager
def load_llm(model_id: str, prompt_type: str):
    """Context manager to load and clean up tokenizer/model pairs."""
    device_map = "auto"
    quantization_config = None
    if torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if prompt_type == "llama3" and tokenizer.chat_template is None:
        tokenizer.use_default_system_prompt = True

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        quantization_config=quantization_config,
    )
    try:
        yield tokenizer, model
    finally:
        del model
        del tokenizer
        torch.cuda.empty_cache()


def get_llm_scores(article_text: str, tokenizer, model, prompt_type: str, device: torch.device) -> Optional[Dict[str, int]]:
    """Score a news article using the specified LLM configuration."""
    if not isinstance(article_text, str) or not article_text.strip():
        return None

    instruction = (
        "Analyze the following financial news article regarding Bitcoin. Provide your answer ONLY in valid JSON with keys "
        "'sentiment_score' and 'risk_score' from 1 to 5.\n\n"
        f"Article: '{article_text[:2000]}'\n\nJSON Output:"
    )

    if prompt_type == "gemma":
        messages = [{"role": "user", "content": instruction}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    elif prompt_type == "llama3":
        messages = [
            {"role": "system", "content": "You are a helpful financial analyst."},
            {"role": "user", "content": instruction},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    elif prompt_type == "deepseek":
        prompt = f"### Instruction:\n{instruction}\n### Response:\n"
    else:
        prompt = instruction

    try:
        inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)
        eos_tokens = [tokenizer.eos_token_id]
        if prompt_type == "llama3":
            try:
                eos_tokens.append(tokenizer.convert_tokens_to_ids("<|eot_id|>"))
            except KeyError:
                pass

        output = model.generate(input_ids=inputs, max_new_tokens=100, eos_token_id=eos_tokens)
        decoded = tokenizer.decode(output[0][inputs.shape[-1]:], skip_special_tokens=True)
        json_payload = decoded[decoded.find("{"): decoded.rfind("}") + 1]
        if not json_payload:
            return None
        scores = json.loads(json_payload)
        return {
            "sentiment_score": int(scores["sentiment_score"]),
            "risk_score": int(scores["risk_score"]),
        }
    except Exception:
        return None


def _localize_utc(series: pd.Series) -> pd.Series:
    timestamps = pd.to_datetime(series, errors="coerce")
    if getattr(timestamps.dt, "tz", None) is None:
        return timestamps.dt.tz_localize("UTC")
    return timestamps.dt.tz_convert("UTC")


def score_news_dataframe(model_name: str, token: str) -> pd.DataFrame:
    config = MODELS_CONFIG[model_name]
    news_df = pd.read_csv(NEWS_CSV_PATH)
    news_df["system_time"] = _localize_utc(news_df["date_time"])
    news_df = news_df.sort_values("system_time").reset_index(drop=True)

    if config["model_id"] is None:
        return news_df[["system_time"]].drop_duplicates()

    if not token:
        raise RuntimeError("Hugging Face token is required for LLM-based processing.")

    login(token=token, add_to_git_credential=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tqdm.pandas(desc=f"Scoring {model_name}")
    with load_llm(config["model_id"], config["prompt_type"]) as (tokenizer, model):
        scored = news_df.copy()
        scored["scores"] = scored["article_text"].progress_apply(
            lambda text: get_llm_scores(text, tokenizer, model, config["prompt_type"], device)
        )
    scored = scored.dropna(subset=["scores"])
    if scored.empty:
        return pd.DataFrame({
            "system_time": pd.Series(dtype="datetime64[ns, UTC]"),
            "sentiment_score": pd.Series(dtype="float64"),
            "risk_score": pd.Series(dtype="float64"),
        })

    scores_df = pd.DataFrame(scored["scores"].tolist(), index=scored.index)
    scored["sentiment_score"] = scores_df["sentiment_score"]
    scored["risk_score"] = scores_df["risk_score"]

    aggregated = (
        scored[["system_time", "sentiment_score", "risk_score"]]
        .groupby("system_time")
        .mean()
        .reset_index()
    )
    return aggregated


def merge_lob_and_news(news_features: pd.DataFrame, model_name: str) -> pd.DataFrame:
    lob_df = pd.read_csv(LOB_DATA_PATH)
    lob_df["system_time"] = pd.to_datetime(lob_df["system_time"], utc=True)
    lob_df = lob_df.sort_values("system_time")

    merged = pd.merge_asof(
        lob_df,
        news_features.sort_values("system_time"),
        on="system_time",
        direction="backward",
    )

    if model_name != "no_llm":
        merged["last_news_time"] = merged["system_time"].where(~merged["sentiment_score"].isna()).ffill()
        merged["time_since_news"] = merged["system_time"] - merged["last_news_time"]
        influence_window = timedelta(minutes=NEWS_INFLUENCE_MINUTES)
        expired_mask = merged["time_since_news"] > influence_window
        merged.loc[expired_mask, ["sentiment_score", "risk_score"]] = (
            NEUTRAL_SENTIMENT_SCORE,
            NEUTRAL_RISK_SCORE,
        )
        merged[["sentiment_score", "risk_score"]] = merged[["sentiment_score", "risk_score"]].fillna(
            {
                "sentiment_score": NEUTRAL_SENTIMENT_SCORE,
                "risk_score": NEUTRAL_RISK_SCORE,
            }
        )
        merged = merged.drop(columns=["last_news_time", "time_since_news"], errors="ignore")
    else:
        merged = merged.drop(columns=["sentiment_score", "risk_score"], errors="ignore")

    return merged


def save_final_dataset(dataset: pd.DataFrame, model_name: str) -> Path:
    filename = FINAL_DATASET_TEMPLATE.format(model=model_name)
    output_path = DATA_DIR / filename
    dataset.to_csv(output_path, index=False)
    return output_path


def main():
    token = load_hf_token()
    generated_paths = []
    for model_name in MODELS_TO_TEST:
        print(f"\n=== Processing {model_name.upper()} ===")
        news_features = score_news_dataframe(model_name, token)
        final_dataset = merge_lob_and_news(news_features, model_name)
        output_path = save_final_dataset(final_dataset, model_name)
        print(f"Saved dataset to {output_path}")
        generated_paths.append(output_path)

    if generated_paths:
        print("\nAll datasets generated successfully:")
        for path in generated_paths:
            print(f" - {path}")


if __name__ == "__main__":
    main()

