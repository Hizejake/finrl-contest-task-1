"""Dataset generation script using multiple LLMs.

This script loads `news_train.csv` and `BTC_1min.csv` and creates a new
training dataset by estimating the sentiment of each news headline with a
selected LLM. The resulting CSV will be written to the `datasets/`
folder as `dataset_<model>.csv`.
"""

import argparse
import os
from typing import List

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm

MODEL_MAP = {
    "gemma": "google/gemma-1.1-2b-it",
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "deepseek": "deepseek-ai/deepseek-llm-7b-instruct",
}

def load_pipeline(model_key: str):
    """Load a quantized 4-bit text-generation pipeline."""
    model_id = MODEL_MAP[model_key]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_4bit=True,
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

def infer_sentiments(texts: List[str], pipe):
    """Infer sentiment score using the provided pipeline."""
    sentiments = []
    for text in tqdm(texts, desc="LLM inference"):
        prompt = (
            "Does the following news have positive or negative effect on BTC price? "
            "Respond with `1` for positive and `-1` for negative.\n" + text + "\nAnswer:"
        )
        if pipe is None:
            generated = "0"
        else:
            out = pipe(prompt, max_new_tokens=1)[0]["generated_text"]
            generated = out.split(prompt)[-1].strip()
        if "1" in generated:
            sentiments.append(1)
        elif "-1" in generated:
            sentiments.append(-1)
        else:
            sentiments.append(0)
    return sentiments

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["gemma", "llama3", "deepseek", "none"], default="none")
    parser.add_argument("--news_csv", default="news_train.csv")
    parser.add_argument("--btc_csv", default="BTC_1min.csv")
    parser.add_argument("--out_dir", default="datasets")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    news = pd.read_csv(args.news_csv)
    btc = pd.read_csv(args.btc_csv)

    pipe = None if args.model == "none" else load_pipeline(args.model)
    sentiments = infer_sentiments(news["text"].astype(str).tolist(), pipe)
    news["sentiment"] = sentiments

    merged = btc.merge(news[["timestamp", "sentiment"]], on="timestamp", how="left").fillna(0)
    out_path = os.path.join(args.out_dir, f"dataset_{args.model}.csv")
    merged.to_csv(out_path, index=False)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
