# factor_mining/data_processing.py
"""Data processing utilities for FinAI Contest 2025 Task 1."""

import pandas as pd
import numpy as np
from datetime import timedelta
from tqdm import tqdm
from .llm_scoring import get_llm_scores
from config import (
    NEWS_INFLUENCE_MINUTES, 
    NEUTRAL_SENTIMENT_SCORE, 
    NEUTRAL_RISK_SCORE,
    NEWS_CSV_PATH,
    LOB_DATA_PATH
)

def process_news_data(news_csv_path: str, model_config: dict, tokenizer=None, model=None, device=None):
    """
    Process news data and extract LLM scores.
    
    Args:
        news_csv_path: Path to news CSV file
        model_config: Model configuration dictionary
        tokenizer: HuggingFace tokenizer (optional)
        model: HuggingFace model (optional)
        device: Device for inference (optional)
        
    Returns:
        pd.DataFrame: Processed news data with sentiment and risk scores
    """
    print(f"Processing news data for {model_config.get('model_id', 'no_llm')}...")
    
    # Load news data
    news_df = pd.read_csv(news_csv_path)
    news_df['system_time'] = pd.to_datetime(news_df['date_time']).dt.tz_convert('UTC')
    news_df = news_df.sort_values('system_time').reset_index(drop=True)
    
    # If using LLM model, extract scores
    if model_config['model_id'] is not None and tokenizer is not None and model is not None:
        print(f"Extracting LLM scores using {model_config['model_id']}...")
        
        tqdm.pandas(desc=f"Scoring {model_config['model_id']}")
        news_df['scores'] = news_df['article_text'].progress_apply(
            lambda x: get_llm_scores(x, tokenizer, model, model_config['prompt_type'], device)
        )
        
        # Remove rows where scoring failed
        news_df = news_df.dropna(subset=['scores'])
        
        if not news_df.empty:
            # Extract scores into separate columns
            scores_df = pd.DataFrame(news_df['scores'].tolist(), index=news_df.index)
            news_df['sentiment_score'] = scores_df['sentiment_score']
            news_df['risk_score'] = scores_df['risk_score']
            
            # Group by time and average scores
            news_for_merge = news_df[['system_time', 'sentiment_score', 'risk_score']]\
                .groupby('system_time').mean().reset_index()
        else:
            # Create empty DataFrame with correct schema
            news_for_merge = pd.DataFrame({
                'system_time': pd.Series(dtype='datetime64[ns, UTC]'),
                'sentiment_score': pd.Series(dtype='float64'),
                'risk_score': pd.Series(dtype='float64')
            })
    else:
        # No LLM processing - just return time information
        news_for_merge = news_df[['system_time']].drop_duplicates()
    
    return news_for_merge

def merge_data(lob_csv_path: str, news_data: pd.DataFrame, model_name: str):
    """
    Merge LOB data with news data and apply temporal filtering.
    
    Args:
        lob_csv_path: Path to LOB CSV file
        news_data: Processed news data
        model_name: Name of the model for identification
        
    Returns:
        pd.DataFrame: Merged dataset ready for training
    """
    print("Merging LOB and news data...")
    
    # Load LOB data
    lob_df = pd.read_csv(lob_csv_path)
    lob_df['system_time'] = pd.to_datetime(lob_df['system_time'], utc=True)
    lob_df = lob_df.sort_values('system_time')
    
    # Merge with news data using backward fill
    merged = pd.merge_asof(lob_df, news_data, on='system_time', direction='backward')
    
    # Apply temporal filtering for LLM models
    if model_name != "no_llm" and 'sentiment_score' in merged.columns:
        print("Applying temporal news influence filtering...")
        
        # Track when news was last available
        merged['last_news_time'] = merged['system_time'].where(
            ~merged['sentiment_score'].isnull()
        ).ffill()
        
        # Calculate time since last news
        merged['time_since_news'] = merged['system_time'] - merged['last_news_time']
        
        # Set influence window
        influence_window = timedelta(minutes=NEWS_INFLUENCE_MINUTES)
        
        # Reset scores to neutral if news is too old
        mask = merged['time_since_news'] > influence_window
        merged.loc[mask, ['sentiment_score', 'risk_score']] = (
            NEUTRAL_SENTIMENT_SCORE, 
            NEUTRAL_RISK_SCORE
        )
        
        # Fill remaining NaN values with neutral scores
        merged[['sentiment_score', 'risk_score']] = merged[['sentiment_score', 'risk_score']]\
            .fillna(NEUTRAL_SENTIMENT_SCORE)
    
    # Clean up temporary columns
    final_df = merged.drop(columns=[
        col for col in ['time_since_news', 'last_news_time'] 
        if col in merged.columns
    ])
    
    return final_df

def prepare_features(df: pd.DataFrame, model_name: str):
    """
    Prepare feature columns for training.
    
    Args:
        df: Merged dataset
        model_name: Model name to determine feature inclusion
        
    Returns:
        tuple: (df, feature_columns)
    """
    print("Preparing features...")
    
    # Define non-feature columns
    non_feature_columns = ['system_time', 'news_time']
    
    # For no_llm model, exclude sentiment/risk scores from features
    if model_name == 'no_llm':
        non_feature_columns.extend(['sentiment_score', 'risk_score'])
    
    # Get feature columns
    feature_columns = [col for col in df.columns if col not in non_feature_columns]
    
    # Create midpoint if not exists
    if 'midpoint' not in df.columns:
        df['midpoint'] = (df['bids_price_0'] + df['asks_price_0']) / 2
        if 'midpoint' not in feature_columns:
            feature_columns.append('midpoint')
    
    print(f"Selected {len(feature_columns)} feature columns")
    
    return df, feature_columns