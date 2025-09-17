# factor_mining/llm_scoring.py
"""LLM-based factor extraction and rational inattention gate for FinAI Contest 2025 Task 1."""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from config import NEUTRAL_SENTIMENT_SCORE

class RIGate:
    """Rational Inattention Gate to filter LLM signals based on deviation from neutral."""
    
    def __init__(self, threshold: float = 1.0, neutral_score: int = NEUTRAL_SENTIMENT_SCORE):
        """
        Initialize the Rational Inattention Gate.
        
        Args:
            threshold: Minimum deviation required to pay attention
            neutral_score: Neutral score for sentiment and risk (default: 3)
        """
        self.threshold = threshold
        self.neutral = neutral_score
    
    def should_pay_attention(self, sentiment_score: float, risk_score: float) -> bool:
        """
        Determine if the agent should pay attention to LLM signals.
        
        Args:
            sentiment_score: Sentiment score from LLM (1-5)
            risk_score: Risk score from LLM (1-5)
            
        Returns:
            bool: True if deviation exceeds threshold, False otherwise
        """
        deviation = abs(sentiment_score - self.neutral) + abs(risk_score - self.neutral)
        return deviation >= self.threshold

def get_llm_scores(article_text: str, tokenizer, model, prompt_type: str, device) -> dict:
    """
    Extract sentiment and risk scores from financial news using LLM.
    
    Args:
        article_text: News article text
        tokenizer: HuggingFace tokenizer
        model: HuggingFace model
        prompt_type: Type of prompt formatting ("gemma", "llama3", or "deepseek")
        device: Device to run inference on
        
    Returns:
        dict: Contains 'sentiment_score' and 'risk_score' (1-5) or None if failed
    """
    if not isinstance(article_text, str) or not article_text.strip():
        return None
    
    # Create instruction prompt
    instruction = (
        f"Analyze the following financial news article regarding Bitcoin. "
        f"Provide your answer ONLY in valid JSON with keys 'sentiment_score' and 'risk_score' from 1 to 5.\n\n"
        f"Article: '{article_text[:2000]}'\n\n"
        f"JSON Output:"
    )
    
    # Format prompt based on model type
    if prompt_type == "gemma":
        chat = [{"role": "user", "content": instruction}]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    elif prompt_type == "llama3":
        messages = [
            {"role": "system", "content": "You are a helpful financial analyst."},
            {"role": "user", "content": instruction}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:  # deepseek or default
        prompt = f"### Instruction:\n{instruction}\n### Response:\n"
    
    try:
        # Tokenize input
        inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)
        
        # Set termination tokens
        termination_tokens = [tokenizer.eos_token_id]
        if prompt_type == "llama3":
            eot_token = tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if eot_token is not None:
                termination_tokens.append(eot_token)
        
        # Generate response
        with torch.no_grad():
            response = model.generate(
                input_ids=inputs,
                max_new_tokens=100,
                eos_token_id=termination_tokens,
                do_sample=False,
                temperature=0.1
            )
        
        # Decode response
        output = tokenizer.decode(response[0][inputs.shape[-1]:], skip_special_tokens=True)
        
        # Extract JSON from output
        json_start = output.find('{')
        json_end = output.rfind('}') + 1
        json_part = output[json_start:json_end]
        
        if not json_part:
            return None
        
        # Parse JSON
        scores = json.loads(json_part)
        
        return {
            'sentiment_score': int(scores['sentiment_score']),
            'risk_score': int(scores['risk_score'])
        }
        
    except Exception as e:
        print(f"LLM scoring error: {e}")
        return None

def load_llm_model(model_config: dict):
    """
    Load and configure LLM model for inference.
    
    Args:
        model_config: Dictionary containing 'model_id' and 'prompt_type'
        
    Returns:
        tuple: (tokenizer, model, device) or (None, None, None) if failed
    """
    if model_config['model_id'] is None:
        return None, None, None
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Configure quantization for memory efficiency
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_config['model_id'])
        model = AutoModelForCausalLM.from_pretrained(
            model_config['model_id'],
            quantization_config=quantization_config,
            device_map="auto"
        )
        
        # Add padding token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return tokenizer, model, device
        
    except Exception as e:
        print(f"Error loading LLM model: {e}")
        return None, None, None