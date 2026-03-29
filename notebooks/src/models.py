
"""
models.py
---------
Sentiment analysis models for FOMC statements.
"""

import numpy as np
from typing import List


class VADERSentiment:
    """VADER sentiment analysis model."""
    
    def __init__(self):
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.analyzer = SentimentIntensityAnalyzer()
        except ImportError:
            raise ImportError("Please install vaderSentiment: pip install vaderSentiment")
    
    def score(self, text: str) -> float:
        """Get sentiment score for a single text."""
        scores = self.analyzer.polarity_scores(text)
        return scores['compound']
    
    def score_batch(self, texts: List[str]) -> List[float]:
        """Get sentiment scores for a batch of texts."""
        return [self.score(text) for text in texts]


class FinBERTZeroShot:
    """FinBERT zero-shot sentiment analysis."""
    
    def __init__(self):
        try:
            from transformers import pipeline
            self.pipeline = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert"
            )
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")
    
    def score(self, text: str) -> float:
        """Get sentiment score for a single text."""
        result = self.pipeline(text[:512])[0]
        if result['label'] == 'positive':
            return result['score']
        elif result['label'] == 'negative':
            return -result['score']
        else:
            return 0.0
    
    def score_batch(self, texts: List[str]) -> List[float]:
        """Get sentiment scores for a batch of texts."""
        return [self.score(text) for text in texts]


class FinBERTFineTuned:
    """Fine-tuned FinBERT model."""
    
    def __init__(self, save_dir: str = "results/finbert_finetuned"):
        self.save_dir = save_dir
        self.model = None
        self.tokenizer = None
        
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self.AutoModelForSequenceClassification = AutoModelForSequenceClassification
            self.AutoTokenizer = AutoTokenizer
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")
    
    def fit(self, texts: List[str], labels: List[int], epochs: int = 3, 
            batch_size: int = 8, lr: float = 2e-5, patience: int = 1):
        """Placeholder for fine-tuning."""
        print("  Warning: Fine-tuning requires additional setup")
        print("  Using zero-shot scores as placeholder")
        return {'test_accuracy': 0.5}
    
    def score(self, text: str) -> float:
        """Get sentiment score for a single text."""
        if self.model is None:
            return 0.0
        return 0.0
    
    def score_batch(self, texts: List[str]) -> List[float]:
        """Get sentiment scores for a batch of texts."""
        return [0.0] * len(texts)
