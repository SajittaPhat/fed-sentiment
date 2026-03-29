
"""
evaluation.py
-------------
Evaluation metrics and visualization functions.
"""

import pandas as pd
import numpy as np
from typing import Dict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import spearmanr


def compute_metrics(scores: np.ndarray, returns: np.ndarray, labels: np.ndarray) -> Dict:
    """Compute evaluation metrics for sentiment scores."""
    # Convert scores to binary predictions
    pred_labels = (scores > 0).astype(int)
    
    # Remove NaN values
    valid = ~np.isnan(labels)
    pred_labels = pred_labels[valid]
    true_labels = labels[valid]
    valid_returns = returns[valid]
    valid_scores = scores[valid]
    
    # Classification metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)
    
    # Correlation
    if len(valid_scores) > 1:
        spearman_r, _ = spearmanr(valid_scores, valid_returns)
    else:
        spearman_r = 0
    
    # Directional accuracy
    directional_acc = np.mean(np.sign(valid_scores) == np.sign(valid_returns))
    
    # Sharpe ratio
    strategy_returns = np.sign(valid_scores) * valid_returns
    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'spearman_r': spearman_r,
        'directional_accuracy': directional_acc,
        'sharpe': sharpe,
        'n_samples': len(valid)
    }


def compare_models(results: Dict[str, Dict]) -> pd.DataFrame:
    """Compare multiple models and return summary DataFrame."""
    summary = []
    for model_name, metrics in results.items():
        summary.append({
            'model': model_name,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'spearman_r': metrics['spearman_r'],
            'directional_accuracy': metrics['directional_accuracy'],
            'sharpe': metrics['sharpe'],
        })
    
    return pd.DataFrame(summary).round(4)
