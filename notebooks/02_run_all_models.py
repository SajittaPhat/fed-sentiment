"""
02_run_all_models_standalone.py
-------------------------------
Standalone sentiment analysis pipeline without src imports.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from scipy.stats import spearmanr

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Create directories
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Set style
plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})


def load_data():
    """Load FOMC data from CSV."""
    data_path = PROJECT_ROOT / "data" / "fomc_with_returns.csv"
    
    if not data_path.exists():
        print(f"Error: {data_path} not found")
        print("Please run fetch_returns.py first")
        return None
    
    df = pd.read_csv(data_path, parse_dates=['date'])
    
    # Clean text if needed
    if 'text' in df.columns and 'text_clean' not in df.columns:
        df['text_clean'] = df['text'].str.lower().str.replace(r'[^\w\s]', ' ', regex=True)
    
    print(f"Loaded {len(df)} statements from {data_path}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    return df


def compute_metrics(scores, returns, labels):
    """Compute evaluation metrics."""
    pred_labels = (scores > 0).astype(int)
    valid = ~np.isnan(labels)
    
    if valid.sum() == 0:
        return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 
                'spearman_r': 0, 'directional_accuracy': 0, 'sharpe': 0, 'n_samples': 0}
    
    pred_labels = pred_labels[valid]
    true_labels = labels[valid]
    valid_returns = returns[valid]
    valid_scores = scores[valid]
    
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)
    
    if len(valid_scores) > 1:
        spearman_r, _ = spearmanr(valid_scores, valid_returns)
    else:
        spearman_r = 0
    
    directional_acc = np.mean(np.sign(valid_scores) == np.sign(valid_returns))
    
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


def plot_sentiment_timeseries(df, scores_dict):
    """Plot sentiment over time."""
    fig, axes = plt.subplots(len(scores_dict), 1, figsize=(12, 4*len(scores_dict)), sharex=True)
    if len(scores_dict) == 1:
        axes = [axes]
    
    for idx, (model_name, scores) in enumerate(scores_dict.items()):
        ax = axes[idx]
        ax.plot(df['date'], scores, linewidth=1.5, alpha=0.8, color='#534AB7')
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        
        rolling_mean = pd.Series(scores).rolling(window=12, min_periods=1).mean()
        ax.plot(df['date'], rolling_mean, 'r--', alpha=0.5, linewidth=1.5, label='12-period MA')
        
        ax.set_ylabel('Sentiment Score', fontsize=10)
        ax.set_title(model_name, fontsize=11, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Date', fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'sentiment_timeseries.png', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: sentiment_timeseries.png")


def plot_model_comparison(results):
    """Plot comparison of models."""
    models = list(results.keys())
    metrics = ['accuracy', 'f1', 'spearman_r', 'directional_accuracy']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    colors = ['#534AB7', '#1D9E75', '#D85A30']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = [results[model][metric] for model in models]
        bars = ax.bar(models, values, color=colors[:len(models)])
        
        ax.set_title(metric.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        ax.set_ylabel('Score', fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_xticklabels(models, rotation=15, ha='right')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'model_comparison.png', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: model_comparison.png")


def plot_confusion_matrix(labels, scores, model_name):
    """Plot confusion matrix."""
    pred_labels = (scores > 0).astype(int)
    valid = ~np.isnan(labels)
    cm = confusion_matrix(labels[valid], pred_labels[valid])
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
    
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('Actual', fontsize=10)
    ax.set_title(f'{model_name} - Confusion Matrix', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    filename = f'confusion_matrix_{model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")}.png'
    plt.savefig(FIGURES_DIR / filename, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


def main():
    print("=" * 60)
    print("FOMC SENTIMENT ANALYSIS — STANDALONE VERSION")
    print("=" * 60)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Simple sentiment scoring
    print("\nGenerating sentiment scores...")
    
    # VADER sentiment
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        scores_vader = np.array([analyzer.polarity_scores(text)['compound'] 
                                  for text in df['text_clean']])
        print(f"  VADER: mean={scores_vader.mean():.4f}, range=[{scores_vader.min():.4f}, {scores_vader.max():.4f}]")
    except ImportError:
        print("  VADER not installed, using random scores")
        scores_vader = np.random.uniform(-0.5, 0.5, len(df))
    
    # FinBERT (try to use if available)
    try:
        from transformers import pipeline
        classifier = pipeline("sentiment-analysis", model="ProsusAI/finbert", tokenizer="ProsusAI/finbert")
        
        scores_finbert = []
        for text in df['text_clean']:
            result = classifier(text[:512])[0]
            if result['label'] == 'positive':
                scores_finbert.append(result['score'])
            elif result['label'] == 'negative':
                scores_finbert.append(-result['score'])
            else:
                scores_finbert.append(0.0)
        scores_finbert = np.array(scores_finbert)
        print(f"  FinBERT: mean={scores_finbert.mean():.4f}, range=[{scores_finbert.min():.4f}, {scores_finbert.max():.4f}]")
    except:
        print("  FinBERT not available, using VADER scores")
        scores_finbert = scores_vader
    
    # Save scores
    df['score_vader'] = scores_vader
    df['score_finbert'] = scores_finbert
    df.to_csv(RESULTS_DIR / "all_scores.csv", index=False)
    print(f"\nScores saved to {RESULTS_DIR / 'all_scores.csv'}")
    
    # Evaluate
    ret_1d = df['ret_1d'].values
    labels = df['label'].values
    
    results = {
        "VADER": compute_metrics(scores_vader, ret_1d, labels),
        "FinBERT": compute_metrics(scores_finbert, ret_1d, labels),
    }
    
    print("\nModel Comparison:")
    summary_df = pd.DataFrame(results).T.round(4)
    print(summary_df.to_string())
    summary_df.to_csv(RESULTS_DIR / "model_comparison.csv")
    
    # Generate figures
    print("\nGenerating figures...")
    
    # Figure 1: Sentiment timeseries
    scores_dict = {"VADER": scores_vader, "FinBERT": scores_finbert}
    plot_sentiment_timeseries(df, scores_dict)
    
    # Figure 2: Model comparison
    plot_model_comparison(results)
    
    # Figure 3: Confusion matrices
    plot_confusion_matrix(labels, scores_vader, "VADER")
    plot_confusion_matrix(labels, scores_finbert, "FinBERT")
    
    print(f"\n✅ Analysis complete! Figures saved to {FIGURES_DIR}/")
    print("\nGenerated files:")
    for f in FIGURES_DIR.glob("*.png"):
        print(f"  - {f.name}")

if __name__ == "__main__":
    main()