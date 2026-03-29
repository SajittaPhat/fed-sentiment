"""
03_topic_modelling.py
---------------------
Unsupervised topic modelling on FOMC statements using LDA.
Validates results via coherence scores across k=2..10 topics.
Produces a stacked-area topic evolution chart.

Usage:
    python notebooks/03_topic_modelling.py
"""

import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

# Create directories
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
FIGURE_DIR = Path("results/figures")
FIGURE_DIR.mkdir(exist_ok=True)

# Define stopwords
EXTRA_STOPWORDS = {
    "committee", "federal", "reserve", "board", "meeting",
    "vote", "voted", "member", "members", "bank", "system",
    "open", "market", "monetary", "policy", "percent",
    "basis", "points", "rate", "rates", "december", "january",
    "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november",
}
ALL_STOPS = STOPWORDS.union(EXTRA_STOPWORDS)


def clean_text(text):
    """Simple text cleaning function."""
    if not isinstance(text, str):
        return ""
    import re
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = ' '.join(text.split())
    return text


def tokenise(text: str) -> list[str]:
    """Tokenize and clean text."""
    tokens = simple_preprocess(text, deacc=True)
    return [t for t in tokens if t not in ALL_STOPS and len(t) > 2]


def find_optimal_k(corpus, dictionary, texts_tokenised, k_range=range(2, 11)):
    """Grid search over number of topics using Cv coherence."""
    scores = {}
    
    print("  Testing k values:", list(k_range))
    for k in k_range:
        try:
            model = LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=k,
                random_state=42,
                passes=10,
                alpha="auto",
                iterations=100,  # Add iterations to avoid convergence issues
            )
            
            cm = CoherenceModel(
                model=model, 
                texts=texts_tokenised,
                dictionary=dictionary, 
                coherence="c_v"
            )
            scores[k] = cm.get_coherence()
            print(f"    k={k:2d}  coherence={scores[k]:.4f}")
            
        except Exception as e:
            print(f"    k={k:2d}  Error: {str(e)[:50]}")
            scores[k] = 0.0

    # Plot coherence
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(scores.keys()), list(scores.values()),
            marker="o", color="#534AB7", linewidth=2, markersize=8)
    ax.set_xlabel("Number of topics (k)", fontsize=10)
    ax.set_ylabel("Coherence score (Cv)", fontsize=10)
    ax.set_title("LDA Coherence by Number of Topics", fontsize=12, fontweight="bold")
    
    if scores:
        best_k = max(scores, key=scores.get)
        ax.axvline(best_k, color="#D85A30",
                   linewidth=1.5, linestyle="--", 
                   label=f"Optimal k={best_k} (coherence={scores[best_k]:.4f})")
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "lda_coherence.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"\n  Saved: {FIGURE_DIR / 'lda_coherence.png'}")
    
    if scores:
        best_k = max(scores, key=scores.get)
        print(f"\n  Optimal k = {best_k} (coherence = {scores[best_k]:.4f})")
        return best_k
    else:
        print("\n  Warning: No valid coherence scores, using k=5 as default")
        return 5


def plot_topic_evolution(df, topic_cols, topic_labels):
    """Plot topic evolution over time as stacked area chart."""
    if len(topic_cols) == 0:
        print("  No topics to plot")
        return
    
    # Prepare data
    df = df.copy()
    df['year'] = df['date'].dt.year
    yearly_topics = df.groupby('year')[topic_cols].mean()
    
    # Create stacked area plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Use a color palette
    colors = plt.cm.Set3(np.linspace(0, 1, len(topic_cols)))
    
    ax.stackplot(yearly_topics.index, yearly_topics.T.values,
                 labels=[label[:30] for label in topic_labels],
                 colors=colors, alpha=0.8)
    
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Average Topic Proportion", fontsize=11)
    ax.set_title("Topic Evolution in FOMC Statements", fontsize=13, fontweight="bold")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "topic_evolution.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {FIGURE_DIR / 'topic_evolution.png'}")


def print_topic_summary(lda, num_topics, num_words=8):
    """Print topic summary with top words."""
    print("\n" + "=" * 70)
    print("TOPIC SUMMARY")
    print("=" * 70)
    
    topic_labels = []
    for i in range(num_topics):
        words = [w for w, _ in lda.show_topic(i, topn=num_words)]
        label = f"Topic {i+1}: {', '.join(words[:3])}"
        topic_labels.append(label)
        print(f"\n  Topic {i+1}:")
        print(f"    Keywords: {', '.join(words)}")
    
    return topic_labels


def main():
    print("=" * 60)
    print("LDA TOPIC MODELLING")
    print("=" * 60)
    
    # Check if data exists
    data_path = Path("../data/fomc_with_returns.csv")
    if not data_path.exists():
        print(f"\nError: {data_path} not found")
        print("Please run fetch_returns.py first")
        return
    
    # Load cleaned text
    print("\n[1/5] Loading data...")
    df = pd.read_csv(data_path, parse_dates=["date"])
    print(f"  Loaded {len(df)} statements")
    
    # Clean text
    print("\n[2/5] Cleaning text...")
    df["text_clean"] = df["text"].apply(clean_text)
    
    # Tokenise
    print("\n[3/5] Tokenizing text...")
    texts_tok = [tokenise(t) for t in df["text_clean"]]
    
    # Remove empty documents
    non_empty = [len(t) > 0 for t in texts_tok]
    print(f"  {sum(non_empty)} documents have content after cleaning")
    texts_tok = [t for t in texts_tok if len(t) > 0]
    df = df[non_empty].reset_index(drop=True)
    
    # Create dictionary and corpus
    dictionary = corpora.Dictionary(texts_tok)
    dictionary.filter_extremes(no_below=3, no_above=0.85)
    corpus = [dictionary.doc2bow(t) for t in texts_tok]
    
    print(f"  Vocabulary size: {len(dictionary)}")
    print(f"  Average tokens per doc: {np.mean([len(t) for t in texts_tok]):.1f}")
    
    # Find optimal k
    print("\n[4/5] Finding optimal number of topics...")
    best_k = find_optimal_k(corpus, dictionary, texts_tok, k_range=range(2, 9))
    
    # Fit final LDA model
    print(f"\n[5/5] Fitting final LDA model with k={best_k}...")
    try:
        lda = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=best_k,
            random_state=42,
            passes=20,
            alpha="auto",
            eta="auto",
            iterations=200,
            chunksize=100,
        )
        
        # Print topic summary
        topic_labels = print_topic_summary(lda, best_k)
        
        # Compute per-document topic distributions
        print("\n  Computing document-topic distributions...")
        doc_topics = np.zeros((len(corpus), best_k))
        for doc_idx, bow in enumerate(corpus):
            try:
                topics = lda.get_document_topics(bow, minimum_probability=0.0)
                for topic_id, prob in topics:
                    doc_topics[doc_idx, topic_id] = prob
            except:
                # Fallback for any errors
                doc_topics[doc_idx] = np.ones(best_k) / best_k
        
        # Add topic columns to dataframe
        topic_cols = [f"topic_{i}" for i in range(best_k)]
        for i, col in enumerate(topic_cols):
            df[col] = doc_topics[:, i]
        
        # Save enriched DataFrame
        output_path = RESULTS_DIR / "fomc_with_topics.csv"
        df.to_csv(output_path, index=False)
        print(f"\n  Saved enriched data to {output_path}")
        
        # Plot topic evolution
        print("\n  Generating topic evolution plot...")
        plot_topic_evolution(df, topic_cols, topic_labels)
        
        # Final coherence
        cm_final = CoherenceModel(
            model=lda, 
            texts=texts_tok,
            dictionary=dictionary, 
            coherence="c_v"
        )
        final_coherence = cm_final.get_coherence()
        print(f"\n  Final model coherence (Cv): {final_coherence:.4f}")
        
        # Summary statistics
        print("\n" + "=" * 60)
        print("SUMMARY STATISTICS")
        print("=" * 60)
        print(f"  Total documents: {len(df)}")
        print(f"  Number of topics: {best_k}")
        print(f"  Coherence score: {final_coherence:.4f}")
        print(f"  Vocabulary size: {len(dictionary)}")
        
        # Print dominant topics per year
        print("\n  Dominant topics by year:")
        df['year'] = df['date'].dt.year
        yearly_dominant = df.groupby('year')[topic_cols].mean().idxmax(axis=1)
        for year, topic in yearly_dominant.items():
            print(f"    {year}: Topic {int(topic.split('_')[1]) + 1}")
        
    except Exception as e:
        print(f"\n  Error during LDA fitting: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("TOPIC MODELLING COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to:")
    print(f"  - Data: {RESULTS_DIR / 'fomc_with_topics.csv'}")
    print(f"  - Figures: {FIGURE_DIR}/")
    print(f"    - lda_coherence.png")
    print(f"    - topic_evolution.png")


if __name__ == "__main__":
    main()