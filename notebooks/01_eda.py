"""
01_eda.py
---------
Exploratory Data Analysis of the FOMC statement corpus.

Run this AFTER data collection:
    python data/scrape_fomc.py
    python data/fetch_returns.py

Outputs figures to results/figures/
"""

import sys
import os
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# Determine the project root directory
# This script is in notebooks/01_eda.py, so project root is parent directory
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"

# Create figure directory if it doesn't exist
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})


def load_and_preprocess(data_path=None):
    """
    Load and preprocess FOMC statement data.
    
    Parameters:
    -----------
    data_path : str or Path, optional
        Path to the CSV file. If None, uses default path.
    
    Returns:
    --------
    DataFrame with loaded data
    """
    if data_path is None:
        # Use the data directory in project root
        data_path = DATA_DIR / "fomc_with_returns.csv"
    else:
        data_path = Path(data_path)
        # If it's a relative path, try to resolve it relative to project root
        if not data_path.is_absolute():
            # Try both relative to current working directory and project root
            if not data_path.exists():
                data_path = PROJECT_ROOT / data_path
    
    if not data_path.exists():
        print(f"Error: {data_path} not found")
        print(f"Looking in: {data_path}")
        print(f"Data directory exists: {DATA_DIR.exists()}")
        if DATA_DIR.exists():
            print(f"Files in data directory: {list(DATA_DIR.glob('*.csv'))}")
        return None
    
    df = pd.read_csv(data_path, parse_dates=['date'])
    print(f"Loaded {len(df)} rows from {data_path}")
    
    # Check if label column exists
    if 'label' in df.columns:
        # Count valid labels
        valid_labels = df['label'].notna().sum()
        print(f"Rows with labels: {valid_labels}/{len(df)}")
        
        if valid_labels > 0:
            label_counts = df['label'].value_counts()
            print(f"Label distribution:")
            print(f"  Up (1):  {label_counts.get(1, 0)}")
            print(f"  Down (0): {label_counts.get(0, 0)}")
    
    # Create token count column (approximate)
    if 'text' in df.columns:
        # Approximate token count by splitting on whitespace
        df['n_tokens'] = df['text'].str.split().str.len()
        print(f"Text length range: {df['n_tokens'].min()} - {df['n_tokens'].max()} tokens")
    
    return df


def main():
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Figure directory: {FIGURE_DIR}")

    # Load data
    df = load_and_preprocess("data/fomc_with_returns.csv")
    
    if df is None:
        print("\nFailed to load data. Exiting.")
        print("\nPossible solutions:")
        print("1. Make sure you've run the data collection scripts:")
        print("   python data/scrape_fomc.py")
        print("   python data/fetch_returns.py")
        print("2. Check that the data file exists at:", DATA_DIR / "fomc_with_returns.csv")
        print("3. Run this script from the project root directory:")
        print("   cd /path/to/fed-sentiment")
        print("   python notebooks/01_eda.py")
        return
    
    # Basic information
    print(f"\nDate range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Total statements: {len(df)}")
    
    # Return statistics (if available)
    if 'ret_1d' in df.columns and df['ret_1d'].notna().any():
        print(f"\nReturn statistics (1-day log):")
        print(df["ret_1d"].describe().round(5))
    
    # Label balance (if available)
    if 'label' in df.columns and df['label'].notna().any():
        print(f"\nLabel balance:")
        label_counts = df["label"].value_counts()
        print(f"  Market up (1):   {label_counts.get(1, 0)}")
        print(f"  Market down (0): {label_counts.get(0, 0)}")
    
    # ── Figure 1: Token length distribution ─────────────────────────────────
    if 'n_tokens' in df.columns and df['n_tokens'].notna().any():
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df["n_tokens"].dropna(), bins=30, color="#534AB7", alpha=0.8, edgecolor="white")
        ax.axvline(512, color="#D85A30", linewidth=1.5, linestyle="--", label="BERT limit (512)")
        
        # Add median line
        median_tokens = df["n_tokens"].median()
        ax.axvline(median_tokens, color="#1D9E75", linewidth=1.5, linestyle=":", label=f"Median: {median_tokens:.0f}")
        
        ax.set_xlabel("Token count (approx.)", fontsize=10)
        ax.set_ylabel("Number of statements", fontsize=10)
        ax.set_title("Distribution of FOMC Statement Lengths", fontsize=12, fontweight="bold")
        ax.legend()
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "eda_token_lengths.png", bbox_inches="tight")
        plt.close()
        print("\nSaved: results/figures/eda_token_lengths.png")
    
    # ── Figure 2: Market returns after FOMC ──────────────────────────────────
    return_cols = [col for col in ["ret_1d", "ret_3d", "ret_5d"] 
                   if col in df.columns and df[col].notna().any()]
    
    if return_cols:
        fig, axes = plt.subplots(1, len(return_cols), figsize=(12, 4))
        if len(return_cols) == 1:
            axes = [axes]
        
        titles = {
            "ret_1d": "1-day returns",
            "ret_3d": "3-day returns", 
            "ret_5d": "5-day returns"
        }
        
        for ax, col in zip(axes, return_cols):
            returns = df[col].dropna()
            ax.hist(returns, bins=25, color="#1D9E75", alpha=0.8, edgecolor="white")
            ax.axvline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.5)
            ax.axvline(returns.mean(), color="#D85A30", linewidth=1.5, linestyle="--", 
                      label=f"Mean: {returns.mean():.4f}")
            ax.set_title(titles.get(col, col), fontsize=10)
            ax.set_xlabel("Log return", fontsize=9)
            ax.legend(fontsize=8)
        
        axes[0].set_ylabel("Count", fontsize=9)
        fig.suptitle("S&P 500 Returns After FOMC Meetings", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "eda_return_distributions.png", bbox_inches="tight")
        plt.close()
        print("Saved: results/figures/eda_return_distributions.png")
    
    # ── Figure 3: Statement frequency per year ───────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    df["year"] = df["date"].dt.year
    yearly = df.groupby("year").size()
    
    ax.bar(yearly.index, yearly.values, color="#534AB7", alpha=0.8, edgecolor="white")
    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel("Number of FOMC meetings", fontsize=10)
    ax.set_title("FOMC Meeting Frequency by Year", fontsize=12, fontweight="bold")
    
    # Add value labels on bars
    for i, (year, count) in enumerate(yearly.items()):
        if count > 5:  # Only label bars with more than 5 meetings to avoid clutter
            ax.text(year, count + 0.5, str(count), ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "eda_meeting_frequency.png", bbox_inches="tight")
    plt.close()
    print("Saved: results/figures/eda_meeting_frequency.png")
    
    # ── Figure 4: Returns over time ──────────────────────────────────────────
    if 'ret_1d' in df.columns and df['ret_1d'].notna().any():
        fig, ax = plt.subplots(figsize=(12, 5))
        
        # Color returns by positive/negative
        colors = ['#1D9E75' if x > 0 else '#D85A30' for x in df['ret_1d'].dropna()]
        
        ax.bar(df['date'][df['ret_1d'].notna()], 
               df['ret_1d'].dropna(), 
               color=colors, alpha=0.7, width=20)
        
        ax.axhline(0, color='black', linewidth=0.8, linestyle='-')
        ax.axhline(df['ret_1d'].mean(), color='blue', linewidth=1.5, linestyle='--', 
                  label=f'Mean: {df["ret_1d"].mean():.4f}')
        
        ax.set_xlabel("Date", fontsize=10)
        ax.set_ylabel("1-day log return", fontsize=10)
        ax.set_title("S&P 500 Returns After FOMC Meetings Over Time", fontsize=12, fontweight="bold")
        ax.legend()
        
        # Format x-axis to show years
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "eda_returns_over_time.png", bbox_inches="tight")
        plt.close()
        print("Saved: results/figures/eda_returns_over_time.png")
    
    # ── Figure 5: VIX distribution if available ──────────────────────────────
    if 'vix' in df.columns and df['vix'].notna().any():
        fig, ax = plt.subplots(figsize=(10, 5))
        
        ax.hist(df['vix'].dropna(), bins=30, color="#534AB7", alpha=0.8, edgecolor="white")
        ax.axvline(df['vix'].mean(), color="#D85A30", linewidth=1.5, linestyle="--", 
                  label=f"Mean: {df['vix'].mean():.1f}")
        ax.axvline(df['vix'].median(), color="#1D9E75", linewidth=1.5, linestyle=":", 
                  label=f"Median: {df['vix'].median():.1f}")
        
        ax.set_xlabel("VIX Index Value", fontsize=10)
        ax.set_ylabel("Number of FOMC statements", fontsize=10)
        ax.set_title("VIX Levels on FOMC Meeting Dates", fontsize=12, fontweight="bold")
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "eda_vix_distribution.png", bbox_inches="tight")
        plt.close()
        print("Saved: results/figures/eda_vix_distribution.png")
    
    # ── Summary statistics ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    if 'ret_1d' in df.columns and df['ret_1d'].notna().any():
        print("\nReturn Statistics:")
        for col in ['ret_1d', 'ret_3d', 'ret_5d']:
            if col in df.columns and df[col].notna().any():
                valid = df[col].dropna()
                print(f"\n{col.upper()}:")
                print(f"  Mean:     {valid.mean():.6f}")
                print(f"  Std:      {valid.std():.6f}")
                print(f"  Positive: {(valid > 0).sum()} ({(valid > 0).sum()/len(valid)*100:.1f}%)")
                print(f"  Negative: {(valid < 0).sum()} ({(valid < 0).sum()/len(valid)*100:.1f}%)")
    
    if 'vix' in df.columns and df['vix'].notna().any():
        print(f"\nVIX Statistics:")
        print(f"  Mean:   {df['vix'].mean():.2f}")
        print(f"  Median: {df['vix'].median():.2f}")
        print(f"  Min:    {df['vix'].min():.2f}")
        print(f"  Max:    {df['vix'].max():.2f}")
    
    print("\nEDA complete.")


if __name__ == "__main__":
    main()