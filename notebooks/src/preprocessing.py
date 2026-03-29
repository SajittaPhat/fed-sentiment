
"""
preprocessing.py
----------------
Data preprocessing functions for FOMC statement sentiment analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
from typing import Optional


def load_and_preprocess(data_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load and preprocess FOMC statement data.
    """
    # Determine project root
    if data_path is None:
        possible_paths = [
            Path("data/fomc_with_returns.csv"),
            Path("../data/fomc_with_returns.csv"),
            Path("../../data/fomc_with_returns.csv"),
            Path(__file__).parent.parent / "data" / "fomc_with_returns.csv",
        ]
        
        data_path = None
        for path in possible_paths:
            if path.exists():
                data_path = path
                break
        
        if data_path is None:
            raise FileNotFoundError("Could not find fomc_with_returns.csv in data directory")
    else:
        data_path = Path(data_path)
    
    # Load data
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, parse_dates=['date'])
    
    # Clean text
    if 'text' in df.columns:
        df['text_clean'] = df['text'].apply(clean_text)
    else:
        raise ValueError("Column 'text' not found in data")
    
    # Create token count
    df['n_tokens'] = df['text'].str.split().str.len()
    
    print(f"Loaded {len(df)} rows")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    return df


def clean_text(text: str) -> str:
    """Basic text cleaning function."""
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove special characters (keep letters, numbers, spaces)
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text
