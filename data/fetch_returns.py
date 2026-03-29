"""
fetch_returns.py
----------------
Fetches S&P 500 (^GSPC) and VIX (^VIX) data from multiple sources,
then aligns each FOMC statement date to next-trading-day returns.

Data sources in order of preference:
  1. Local cache (market_data_cache.csv)
  2. Yahoo Finance (^GSPC, ^VIX)
  3. FRED (Federal Reserve Economic Data: SP500 for S&P 500, VIXCLS for VIX)
  4. Manual CSV files (sp500_manual.csv, vix_manual.csv)
  5. Fallback to approximate US business-day calendar

Updated to use proper FRED series that cover 2000-2024:
  - S&P 500: Use 'SP500' (available from 2016) OR combine with 'SP500' from Yahoo
  - For full coverage, we'll use Yahoo as primary and FRED as secondary
"""

import sys
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

INPUT_PATH = Path(__file__).parent / "fomc_statements.csv"
OUTPUT_PATH = Path(__file__).parent / "fomc_with_returns.csv"
CACHE_PATH = Path(__file__).parent / "market_data_cache.csv"

# FOMC statement date range from your data
START_DATE = "1999-01-01"  # Slightly earlier than first statement (2000-02-02)
END_DATE = "2025-06-01"    # Slightly later than last statement (2024-12-18)

# FRED series IDs
FRED_SPX_SERIES = "SP500"      # S&P 500 (starts 2016-03-28)
FRED_VIX_SERIES = "VIXCLS"     # VIX (starts 1990-01-02)


# ─────────────────────────────────────────────────────────────────────────────
# Data Source Functions
# ─────────────────────────────────────────────────────────────────────────────

def fetch_from_yahoo(start: str, end: str) -> tuple[pd.Series | None, pd.Series | None]:
    """
    Attempt to fetch data from Yahoo Finance.
    Returns (spx_series, vix_series) or (None, None) on failure.
    """
    try:
        print("  Attempting Yahoo Finance...")
        
        # Download with retries and error handling
        spx = yf.download("^GSPC", start=start, end=end, 
                         progress=False, auto_adjust=True, 
                         threads=False, prepost=False)
        
        vix = yf.download("^VIX", start=start, end=end, 
                         progress=False, auto_adjust=True,
                         threads=False, prepost=False)
        
        if spx.empty or vix.empty:
            print("  Yahoo Finance returned empty data")
            return None, None
        
        # Extract close prices (handles both single and multi-level columns)
        spx_close = spx['Close'] if 'Close' in spx.columns else spx.iloc[:, 0]
        vix_close = vix['Close'] if 'Close' in vix.columns else vix.iloc[:, 0]
        
        # Ensure we have datetime index
        spx_close.index = pd.to_datetime(spx_close.index)
        vix_close.index = pd.to_datetime(vix_close.index)
        
        # Drop any NaN values
        spx_close = spx_close.dropna()
        vix_close = vix_close.dropna()
        
        if len(spx_close) > 0 and len(vix_close) > 0:
            print(f"  Yahoo Finance success: {len(spx_close):,} days for SPX, {len(vix_close):,} for VIX")
            return spx_close, vix_close
        
    except Exception as e:
        print(f"  Yahoo Finance failed: {str(e)[:100]}")
    
    return None, None


def fetch_from_fred(start: str, end: str) -> tuple[pd.Series | None, pd.Series | None]:
    """
    Attempt to fetch data from FRED (Federal Reserve Economic Data).
    Returns (spx_series, vix_series) or (None, None) on failure.
    """
    try:
        print("  Attempting FRED...")
        
        # Try to import pandas_datareader
        try:
            import pandas_datareader as pdr
        except ImportError:
            print("  pandas_datareader not installed. Install with: pip install pandas-datareader")
            return None, None
        
        # Fetch VIX from FRED (available from 1990)
        vix = pdr.DataReader(FRED_VIX_SERIES, 'fred', start=start, end=end)
        vix_close = vix[FRED_VIX_SERIES] if isinstance(vix, pd.DataFrame) else vix
        vix_close.index = pd.to_datetime(vix_close.index)
        vix_close = vix_close.dropna()
        
        # Fetch S&P 500 from FRED (available from 2016 only)
        spx = pdr.DataReader(FRED_SPX_SERIES, 'fred', start=start, end=end)
        spx_close = spx[FRED_SPX_SERIES] if isinstance(spx, pd.DataFrame) else spx
        spx_close.index = pd.to_datetime(spx_close.index)
        spx_close = spx_close.dropna()
        
        # Check if we have data
        if spx_close.empty:
            print(f"  FRED S&P 500 data only available from {spx_close.index.min() if not spx_close.empty else 'N/A'} onwards")
            print("  Will try to combine with Yahoo Finance data")
            return None, None
        
        if vix_close.empty:
            print("  FRED VIX data unavailable")
            return None, None
        
        print(f"  FRED success: {len(spx_close):,} days for SPX, {len(vix_close):,} for VIX")
        return spx_close, vix_close
        
    except Exception as e:
        print(f"  FRED failed: {str(e)[:100]}")
    
    return None, None


def fetch_combined_data(start: str, end: str) -> tuple[pd.Series | None, pd.Series | None]:
    """
    Combine multiple sources to get full date range.
    Priority: Yahoo > FRED > Manual
    """
    print("  Attempting to combine multiple data sources...")
    
    # Try Yahoo first (best coverage)
    spx_yahoo, vix_yahoo = fetch_from_yahoo(start, end)
    if spx_yahoo is not None and vix_yahoo is not None:
        print("  Using Yahoo Finance data for full coverage")
        return spx_yahoo, vix_yahoo
    
    # If Yahoo fails, try FRED for VIX and combine with other SPX source
    print("  Yahoo failed, trying FRED for VIX...")
    try:
        import pandas_datareader as pdr
        
        # Get VIX from FRED (good coverage)
        vix = pdr.DataReader(FRED_VIX_SERIES, 'fred', start=start, end=end)
        vix_close = vix[FRED_VIX_SERIES] if isinstance(vix, pd.DataFrame) else vix
        vix_close.index = pd.to_datetime(vix_close.index)
        vix_close = vix_close.dropna()
        
        # For SPX, we need an alternative source
        # Try to get from Yahoo again (maybe partial data)
        spx_partial = None
        try:
            spx = yf.download("^GSPC", start=start, end=end, 
                             progress=False, auto_adjust=True)
            if not spx.empty:
                spx_partial = spx['Close'] if 'Close' in spx.columns else spx.iloc[:, 0]
                spx_partial.index = pd.to_datetime(spx_partial.index)
                spx_partial = spx_partial.dropna()
                print(f"  Got partial SPX from Yahoo: {len(spx_partial):,} days")
        except:
            pass
        
        if spx_partial is not None and len(spx_partial) > 0 and len(vix_close) > 0:
            print(f"  Combined data: SPX from Yahoo, VIX from FRED")
            return spx_partial, vix_close
        
    except Exception as e:
        print(f"  Combined approach failed: {str(e)[:100]}")
    
    return None, None


def load_from_cache() -> tuple[pd.Series | None, pd.Series | None]:
    """
    Load cached market data if available.
    Returns (spx_series, vix_series) or (None, None).
    """
    if CACHE_PATH.exists():
        try:
            cache = pd.read_csv(CACHE_PATH, parse_dates=['date'], index_col='date')
            
            if 'spx' in cache.columns and 'vix' in cache.columns:
                spx = cache['spx'].dropna()
                vix = cache['vix'].dropna()
                
                if len(spx) > 0 and len(vix) > 0:
                    print(f"  Loaded cached data: {len(spx):,} days")
                    return spx, vix
                else:
                    print("  Cache file is empty")
        except Exception as e:
            print(f"  Failed to load cache: {str(e)[:100]}")
    
    return None, None


def save_to_cache(spx: pd.Series, vix: pd.Series):
    """Save market data to cache for future use."""
    try:
        # Align the series
        aligned = pd.DataFrame({
            'spx': spx,
            'vix': vix
        }).dropna()
        
        aligned.to_csv(CACHE_PATH, index=True, index_label='date')
        print(f"  Saved {len(aligned):,} days to cache: {CACHE_PATH.name}")
    except Exception as e:
        print(f"  Failed to save cache: {str(e)[:100]}")


def load_manual_data() -> tuple[pd.Series | None, pd.Series | None]:
    """
    Load manually downloaded CSV files.
    Expected format: CSV with 'Date' and 'Close' columns.
    """
    spx_manual_path = Path(__file__).parent / "sp500_manual.csv"
    vix_manual_path = Path(__file__).parent / "vix_manual.csv"
    
    if spx_manual_path.exists() and vix_manual_path.exists():
        try:
            print("  Loading manual data from CSV files...")
            
            spx = pd.read_csv(spx_manual_path, parse_dates=['Date'], index_col='Date')
            vix = pd.read_csv(vix_manual_path, parse_dates=['Date'], index_col='Date')
            
            # Extract close prices
            spx_close = spx['Close'] if 'Close' in spx.columns else spx.iloc[:, 0]
            vix_close = vix['Close'] if 'Close' in vix.columns else vix.iloc[:, 0]
            
            spx_close = spx_close.dropna()
            vix_close = vix_close.dropna()
            
            if len(spx_close) > 0 and len(vix_close) > 0:
                print(f"  Manual data loaded: {len(spx_close):,} SPX, {len(vix_close):,} VIX")
                return spx_close, vix_close
            
        except Exception as e:
            print(f"  Failed to load manual data: {str(e)[:100]}")
    
    return None, None


def build_market_dataframe(spx: pd.Series, vix: pd.Series) -> pd.DataFrame:
    """
    Build complete market DataFrame from price series.
    Calculates daily returns and aligns data.
    """
    # Create DataFrame and align
    market = pd.DataFrame(index=pd.date_range(start=min(spx.index.min(), vix.index.min()), 
                                              end=max(spx.index.max(), vix.index.max()), 
                                              freq='D'))
    market['spx'] = spx
    market['vix'] = vix
    
    # Forward fill for days with missing data (weekends/holidays)
    market = market.fillna(method='ffill')
    
    # Keep only weekdays and drop any remaining NaN
    market = market[market.index.dayofweek < 5].dropna()
    
    # Calculate daily returns
    market['ret_1d'] = np.log(market['spx'] / market['spx'].shift(1))
    
    # Sort index
    market = market.sort_index()
    
    print(f"  Built market data: {len(market):,} trading days "
          f"({market.index[0].date()} → {market.index[-1].date()})")
    
    return market


def fetch_market_data() -> tuple[pd.DataFrame, bool]:
    """
    Fetch market data from multiple sources in order:
    1. Local cache
    2. Yahoo Finance
    3. Combined approach (Yahoo + FRED)
    4. FRED only
    5. Manual CSV files
    6. Fallback to approximate calendar
    
    Returns:
        tuple: (market_dataframe, has_real_data)
    """
    print("\nFetching market data...")
    
    # Try each data source in order
    sources = [
        ("cache", load_from_cache),
        ("Yahoo Finance", lambda: fetch_from_yahoo(START_DATE, END_DATE)),
        ("combined sources", lambda: fetch_combined_data(START_DATE, END_DATE)),
        ("FRED", lambda: fetch_from_fred(START_DATE, END_DATE)),
        ("manual files", load_manual_data),
    ]
    
    for source_name, source_func in sources:
        try:
            spx, vix = source_func()
            if spx is not None and vix is not None and len(spx) > 0 and len(vix) > 0:
                print(f"  ✓ Successfully fetched from {source_name}")
                
                # Cache the data if it's not already from cache
                if source_name != "cache":
                    save_to_cache(spx, vix)
                
                # Build and return market DataFrame
                return build_market_dataframe(spx, vix), True
        except Exception as e:
            print(f"  {source_name} error: {str(e)[:100]}")
            continue
    
    # Fallback to approximate calendar
    print("\n  WARNING: Could not fetch market data from any source.")
    print("  Falling back to approximate US business-day calendar.")
    print("  Returns will be NaN. To fix:")
    print("    1. Check internet connection/proxy settings")
    print("    2. Install pandas-datareader: pip install pandas-datareader")
    print("    3. Manually download data to sp500_manual.csv and vix_manual.csv")
    print("       Format: Date,Close (one row per trading day)")
    print("    4. Ensure you have a working connection to Yahoo Finance or FRED")
    
    # Create approximate trading days
    bdays = pd.bdate_range(start=START_DATE, end=END_DATE)
    market = pd.DataFrame(
        {"spx": np.nan, "vix": np.nan, "ret_1d": np.nan},
        index=bdays,
    )
    
    return market, False


# ─────────────────────────────────────────────────────────────────────────────
# Core Alignment Logic
# ─────────────────────────────────────────────────────────────────────────────

def next_trading_day(date: pd.Timestamp, trading_days: pd.DatetimeIndex) -> pd.Timestamp | None:
    """
    Return the first trading day strictly after `date`.
    Returns None if no trading day found.
    """
    if len(trading_days) == 0:
        return None
    
    future = trading_days[trading_days > date]
    return future[0] if len(future) > 0 else None


def cumulative_log_return(market: pd.DataFrame, start_day: pd.Timestamp, n_days: int) -> float | None:
    """
    Sum `n_days` consecutive daily log returns starting from `start_day`.
    Returns None if data is missing or insufficient.
    """
    if start_day not in market.index:
        return None
    
    pos = market.index.get_loc(start_day)
    end_pos = pos + n_days
    
    if end_pos > len(market):
        return None
    
    rets = market["ret_1d"].iloc[pos:end_pos]
    
    if rets.isna().any():
        return None
    
    return float(rets.sum())


def align_returns() -> pd.DataFrame:
    """
    Main function to align FOMC statements with market returns.
    Creates the final dataset with return labels.
    """
    # ── Load FOMC statements ──────────────────────────────────────────────────
    if not INPUT_PATH.exists():
        sys.exit(
            f"ERROR: {INPUT_PATH} not found.\n"
            "Run `python data/scrape_fomc.py` first."
        )
    
    fomc = pd.read_csv(INPUT_PATH, parse_dates=["date"])
    fomc["date"] = pd.to_datetime(fomc["date"])
    fomc = fomc.sort_values("date").reset_index(drop=True)
    
    print(f"Loaded {len(fomc)} FOMC statements "
          f"({fomc['date'].min().date()} → {fomc['date'].max().date()})")
    
    # ── Fetch market data ─────────────────────────────────────────────────────
    market, has_real_data = fetch_market_data()
    trading_days = market.index
    
    # ── Align each statement ──────────────────────────────────────────────────
    rows = []
    no_trading_day_count = 0
    missing_returns_count = 0
    
    for idx, stmt in fomc.iterrows():
        stmt_date = pd.Timestamp(stmt["date"])
        
        # Find next trading day after statement date
        t1 = next_trading_day(stmt_date, trading_days)
        
        if t1 is None:
            no_trading_day_count += 1
            # Still add row with placeholder values
            rows.append({
                "date": stmt_date,
                "text": stmt["text"],
                "url": stmt.get("url", ""),
                "t1_date": None,
                "ret_1d": None,
                "ret_3d": None,
                "ret_5d": None,
                "vix": None,
                "label": None,
            })
            continue
        
        # Calculate returns if we have real data
        if has_real_data:
            ret_1d = cumulative_log_return(market, t1, 1)
            ret_3d = cumulative_log_return(market, t1, 3)
            ret_5d = cumulative_log_return(market, t1, 5)
            
            # Create label (1 = market up, 0 = down)
            label = int(ret_1d > 0) if ret_1d is not None else None
            
            # Get VIX value (use statement date if it's a trading day, else t1)
            vix_day = stmt_date if stmt_date in market.index else t1
            vix_val = (float(market.loc[vix_day, "vix"])
                      if vix_day in market.index and not pd.isna(market.loc[vix_day, "vix"])
                      else None)
            
            # Count missing returns
            if ret_1d is None:
                missing_returns_count += 1
        else:
            ret_1d = ret_3d = ret_5d = label = vix_val = None
            missing_returns_count += 1
        
        rows.append({
            "date": stmt_date,
            "text": stmt["text"],
            "url": stmt.get("url", ""),
            "t1_date": t1,
            "ret_1d": ret_1d,
            "ret_3d": ret_3d,
            "ret_5d": ret_5d,
            "vix": vix_val,
            "label": label,
        })
    
    # ── Build output DataFrame ────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    
    # If we have real data, drop rows where we have no label (no return data)
    if has_real_data:
        original_len = len(df)
        df = df[df["label"].notna()].copy()
        if not df.empty:
            df["label"] = df["label"].astype(int)
        
        dropped = original_len - len(df)
        if dropped > 0:
            print(f"\n  Dropped {dropped} statements with no return data (outside market data range)")
    
    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)
    
    # ── Save to CSV ───────────────────────────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    
    # ── Print summary statistics ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"✓ Saved {len(df)} rows → {OUTPUT_PATH}")
    print(f"{'='*60}")
    
    if no_trading_day_count > 0:
        print(f"  ⚠ No subsequent trading day found for {no_trading_day_count} statements")
    
    if has_real_data and not df.empty:
        print(f"\n📊 Data Quality Summary:")
        print(f"  Statements with returns: {df['ret_1d'].notna().sum()}/{len(df)}")
        
        if missing_returns_count > 0:
            print(f"  Statements with missing returns: {missing_returns_count}")
        
        print(f"\n  Label balance:")
        up_count = df['label'].sum() if 'label' in df.columns else 0
        down_count = (df['label'] == 0).sum() if 'label' in df.columns else 0
        print(f"    Up (label=1):  {up_count}")
        print(f"    Down (label=0): {down_count}")
        
        print(f"\n📈 Return Statistics:")
        for period, col in [('1-day', 'ret_1d'), ('3-day', 'ret_3d'), ('5-day', 'ret_5d')]:
            valid_returns = df[col].dropna()
            if len(valid_returns) > 0:
                print(f"\n  {period} returns ({len(valid_returns)} observations):")
                print(f"    Mean:    {valid_returns.mean():.5f}")
                print(f"    Std:     {valid_returns.std():.5f}")
                print(f"    Min:     {valid_returns.min():.5f}")
                print(f"    Max:     {valid_returns.max():.5f}")
        
        if df['vix'].notna().any():
            print(f"\n📊 VIX Statistics:")
            print(f"  Mean VIX:   {df['vix'].mean():.2f}")
            print(f"  Min VIX:    {df['vix'].min():.2f}")
            print(f"  Max VIX:    {df['vix'].max():.2f}")
        
        # Date range of statements with returns
        statements_with_returns = df[df['ret_1d'].notna()]
        if not statements_with_returns.empty:
            print(f"\n📅 Statements with returns: "
                  f"{statements_with_returns['date'].min().date()} → "
                  f"{statements_with_returns['date'].max().date()}")
            print(f"  Total period coverage: "
                  f"{len(statements_with_returns)} out of {len(fomc)} statements "
                  f"({len(statements_with_returns)/len(fomc)*100:.1f}%)")
    
    else:
        print("\n⚠ NO REAL MARKET DATA AVAILABLE")
        print("  Returns are placeholder NaN values.")
        print("\n  To get real data, try one of these:")
        print("    1. Check internet connection and proxy settings")
        print("    2. Install pandas-datareader: pip install pandas-datareader")
        print("    3. Manually download market data to:")
        print("       - sp500_manual.csv (Date,Close columns)")
        print("       - vix_manual.csv (Date,Close columns)")
        print("    4. Ensure your network allows access to Yahoo Finance or FRED")
    
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main Execution
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        df = align_returns()
        
        # Print first few rows for verification
        print(f"\n{'='*60}")
        print("Sample of output data (first 10 rows):")
        print(f"{'='*60}")
        display_cols = ['date', 't1_date', 'ret_1d', 'ret_3d', 'label', 'vix']
        print(df[display_cols].head(10).to_string())
        
        print(f"\n{'='*60}")
        print("Last 10 rows of output data:")
        print(f"{'='*60}")
        print(df[display_cols].tail(10).to_string())
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)