# Fed Sentiment: FOMC Communication & Equity Market Returns


## Research Question

Does the sentiment of Federal Reserve FOMC post-meeting statements predict short-term U.S. equity market returns, and has the tone of Fed communication shifted across monetary policy regimes (2016–2024)?

## Reproducing Results

### 1. Environment setup

```bash
git clone https://github.com/SajittaPhat/fed-sentiment.git
cd fed-sentiment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download FOMC statements (~5 minutes)

```bash
python data/scrape_fomc.py
```

Downloads all post-meeting FOMC statements from `federalreserve.gov` (2000–2024) and saves to `data/fomc_statements.csv`.

### 3. Fetch market returns (~1 minute)

```bash
python data/fetch_returns.py
```

Aligns each statement date to next-day S&P 500 log returns using `yfinance`. Saves to `data/fomc_with_returns.csv`.

### 4. Run EDA

```bash
python notebooks/01_eda.py
```

Generates token length distribution, return distribution, and meeting frequency charts in `results/figures/`.

### 5. Run all sentiment models (~20 min on CPU, ~5 min on GPU)

```bash
python notebooks/02_run_all_models.py
```

Runs VADER and FinBERT (zero-shot). Saves scores, prints model comparison table, generates all plots.

### 6. Run topic modelling (~10 minutes)

```bash
python notebooks/03_topic_modelling.py
```

Fits LDA, selects optimal k via coherence, generates topic evolution chart.

## Hardware Requirements

- CPU: All scripts run on CPU (~30 min total)
- GPU: Automatically used if available via PyTorch CUDA. Reduces fine-tuning to ~5 min.
- RAM: 8 GB minimum (16 GB recommended for FinBERT fine-tuning)

## Data Sources

- **FOMC statements**: [Federal Reserve](https://www.federalreserve.gov/monetarypolicy/fomc_historical.htm) (public domain)
- **S&P 500 / VIX**: Yahoo Finance via `yfinance`
