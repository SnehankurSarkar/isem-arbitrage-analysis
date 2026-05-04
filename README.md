# Energy Market Arbitrage Analytics

**Predicting short-term I-SEM electricity price spreads and evaluating confidence-filtered trading strategy performance.**

This repository is a professional, public-safe reproduction structure for an MSc Data Analytics Analytathon project using Energia-provided I-SEM market data. The original private dataset is not included. The repository preserves the submitted notebook as the source of truth, modularises the analysis code for maintainability, and provides pre-generated result summaries so reviewers can understand the work without accessing proprietary data.

---

## Executive summary

The project investigates whether demand, wind, solar generation, interconnector flows and system variables can explain and predict short-term price spreads between sequential I-SEM markets.

| Headline result | Finding |
|---|---|
| Best predictive transition | **DAM → IDA1** |
| DAM → IDA1 test accuracy | **70.18%** |
| DAM → IDA1 ROC-AUC | **0.7718** |
| Most profitable transition | **DAM → BM** |
| DAM → BM gross profitability | **€33.75/MWh** |
| DAM → BM total gross P&L | **€480,887.04** |
| DAM → BM executed trades | **4,258** |
| DAM → BM hit rate | **70.36%** |

The main strategic interpretation is that **DAM → IDA1 is the more predictable/stable modelling target**, while **DAM → BM offers the highest gross profitability but with higher risk and stronger need for drawdown monitoring**.

---

## Repository design

```text
energy-market-arbitrage-analytics/
├── config/                    # Reproduction settings and model constants
├── data/
│   ├── sample/                # Public synthetic sample only
│   └── README.md              # Data access and confidentiality notes
├── docs/                      # Methodology, results, data dictionary, model card
├── notebooks/
│   └── original/              # Original submitted notebook retained as source of truth
├── outputs/                   # Pre-generated result tables from the submitted analysis
├── reports/
│   ├── figures/               # Pre-generated result visualisations
│   └── executive_case_study.md
├── scripts/                   # CLI scripts for sample data and private reproduction runs
├── src/energy_arbitrage/      # Modular reproduction package
├── tests/                     # Fidelity and hygiene checks
├── pyproject.toml
├── requirements.txt
└── Makefile
```

---

## Why this repo is structured this way

The original analysis was completed in a notebook. For a professional GitHub portfolio, the same logic is separated into maintainable Python modules:

| Module | Purpose |
|---|---|
| `data.py` | Load parquet/CSV data, parse timestamps, clean duplicates, handle market-structural missingness |
| `features.py` | Create spreads, net demand, cyclical time features, forecast revisions, lags and transition-specific feature sets |
| `modeling.py` | Build LightGBM classifier/regressor pipelines, temporal split, walk-forward validation |
| `backtesting.py` | Confidence filter, magnitude-proportional sizing, P&L evaluation and leaderboard |
| `visualization.py` | Report-ready figures |
| `pipeline.py` | End-to-end orchestration |

`original_logic.py` is kept only as a compatibility re-export layer. The code is no longer a 700+ line monolith.

---

## Methodology

### 1. Data cleaning

The original dataset contained **72,945 rows × 105 columns**. After duplicate removal, timestamp consolidation and essential-variable filtering, the analytical dataset contained **70,006 observations**. IDA2 and IDA3 missingness was handled as market-structural rather than random, because those auctions only operate during specific delivery windows.

### 2. Target construction

Seven spreads were evaluated:

```text
DAM → IDA1
IDA1 → IDA2
IDA2 → IDA3
IDA1 → BM
IDA2 → BM
IDA3 → BM
DAM → BM
```

Each transition has:

- a classification target: `sign(spread)`
- a regression target: spread magnitude in €/MWh

### 3. Feature engineering

Features include:

- cyclical hour/day/month encodings
- net demand = demand − wind − solar
- wind forecast consensus and uncertainty
- demand and wind forecast revisions
- renewable penetration features
- 1-day, 2-day, 7-day and 14-day lag/rolling features
- transition-specific market-window logic to avoid future information leakage

### 4. Modelling

LightGBM is the primary model family, matching the submitted analysis.

```python
LGBMClassifier(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=6,
    num_leaves=63,
    min_child_samples=30,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
)
```

The regressor uses the same LightGBM parameter family.

### 5. Validation

- Strict chronological split: first 80% train, final 20% test
- No random shuffling
- 5-fold walk-forward cross-validation
- 14-day lag burn-in before modelling

### 6. Trading strategy

The classifier predicts spread direction. The model trades only when confidence exceeds the threshold:

```text
confidence = abs(p_up - 0.5)
trade if confidence > 0.20
```

Position size is scaled to the regression model's predicted spread magnitude. Results are gross of transaction costs.

---

## Pre-generated results

The repo includes report-level outputs so GitHub visitors can inspect the findings immediately:

| File | Description |
|---|---|
| `outputs/original_model_performance.csv` | Accuracy, ROC-AUC and CV results by transition |
| `outputs/original_trading_performance.csv` | Strategy results at confidence threshold 0.20 |
| `outputs/original_market_price_summary.csv` | Cleaned market price summary statistics |
| `reports/figures/original_model_accuracy.png` | Bar chart of test accuracy by transition |
| `reports/figures/original_total_pnl.png` | Bar chart of total gross P&L |
| `reports/figures/original_accuracy_vs_profitability.png` | Accuracy/profitability comparison |

---

## Reproducing locally with the private dataset

The raw Energia dataset is proprietary and intentionally excluded. To reproduce the full results locally, place the file here:

```text
data/private/MarketData_2022-2026.parquet
```

Then run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python scripts/01_run_reproduction_pipeline.py \
  --data data/private/MarketData_2022-2026.parquet \
  --outputs outputs \
  --figures reports/figures
```

Or use:

```bash
make test
```

---

## Public sample data

A synthetic sample is included under `data/sample/` for repository structure and unit tests. It is **not** the Energia dataset and should not be used to validate the reported business results.

---

## Limitations

- Reported P&L is gross of transaction costs and slippage.
- Market impact and liquidity constraints are not modelled.
- BM-related transitions are more volatile and harder to forecast.
- Live deployment would require rolling retraining, costs, risk limits and real-time monitoring.
- Public reproduction of the exact numbers requires the private dataset, which is not committed.

---

## Portfolio value

This project demonstrates:

- energy-market domain understanding
- time-series leakage control
- feature engineering from system fundamentals
- LightGBM classification/regression modelling
- walk-forward validation
- strategy design and backtesting
- P&L/risk interpretation
- professional code packaging and documentation
