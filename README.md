# ⚡ Energy Market Arbitrage Analytics

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boosting-brightgreen)](https://lightgbm.readthedocs.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20Pipeline-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Predicting short-term I-SEM electricity price spreads and evaluating confidence-filtered trading strategy performance.**

This repository is a professional, public-safe reproduction structure for an MSc Data Analytics Analytathon project using Energia-provided I-SEM market data. The original private dataset is not included. The repository preserves the submitted notebook as the source of truth, modularises the analysis code for maintainability, and provides pre-generated result summaries so reviewers can understand the work without accessing proprietary data.

---

## 📋 Table of Contents

- [Executive Summary](#executive-summary)
- [Market Structure](#market-structure)
- [Repository Design](#repository-design)
- [Methodology](#methodology)
- [Model Performance](#model-performance)
- [Trading Strategy Results](#trading-strategy-results)
- [Pre-generated Results](#pre-generated-results)
- [Setup & Installation](#setup--installation)
- [Reproducing Locally](#reproducing-locally-with-the-private-dataset)
- [Limitations](#limitations)
- [Portfolio Value](#portfolio-value)
- [References](#references)

---

## Executive Summary

The project investigates whether demand, wind, solar generation, interconnector flows and system variables can explain and predict short-term price spreads between sequential I-SEM markets.

| Headline Result | Finding |
|---|---|
| Best predictive transition | **DAM → IDA1** |
| DAM → IDA1 test accuracy | **70.18%** |
| DAM → IDA1 ROC-AUC | **0.7718** |
| Most profitable transition | **DAM → BM** |
| DAM → BM gross profitability | **€33.75/MWh** |
| DAM → BM total gross P&L | **€480,887.04** |
| DAM → BM executed trades | **4,258** |
| DAM → BM hit rate | **70.36%** |

The main strategic interpretation is that **DAM → IDA1 is the more predictable and stable modelling target**, while **DAM → BM offers the highest gross profitability but with higher risk and a stronger need for drawdown monitoring**.

### Strategic Recommendation

| Objective | Recommended Transition | Rationale |
|---|---|---|
| Maximise **predictability** | DAM → IDA1 | 70.18% accuracy, 80.93% hit rate, highest trade volume |
| Maximise **profitability** | DAM → BM | €33.75/MWh gross, subject to drawdown monitoring |
| **Balanced** approach | DAM → IDA1 (primary) + DAM → BM (supplementary, reduced volume) | Best risk-adjusted outcome |

---

## Market Structure

The I-SEM is the wholesale electricity market for Ireland and Northern Ireland, operated by SEMO. It comprises a cascade of sequential auction markets — each gate closure is an opportunity to refine positions as real-time delivery approaches and new information arrives:

```
DAM  ──►  IDA1  ──►  IDA2  ──►  IDA3  ──►  BM
```

Arbitrage means **buying in one market and offsetting in a later market**, profiting from the price spread between them. Profitability depends entirely on predicting the **direction and magnitude** of those spreads before they are realised.

---

## Repository Design

```text
energy-market-arbitrage-analytics/
├── config/                      # Reproduction settings and model constants
├── data/
│   ├── sample/                  # Public synthetic sample only
│   └── README.md                # Data access and confidentiality notes
├── docs/                        # Methodology, results, data dictionary, model card
├── notebooks/
│   └── original/                # Original submitted notebook — source of truth
├── outputs/                     # Pre-generated result tables from the submitted analysis
├── reports/
│   ├── figures/                 # Pre-generated result visualisations
│   └── executive_case_study.md
├── scripts/                     # CLI scripts for sample data and private reproduction
├── src/energy_arbitrage/        # Modular reproduction package
├── tests/                       # Fidelity and hygiene checks
├── pyproject.toml
├── requirements.txt
└── Makefile
```

The original analysis was completed in a notebook. For a professional portfolio, the same logic is separated into maintainable Python modules:

| Module | Purpose |
|---|---|
| `data.py` | Load parquet/CSV data, parse timestamps, clean duplicates, handle market-structural missingness |
| `features.py` | Create spreads, net demand, cyclical time features, forecast revisions, lags and transition-specific feature sets |
| `modeling.py` | Build LightGBM classifier/regressor pipelines, temporal split, walk-forward validation |
| `backtesting.py` | Confidence filter, magnitude-proportional sizing, P&L evaluation and leaderboard |
| `visualization.py` | Report-ready figures |
| `pipeline.py` | End-to-end orchestration |

`original_logic.py` is retained only as a compatibility re-export layer — the code is no longer a monolith.

---

## Methodology

### 1. Data Cleaning

The original dataset contained **72,945 rows × 105 columns**. After removing 2,721 fully duplicate rows, resolving 144 duplicate timestamps via NaN-safe mean, and dropping rows missing essential market variables, the analytical dataset contained **70,006 observations**. IDA2 and IDA3 missingness was treated as market-structural (not random), because those auctions only operate during specific delivery windows.

### 2. Target Construction

Seven spreads were evaluated, one per market transition:

```
DAM → IDA1    IDA1 → IDA2    IDA2 → IDA3
IDA1 → BM     IDA2 → BM      IDA3 → BM     DAM → BM
```

Each transition has a classification target (`sign(spread)`) and a regression target (spread magnitude in €/MWh).

### 3. Feature Engineering

All features are constructed strictly from information available at each market's gate closure — no future data leaks into any model input.

| Feature Category | Description |
|---|---|
| Cyclical calendar | Sine/cosine encoding of hour-of-day, day-of-week, month-of-year |
| Net demand | System demand minus wind and solar forecasts per auction horizon |
| Wind consensus & uncertainty | Mean and std dev across Meteo, EmSys, EirGrid forecast providers |
| Forecast revision | Change in wind/demand estimates between successive auction horizons |
| Lag variables | 1-day, 2-day, 7-day, 14-day lags of spreads and prices |
| Renewable penetration | Wind + solar as fraction of total demand per horizon |

### 4. Modelling

LightGBM is the primary model family, matching the submitted analysis exactly:

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

### 5. Validation

- Strict chronological split: first 80% train, final 20% test — no shuffling
- 5-fold walk-forward cross-validation (`TimeSeriesSplit`)
- 14-day lag burn-in before modelling begins
- Transition-specific feature windows to prevent future-market leakage

### 6. Trading Strategy

The classifier predicts spread direction. Trades are only placed when the model exceeds a confidence threshold:

```
confidence = abs(p_up - 0.5)
trade only if confidence > 0.20
```

Position size is scaled proportionally to the regression model's predicted spread magnitude. Results are gross of transaction costs.

---

## Model Performance

| Transition | Test Accuracy | ROC-AUC | CV Accuracy (±Std) |
|---|---|---|---|
| **DAM → IDA1** | **70.18%** | **0.7718** | **69.76% (±2.00%)** |
| DAM → BM | 58.54% | 0.6161 | 56.99% (±2.20%) |
| IDA1 → IDA2 | 57.18% | 0.6178 | 56.72% (±1.98%) |
| IDA1 → BM | 56.24% | 0.5779 | 55.17% (±1.92%) |
| IDA2 → BM | 56.12% | 0.5653 | 55.08% (±1.89%) |
| IDA2 → IDA3 | 55.74% | 0.5862 | 55.12% (±2.01%) |
| IDA3 → BM | 55.60% | 0.5582 | 55.17% (±2.05%) |

The gap between DAM→IDA1 and all BM-related transitions is stark. The BM reflects real-time system imbalances driven by events that are fundamentally harder to anticipate from earlier-horizon data.

---

## Trading Strategy Results

Results at optimal confidence threshold **0.20** (model probability outside [0.30, 0.70]):

| Transition | Test Size | Trades | % Traded | Hit Rate | €/MWh | Total P&L (€) |
|---|---|---|---|---|---|---|
| **DAM → BM** | 13,783 | 4,258 | 30.9% | 70.36% | **€33.75** | **€480,887** |
| IDA1 → BM | 13,829 | 4,064 | 29.4% | 65.28% | €17.67 | €271,178 |
| IDA2 → BM | 6,928 | 1,850 | 26.7% | 63.84% | €17.36 | €95,732 |
| **DAM → IDA1** | 13,653 | **7,350** | **53.8%** | **80.93%** | €14.11 | €375,056 |
| IDA3 → BM | 3,467 | 907 | 26.2% | 59.21% | €9.64 | €33,689 |
| IDA1 → IDA2 | 6,857 | 2,776 | 40.5% | 58.43% | €7.18 | €83,058 |
| IDA2 → IDA3 | 3,417 | 952 | 27.9% | 59.98% | €7.96 | €26,469 |

All figures are gross of transaction costs and market impact. Max drawdown for DAM→BM over the test period: **~€15,852**.

---

## Pre-generated Results

The repo includes report-level outputs so GitHub visitors can inspect findings immediately without running any code:

| File | Description |
|---|---|
| `outputs/original_model_performance.csv` | Accuracy, ROC-AUC and CV results by transition |
| `outputs/original_trading_performance.csv` | Strategy results at confidence threshold 0.20 |
| `outputs/original_market_price_summary.csv` | Cleaned market price summary statistics |
| `reports/figures/original_model_accuracy.png` | Bar chart of test accuracy by transition |
| `reports/figures/original_total_pnl.png` | Bar chart of total gross P&L |
| `reports/figures/original_accuracy_vs_profitability.png` | Accuracy vs profitability scatter |

---

## Setup & Installation

```bash
# Clone the repository
git clone https://github.com/SnehankurSarkar/isem-arbitrage-analysis.git
cd isem-arbitrage-analysis

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

---

## Reproducing Locally with the Private Dataset

The raw Energia dataset is proprietary and intentionally excluded. To reproduce the full results locally, place the file here:

```text
data/private/MarketData_2022-2026.parquet
```

Then run:

```bash
python scripts/01_run_reproduction_pipeline.py \
  --data data/private/MarketData_2022-2026.parquet \
  --outputs outputs \
  --figures reports/figures
```

Or simply:

```bash
make test
```

A synthetic sample is included under `data/sample/` for unit tests only. It is **not** the Energia dataset and must not be used to validate the reported results.

---

## Limitations

- Reported P&L is gross of transaction costs and slippage
- Market impact and liquidity constraints are not modelled
- BM-related transitions are more volatile and harder to forecast
- Live deployment would require rolling retraining, cost modelling, risk limits and real-time monitoring
- Public reproduction of the exact numbers requires the private Energia dataset

---

## Portfolio Value

This project demonstrates:

- Energy-market domain understanding (I-SEM structure, auction sequencing, arbitrage mechanics)
- Time-series leakage control (strict temporal splits, walk-forward CV, gate-closure information barriers)
- Feature engineering from system fundamentals (net demand, wind consensus, forecast revisions, cyclical encoding)
- LightGBM classification and regression modelling
- Confidence-filtered, magnitude-proportional trading strategy design
- P&L and drawdown analysis
- Professional Python packaging, CI, and documentation

---

## Acknowledgements

This project was completed as part of the DSA8023 Analytathon at Queen’s University Belfast using Energia-provided I-SEM market data. I would like to acknowledge my group members : Cathal Curran, Febin Jaison Thuruthiyil and Ujjwala Amalanathan, for their collaboration, discussions, and shared contributions during the project development process. Their input helped shape the analytical direction, interpretation of results, and overall project workflow.

Any public-safe restructuring, modularisation, documentation, and GitHub portfolio presentation in this repository are my own responsibility.

---

## Author

**Snehankur Sarkar**  
MSc Data Analytics · Queen's University Belfast · DSA8023  

---

## License

This project is licensed under the [MIT License](LICENSE). The underlying market data is proprietary to Energia and is excluded from this licence.
