# Code Fidelity Notes

This repository is designed to be both professional and faithful to the submitted Analytathon notebook.

## What is preserved exactly

The modular package keeps the same modelling and trading assumptions used in the original notebook:

- LightGBM classifier/regressor as the primary model family
- `n_estimators=800`
- `learning_rate=0.03`
- `max_depth=6`
- `num_leaves=63`
- `min_child_samples=30`
- `subsample=0.8`
- `colsample_bytree=0.7`
- `reg_alpha=0.1`
- `reg_lambda=1.0`
- `random_state=42`
- 14-day lag burn-in: `48 * 14 = 672` half-hourly periods
- 80/20 chronological train/test split
- 5-fold walk-forward `TimeSeriesSplit`
- IDA2 window rule: `hod >= 11`
- IDA3 window rule: `hod >= 17`
- Confidence filtering with best threshold `0.20`
- Magnitude-proportional volume sizing

## What changed for professionalism

The original notebook logic has been split into maintainable modules:

| Module | Responsibility |
|---|---|
| `data.py` | loading, timestamp parsing, cleaning, market-structural missingness |
| `features.py` | spreads, net demand, cyclical encodings, lags, feature selection |
| `modeling.py` | preprocessing, LightGBM models, temporal split, walk-forward CV |
| `backtesting.py` | confidence filter, volume sizing, P&L, leaderboard |
| `visualization.py` | report-ready figures |
| `pipeline.py` | end-to-end orchestration |

This is a structural refactor, not a change in the modelling assumptions.

## Public-data limitation

The proprietary Energia dataset is not committed. The repository includes:

1. the original notebook as the source of truth under `notebooks/original/`;
2. a synthetic sample dataset for public structure/testing;
3. pre-generated CSV/PNG result summaries from the submitted analysis so visitors can see the real headline findings without accessing private data.

## How to reproduce with the private file

Place the private file locally at:

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
