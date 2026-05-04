# Fidelity notes

This repository has been rebuilt so that the professional Python package represents the original notebook logic rather than a simplified demonstration pipeline.

## Source of truth

The original notebook is preserved at:

```text
notebooks/original/Analytathon1.ipynb
```

A direct script export is also included at:

```text
scripts/original_notebook_export.py
```

The test suite checks the SHA-256 hash of the original notebook so accidental edits are detected.

## Preserved logic

The reproduction module preserves the following original choices:

| Component | Preserved value / logic |
|---|---|
| Main model | LightGBM when available |
| Classifier estimators | 800 |
| Regressor estimators | 800 |
| Learning rate | 0.03 |
| Max depth | 6 |
| Num leaves | 63 |
| Min child samples | 30 |
| Subsample | 0.8 |
| Column sample by tree | 0.7 |
| Regularisation | reg_alpha = 0.1, reg_lambda = 1.0 |
| Fallback model | GradientBoostingClassifier / GradientBoostingRegressor |
| Lag burn-in | 48 × 14 half-hours |
| Train-test split | first 80% train, final 20% test |
| CV | TimeSeriesSplit with 5 folds |
| Confidence thresholds | 0.00, 0.05, 0.10, 0.15, 0.20 |
| Best-threshold plotting | 0.20 |
| Position sizing | abs(predicted spread) / 25th percentile, clipped to [0.1, 10.0] |

## Operational changes only

The package version changes only things required for a professional repository:

1. The private data path is passed as a command-line argument.
2. Figures are saved to `reports/figures/` instead of displayed interactively.
3. Output tables are written to `outputs/`.
4. The proprietary dataset is excluded from git.
5. A synthetic sample exists only for structural tests.

No intentional numerical or methodological simplification has been introduced into `src/energy_arbitrage/original_logic.py`.
