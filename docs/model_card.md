# Model Card — I-SEM Price-Spread Direction and Magnitude Models

## Intended use

The models estimate short-term price-spread direction and magnitude between sequential I-SEM market stages. The outputs are used to evaluate whether confidence-filtered arbitrage decisions could have produced positive gross P&L historically.

## Not intended for

- Live trading without transaction costs and liquidity assumptions
- Automated deployment without rolling monitoring
- Claims of guaranteed profitability
- Public reproduction without the proprietary Energia dataset

## Model architecture

Two models are fitted per transition:

| Task | Target | Model |
|---|---|---|
| Direction | `sign(spread)` | LightGBM classifier |
| Magnitude | `spread` in €/MWh | LightGBM regressor |

Gradient Boosting fallback models are included only for environments where LightGBM is unavailable.

## Validation

- Chronological train/test split: first 80% train, last 20% test
- 5-fold walk-forward cross-validation
- No random shuffling
- Lag burn-in applied before modelling
- Transition-specific feature windows used to avoid future-market leakage

## Key reported results

| Transition | Test Accuracy | ROC-AUC | CV Accuracy |
|---|---:|---:|---:|
| DAM → IDA1 | 70.18% | 0.7718 | 69.76% ± 2.00% |
| DAM → BM | 58.54% | 0.6161 | 56.99% ± 2.20% |
| IDA1 → BM | 56.24% | 0.5779 | 55.17% ± 1.92% |

## Main risks

- Gross P&L excludes trading fees, slippage and market-impact constraints
- Structural market behaviour can shift over time
- 2022 crisis volatility may not generalise
- BM-related signals are harder to forecast and should be monitored closely
