# Results Summary — I-SEM Arbitrage Analytics

## 1. Dataset After Cleaning

| Metric | Value |
|---|---|
| Raw rows | 72,945 |
| Duplicate rows removed | 2,721 |
| Duplicate timestamps resolved | 144 |
| **Final analytical observations** | **70,006** |
| Date range | Jan 2022 – Dec 2025 |
| Temporal resolution | 30 minutes |

### Market Price Statistics

| Market | Mean (€/MWh) | Std Dev | Median | Min | Max | Obs |
|---|---|---|---|---|---|---|
| DAM | 142.86 | 84.57 | 119.00 | −30.00 | 705.47 | 70,006 |
| IDA1 | 141.23 | 84.91 | 117.77 | −70.54 | 724.00 | 70,006 |
| IDA2 | 152.77 | 89.33 | 127.26 | −37.19 | 670.00 | 35,024 |
| IDA3 | 166.33 | 92.39 | 142.00 | −17.00 | 661.04 | 17,512 |
| BM | 141.32 | 109.07 | 126.11 | −456.34 | 874.25 | 70,006 |

> The Balancing Market shows the widest price range (−€456 to +€874/MWh), reflecting its real-time nature and exposure to unexpected system events.

---

## 2. Spread Characterisation

| Spread | Direction Split (+ / −) | Key Observation |
|---|---|---|
| DAM → IDA1 | ~50 / 50 | Near-random directional balance; genuine predictive challenge |
| IDA1 → IDA2 | ~24% positive | Strong negative skew; IDA2 consistently below IDA1 (~76% of the time) |
| IDA2 → IDA3 | ~13% positive | Very strong negative skew (~87% negative); later auctions settle lower |
| IDA1 → BM | ~50 / 50 | Balanced; BM driven by real-time deviations |
| DAM → BM | ~50 / 50 | Balanced; hardest to predict but highest spread magnitude |

### Wind–Spread Relationship

| Wind Regime | IDA1→BM Mean Spread |
|---|---|
| Low wind (Q1) | **+€11/MWh** |
| High wind (Q4) | **−€12/MWh** |

Renewable output systematically shifts the balance between markets, creating directional arbitrage opportunities that are partially predictable.

---

## 3. Classification Model Performance

All models use LightGBM with strict temporal train/test split (80/20) and 5-fold walk-forward CV.

| Transition | Test Accuracy | ROC-AUC | CV Mean (±Std) | Interpretation |
|---|---|---|---|---|
| **DAM → IDA1** | **70.18%** | **0.7718** | **69.76% (±2.00%)** | Strongly predictable; stable over time |
| DAM → BM | 58.54% | 0.6161 | 56.99% (±2.20%) | Moderate signal; BM hard to forecast |
| IDA1 → IDA2 | 57.18% | 0.6178 | 56.72% (±1.98%) | Slight predictability |
| IDA2 → BM | 56.12% | 0.5653 | 55.08% (±1.89%) | Near-random |
| IDA2 → IDA3 | 55.74% | 0.5862 | 55.12% (±2.01%) | Near-random |
| IDA3 → BM | 55.60% | 0.5582 | 55.17% (±2.05%) | Near-random |
| IDA1 → BM | 56.24% | 0.5779 | 55.17% (±1.92%) | Near-random |

**DAM→IDA1 Confusion Matrix (test set):**

```
                  Predicted ↓ (−1)   Predicted ↑ (+1)
Actual ↓ (−1)         5,777               2,026           (74% recall)
Actual ↑ (+1)         2,045               3,805           (65% recall)
```

Accuracy: **70.18%** | Recall ↓: 74.0% | Recall ↑: 65.0%

---

## 4. Trading Strategy Results

### Optimal Confidence Threshold: **0.20**

Only intervals where |p − 0.5| > 0.20 are traded (model probability outside [0.30, 0.70]).

| Transition | Test Size | Trades | % Traded | Hit Rate | €/MWh | Total P&L (€) |
|---|---|---|---|---|---|---|
| **DAM → BM** | 13,783 | 4,258 | 30.9% | **70.36%** | **€33.75** | **€480,887** |
| IDA1 → BM | 13,829 | 4,064 | 29.4% | 65.28% | €17.67 | €271,178 |
| IDA2 → BM | 6,928 | 1,850 | 26.7% | 63.84% | €17.36 | €95,732 |
| **DAM → IDA1** | 13,653 | **7,350** | **53.8%** | 80.93% | €14.11 | €375,056 |
| IDA3 → BM | 3,467 | 907 | 26.2% | 59.21% | €9.64 | €33,689 |
| IDA1 → IDA2 | 6,857 | 2,776 | 40.5% | 58.43% | €7.18 | €83,058 |
| IDA2 → IDA3 | 3,417 | 952 | 27.9% | 59.98% | €7.96 | €26,469 |

### Risk Summary

| Transition | Total P&L (€) | Max Drawdown (€) | Sharpe Proxy |
|---|---|---|---|
| DAM → BM | €480,887 | ~€15,852 | High return, higher risk |
| DAM → IDA1 | €375,056 | Lower | High volume, stable |

---

## 5. Key Insights

### 5.1 Predictability Is Not Uniform

The gap between DAM→IDA1 (70.2% AUC 0.77) and BM-related transitions (55–58%, AUC 0.56–0.62) is stark. The IDA1 price is set by the same demand/wind/solar information that is progressively refined from the DAM. The BM, by contrast, reflects real-time deviations from dispatch that are fundamentally harder to anticipate from earlier-horizon data.

### 5.2 Confidence Filtering Is Essential

Without filtering, the DAM→IDA1 model executes many low-conviction trades that dilute €/MWh profitability. At threshold 0.20, the hit rate jumps to **80.93%** and 53.8% of intervals still trade — a meaningful coverage rate.

### 5.3 Magnitude Sizing Adds Value

By scaling position size to the regression model's predicted spread magnitude, the strategy concentrates risk-adjusted exposure on intervals where the model forecasts large spreads — the intervals most likely to generate material P&L.

### 5.4 DAM→BM: High Reward, Elevated Risk

Despite mediocre classification accuracy (58.5%), the DAM→BM transition produces the highest €/MWh (€33.75) because the BM spread, when correctly called, tends to be large. The max drawdown of ~€15,852 and the strategy's reliance on a 70.36% hit rate mean that sustained losing runs are possible and must be monitored.

### 5.5 Seasonal Non-Stationarity

Monthly spread volatility is non-constant across the year, with elevated volatility during the 2022 energy crisis period. Model performance should be re-evaluated on a rolling monthly basis in live deployment.

---

## 6. Recommendations

| Priority | Action |
|---|---|
| **Primary strategy** | Deploy DAM → IDA1 at confidence threshold 0.20 with magnitude sizing |
| **Supplementary strategy** | Run DAM → BM at reduced position volume with a hard drawdown stop |
| **Monitoring** | Assess rolling monthly accuracy and P&L; retrain quarterly or when accuracy degrades by >3pp |
| **Seasonality** | Consider season-specific model variants or explicit seasonal interaction features |
| **Costs** | Incorporate transaction cost estimates before live deployment; these results are gross of costs |
