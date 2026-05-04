# Methodology — I-SEM Arbitrage Analytics

## 1. Problem Framing

Electricity arbitrage in the I-SEM exploits price differences between consecutive auction markets. The key question is whether these spreads are **predictable** from information available at gate closure, and if so, whether that predictability translates into **trading profit after accounting for risk**.

The problem is decomposed into two tasks:

| Task | Type | Model |
|---|---|---|
| Predict spread **direction** | Binary classification | LightGBM Classifier |
| Predict spread **magnitude** | Regression | LightGBM Regressor |

---

## 2. Data Preparation

### 2.1 Loading

- Source: `MarketData_2022-2026.parquet` (Energia proprietary dataset)
- Initial shape: **72,945 rows × 105 columns**
- Half-hourly resolution; delivery intervals from January 2022 to December 2025

### 2.2 Deduplication

- **2,721 fully duplicate rows** identified and removed
- **144 duplicate timestamps** arising from a split in the `PredictedICFlow` column were collapsed to single records using a NaN-safe mean

### 2.3 Timestamp Construction

A single `ts` (timestamp) column was constructed from the parsed `StartDateTime` column, serving as the canonical temporal index for all time-based operations.

### 2.4 Essential Variable Filtering

Rows missing any essential market price variable were dropped. Final analytical dataset: **70,006 observations**.

### 2.5 Handling Market-Structural Missingness

IDA2 and IDA3 prices are only populated during the windows when those auctions are active:
- **IDA2:** available for delivery intervals within 11:00–23:00
- **IDA3:** available for delivery intervals within 17:00–23:00

Solar forecasts for IDA2/IDA3 contained expected overnight NaN values and unexpected non-zero entries outside auction windows — both corrected.

### 2.6 Wind Data

- Actual wind generation entries with physically implausible negative values (< 0.01% of rows) were clipped to zero
- Wind forecast gaps filled via cascading fallback: **Meteo → EmSys** providers in order of availability

### 2.7 Outlier Treatment

A relaxed **3×IQR threshold** diagnostic confirmed that extreme price observations (~2% of the dataset) are economically meaningful market events, not recording errors. These were **retained**.

---

## 3. Exploratory Data Analysis

### 3.1 Spread Distribution

Seven spreads were computed:

```
spr_IDA1_DAM  = IDA1 - DAM
spr_BM_IDA1   = BM  - IDA1
spr_IDA2_IDA1 = IDA2 - IDA1
spr_IDA3_IDA2 = IDA3 - IDA2
spr_BM_IDA2   = BM  - IDA2
spr_BM_IDA3   = BM  - IDA3
spr_BM_DAM    = BM  - DAM
```

Key observations:
- DAM→IDA1, IDA1→BM, and DAM→BM spreads are approximately **50/50 directional** — no naive majority-class bias to exploit
- IDA1→IDA2 and IDA2→IDA3 spreads show **negative skew** (~76% and ~87% negative, respectively), reflecting the structural tendency for later intraday auctions to settle lower

### 3.2 Intraday Seasonality

Mean spread magnitudes and standard deviations are **systematically higher during morning (07:00–09:00) and evening (17:00–20:00) peak hours**. Overnight spreads are narrow. This motivates including time-of-day as a model feature.

### 3.3 System Fundamentals

Conditional analysis confirms:
- **Low wind generation** → IDA1→BM spread averages **+€11/MWh**
- **High wind generation** → IDA1→BM spread averages **−€12/MWh**
- **Higher net demand** → larger spreads and higher prices
- Relationships are **non-linear**, motivating ensemble methods over linear models

---

## 4. Feature Engineering

All features are constructed **strictly from information available at each market's gate closure**. No future data leaks into model inputs.

### 4.1 Cyclical Calendar Encoding

```python
hour_sin = sin(2π × hour / 24)
hour_cos = cos(2π × hour / 24)
# Similarly for day_of_week (7) and month (12)
```

This preserves the periodic structure of time (e.g., 23:30 is adjacent to 00:00).

### 4.2 Net Demand

```python
net_demand_DAM  = demand_DAM  - wind_DAM  - solar_DAM
net_demand_IDA1 = demand_IDA1 - wind_IDA1 - solar_IDA1
# ...per auction horizon
```

### 4.3 Wind Consensus and Uncertainty

```python
wind_mean_DAM = mean([Meteo_wind_DAM, EmSys_wind_DAM, EirGrid_wind_DAM])
wind_std_DAM  = std([Meteo_wind_DAM, EmSys_wind_DAM, EirGrid_wind_DAM])
```

The inter-provider spread serves as a proxy for **forecast confidence**.

### 4.4 Forecast Revision

```python
wind_revision_IDA1 = wind_IDA1 - wind_DAM
demand_revision_IDA1 = demand_IDA1 - demand_DAM
```

Captures how the system picture has changed between successive auction closures.

### 4.5 Lag Variables

Lag windows of **1-day (48 periods), 2-day, 7-day, and 14-day** for both spread and price columns, capturing momentum and mean-reversion dynamics.

---

## 5. Model Architecture

### 5.1 Preprocessing Pipeline

```
Input Features
    │
    ├── Numeric Columns
    │       └── SimpleImputer(median) → StandardScaler
    └── No categorical columns (all features are engineered numerics)
```

### 5.2 LightGBM Classifier

- **Objective:** Binary classification (spread direction)
- **Output:** Probability that spread > 0 (class +1)
- **Evaluation metrics:** Accuracy, ROC-AUC, confusion matrix, classification report

### 5.3 LightGBM Regressor

- **Objective:** Regression (spread magnitude in €/MWh)
- **Output:** Predicted spread value
- **Evaluation metrics:** MAE, R²

### 5.4 Model Validation

- **Train/test split:** First 80% training, final 20% held out — **strictly temporal, no shuffling**
- **Cross-validation:** 5-fold walk-forward `TimeSeriesSplit` to assess temporal stability
- This approach prevents look-ahead bias that would arise from random k-fold CV on time-series data

---

## 6. Trading Strategy

### 6.1 Signal Generation

```python
p_up = classifier.predict_proba(X_test)[:, class_1_index]
direction = +1 if p_up > 0.5 else -1
```

### 6.2 Confidence Filter

```python
confidence = abs(p_up - 0.5)
trade = True if confidence > threshold else False  # threshold = 0.20
```

Only intervals where the model assigns probability outside [0.30, 0.70] are traded.

### 6.3 Position Sizing (Magnitude-Proportional)

```python
predicted_magnitude = regressor.predict(X_test)
p25 = max(np.percentile(abs(predicted_magnitude), 25), 0.5)
volume = np.clip(abs(predicted_magnitude) / p25, 0.1, 10.0)
```

Higher forecast spreads receive proportionally larger positions.

### 6.4 P&L Calculation

```python
pnl = volume * abs(actual_spread)  if direction == sign(actual_spread)
    = -volume * abs(actual_spread) otherwise
    = 0                             if no trade (confidence filtered)
```

### 6.5 Threshold Tuning

The confidence threshold was selected by grid-searching across candidate values {0.05, 0.10, 0.15, 0.20, 0.25} and choosing the value that **maximised €/MWh profitability while preserving a viable trade count**.

---

## 7. Limitations and Caveats

| Limitation | Details |
|---|---|
| Transaction costs | Not modelled; real-world performance will be lower |
| Market impact | Assumes unlimited liquidity at published prices |
| No re-training | Models trained once on 2022–2025 data; live deployment would require rolling refit |
| Proprietary data | Results are not reproducible without the Energia dataset |
| Seasonal non-stationarity | High 2022 energy-crisis volatility is included in training; such regimes may not repeat |

---

## 8. Tools and Libraries

| Tool | Version | Purpose |
|---|---|---|
| `pandas` | ≥2.1 | Data manipulation |
| `numpy` | ≥1.26 | Numerical computing |
| `lightgbm` | ≥4.3 | Gradient boosted models |
| `scikit-learn` | ≥1.4 | Pipelines, imputation, metrics, CV |
| `matplotlib` | ≥3.8 | Visualisation |
| `pyarrow` | ≥14 | Parquet I/O |
