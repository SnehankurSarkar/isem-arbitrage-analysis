# Executive Case Study — I-SEM Electricity Market Arbitrage

## Problem

Energia provided half-hourly I-SEM market data covering sequential auction prices, demand, wind and solar forecasts, interconnector flows and system variables. The business question was whether price spreads between consecutive market stages could be predicted early enough to support an arbitrage trading strategy.

## Analytical approach

The project separated the trading decision into two linked prediction problems:

1. **Direction:** will the later market price settle above or below the earlier market price?
2. **Magnitude:** how large is the expected spread in €/MWh?

This allows the strategy to trade only where the classifier has sufficient confidence, then scale position size according to the regression model's predicted spread magnitude.

## Data preparation

The original dataset contained 72,945 rows and 105 columns. After duplicate handling, timestamp consolidation, market-window checks and removal of rows missing essential variables, the final analytical dataset contained 70,006 half-hourly observations. Extreme prices were retained because they were economically meaningful market events rather than data-entry errors.

## Modelling design

LightGBM classification and regression models were trained for seven market transitions. The pipeline used a strict chronological 80/20 split, 5-fold walk-forward cross-validation and a 14-day lag burn-in. Feature construction respected market timing so that later-market information was not used to predict earlier decisions.

## Main finding

The most predictable transition was **DAM → IDA1**, with **70.18% test accuracy** and **0.7718 ROC-AUC**. The most profitable transition in the strategy simulation was **DAM → BM**, with **€33.75/MWh**, **€480,887 total gross P&L**, **4,258 executed trades**, and a **70.36% hit rate** at confidence threshold 0.20.

## Recommendation

The recommended deployment design is to use **DAM → IDA1** as the primary, more stable signal and **DAM → BM** as a reduced-volume supplementary strategy because it offers higher return but greater drawdown risk. Any live version would need transaction costs, liquidity limits, monthly monitoring and rolling retraining.

## Portfolio relevance

This project demonstrates end-to-end data analytics capability: data cleaning, market-domain reasoning, feature engineering, leakage control, gradient-boosted modelling, temporal validation, strategy design, risk interpretation and executive communication.
