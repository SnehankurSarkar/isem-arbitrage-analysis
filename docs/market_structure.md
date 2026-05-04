# I-SEM Market Structure Reference

## Overview

The **Integrated Single Electricity Market (I-SEM)** is the wholesale electricity market for the island of Ireland, covering both the Republic of Ireland and Northern Ireland. It is jointly operated by **SEMO** (Single Electricity Market Operator) and was restructured to its current form in October 2018 to align with European electricity market standards.

---

## Sequential Auction Structure

Electricity is traded through a cascade of sequential auction markets. Each gate closure represents a point at which traders can submit or revise bids and offers. As delivery time approaches, information improves (more accurate wind/demand forecasts, actual plant availability) and prices are re-set.

| Market | Abbreviation | Gate Closure | Coverage | Notes |
|---|---|---|---|---|
| Day-Ahead Market | DAM | ~12:00 the day before delivery | All 48 half-hour periods of next day | Pan-European coupling via EUPHEMIA |
| Intraday Auction 1 | IDA1 | ~15:00 the day before delivery | All 48 periods | First opportunity to refine DAM positions |
| Intraday Auction 2 | IDA2 | ~22:00 the day before delivery | Periods 23–48 (11:00–23:00) | Limited to later half of delivery day |
| Intraday Auction 3 | IDA3 | ~10:00 on delivery day | Periods 35–48 (17:00–23:00) | Evening delivery only |
| Balancing Market | BM | Continuous up to ~1hr before delivery | Real-time | Reflects actual system imbalance |

---

## Arbitrage Logic

Price spreads between consecutive markets arise because:

1. **New information** arrives between gate closures (updated wind/demand forecasts)
2. **System events** change dispatch requirements closer to real-time
3. **Portfolio rebalancing** by participants shifts supply/demand at different horizons

An arbitrage trade involves:
- **Buying** in the earlier market (e.g., DAM) when you expect the price to rise
- **Selling** in the later market (e.g., IDA1) when the price has risen — locking in the spread

Or conversely:
- **Selling** in the earlier market when you expect the price to fall
- **Buying** back in the later market at a lower price

---

## Key Market Variables

| Variable | Description | Role in Spread Prediction |
|---|---|---|
| **Wind generation forecast** | Predicted output from wind farms (MW) | High wind → lower prices; primary spread driver |
| **Solar generation forecast** | Predicted output from solar panels (MW) | Seasonal effect; suppresses midday prices |
| **System demand forecast** | Predicted electricity consumption (MW) | Higher demand → higher prices |
| **Net demand** | Demand − Wind − Solar | Composite indicator of price pressure |
| **Interconnector flows** | Power flows to/from Great Britain via IC links | Affects price convergence between markets |
| **Forecast revision** | Change in wind/demand forecast between gate closures | Direct proxy for "new information" |

---

## Forecast Providers

Multiple forecast providers supply wind and demand forecasts for each auction horizon:

| Provider | Variables |
|---|---|
| **Meteo** | Wind and demand forecasts for DAM, IDA1, IDA2, IDA3 |
| **EmSys** | Wind and demand forecasts (backup/alternative) |
| **EirGrid** | Actual wind and demand (post-delivery) |

The spread between providers' wind forecasts serves as a proxy for **forecast uncertainty** — a wider inter-provider range indicates less confidence in the wind prediction.

---

## Interconnectors

| Interconnector | Direction | Capacity |
|---|---|---|
| IE↔GB (Moyle / East-West) | Republic of Ireland ↔ Great Britain | ~500 MW each way |
| GB↔NI | Great Britain ↔ Northern Ireland | ~450 MW each way |
| IE2↔GB2 | Second interconnector (North-South / Celtic) | ~500+ MW |

Interconnector flows influence I-SEM prices by allowing arbitrage with the GB market (EPEX/N2EX). When GB prices are higher than I-SEM, power flows from Ireland to GB, raising I-SEM prices.

---

## References

- SEMO (2025). *I-SEM Market Overview*. https://www.sem-o.com
- CRU / Utility Regulator (2018). *I-SEM High Level Design*. Commission for Regulation of Utilities.
- Unwiejewski et al. (2019). Understanding intraday electricity markets. *International Journal of Forecasting*, 35.
