# Data dictionary

This repository does not include the proprietary Energia dataset. The columns below reflect the fields expected by the reproduction pipeline.

| Column group | Examples | Use |
|---|---|---|
| Time | `TradeDate`, `Time`, `StartDateTime`, `ts`, `hod` | Delivery interval and temporal modelling |
| Market prices | `PriceDAM`, `PriceIDA1`, `PriceIDA2`, `PriceIDA3`, `PriceImbalance` | Spread target construction |
| Demand | `EirGridDemandFc_DAM`, `EirGridDemandFc_IDA1`, `Meteo_ISEMDEMAND_DAM` | System fundamentals and forecast disagreement |
| Wind | `EirGridWindFc_DAM`, `Meteo_ISEMWIND_DAM`, `EmSys_C_ISEMWIND_DAM`, `EmSys_U_ISEMWIND_DAM` | Renewable-output features and uncertainty |
| Solar | `ISEMSOLAR_DAM`, `ISEMSOLAR_IDA1`, `ISEMSOLAR_IDA2`, `ISEMSOLAR_IDA3` | Net-demand and renewable-penetration features |
| System variables | `NIV_Actual`, `TotalPN`, `PredictedICFlow`, `PumpStorage` | Market/system context |
| GB prices | `GB DAM Epex`, `GB DAM N2EX`, `GB DAM HH Epex` | Cross-market price-spread features |
| Post-auction flows | `PostIDA1Flow`, `PostIDA2Flow` | Intraday/BM transition features |

The synthetic sample in `data/sample/` provides similarly named columns for repository tests only.
