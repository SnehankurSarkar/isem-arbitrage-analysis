"""Project constants copied from the submitted Analytathon notebook.

The values in this file are intentionally explicit so that reviewers can see the
assumptions behind the reproduction pipeline without opening the notebook.
"""
from __future__ import annotations

RANDOM_STATE = 42
LAG_BURNIN = 48 * 14  # exact original 14-day lag warm-up period
BEST_THRESH = 0.20
CONF_THRESHOLDS = (0.0, 0.05, 0.10, 0.15, 0.20)

ALWAYS_EXCLUDE = {"ts", "TradeDate", "StartDateTime", "Time", "PriceImbalance"}
TARGET_PREFIXES = ("spr_", "dir_", "NetD_q_")

# transition name -> (spread column, allowed feature-name tokens, always-include features, optional delivery-window rule)
TRANSITIONS = {
    "DAM→IDA1": (
        "spr_IDA1_DAM",
        ["_DAM", "PRICEDAM", "GB_DAM", "GB DAM", "AGGREGATED", "LOADFORECAST", "_MW", "PUMPSTORAGE", "TOTALPN", "NetDemand_DAM", "GB_VS_IRL"],
        ["PriceDAM", "AggregatedForecast", "demand_revision_IDA1_DAM", "wind_revision_IDA1_DAM"],
        None,
    ),
    "IDA1→BM": (
        "spr_BM_IDA1",
        ["_DAM", "_IDA1", "PRICEDAM", "PRICEIDA1", "LOADFORECAST", "_MW", "PUMPSTORAGE", "TOTALPN", "NetDemand_IDA1", "NetDemand_DAM"],
        ["PriceDAM", "AggregatedForecast", "PredictedICFlow", "PriceIDA1"],
        None,
    ),
    "IDA1→IDA2": (
        "spr_IDA2_IDA1",
        ["_DAM", "_IDA1", "PRICEDAM", "PRICEIDA1", "LOADFORECAST", "_MW", "PUMPSTORAGE", "TOTALPN", "NetDemand_IDA1", "NetDemand_DAM"],
        ["PriceDAM", "AggregatedForecast", "PredictedICFlow", "PriceIDA1"],
        "hod >= 11",
    ),
    "IDA2→IDA3": (
        "spr_IDA3_IDA2",
        ["_DAM", "_IDA1", "_IDA2", "PRICEDAM", "PRICEIDA1", "PRICEIDA2", "LOADFORECAST", "_MW", "PUMPSTORAGE", "TOTALPN", "NetDemand_IDA2", "NetDemand_IDA1", "NetDemand_DAM"],
        ["PriceDAM", "PriceIDA1", "PriceIDA2", "PostIDA1Flow", "AggregatedForecast"],
        "hod >= 17",
    ),
    "DAM→BM": (
        "spr_BM_DAM",
        ["_DAM", "PRICEDAM", "GB_DAM", "GB DAM", "AGGREGATED", "LOADFORECAST", "_MW", "PUMPSTORAGE", "TOTALPN", "NetDemand_DAM", "GB_VS_IRL"],
        ["PriceDAM", "AggregatedForecast", "demand_revision_IDA1_DAM", "wind_revision_IDA1_DAM"],
        None,
    ),
    "IDA2→BM": (
        "spr_BM_IDA2",
        ["_DAM", "_IDA1", "_IDA2", "PRICEDAM", "PRICEIDA1", "PRICEIDA2", "LOADFORECAST", "_MW", "PUMPSTORAGE", "TOTALPN", "NetDemand_IDA2", "NetDemand_IDA1", "NetDemand_DAM"],
        ["PriceDAM", "PriceIDA1", "PriceIDA2", "PostIDA1Flow", "AggregatedForecast"],
        "hod >= 11",
    ),
    "IDA3→BM": (
        "spr_BM_IDA3",
        ["_DAM", "_IDA1", "_IDA2", "_IDA3", "PRICEDAM", "PRICEIDA1", "PRICEIDA2", "PRICEIDA3", "LOADFORECAST", "_MW", "PUMPSTORAGE", "TOTALPN", "NetDemand_IDA3", "NetDemand_IDA2", "NetDemand_IDA1", "NetDemand_DAM"],
        ["PriceDAM", "PriceIDA1", "PriceIDA2", "PriceIDA3", "PostIDA1Flow", "PostIDA2Flow", "AggregatedForecast"],
        "hod >= 17",
    ),
}
