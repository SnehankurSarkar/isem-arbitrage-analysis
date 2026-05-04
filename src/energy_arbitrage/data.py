"""Data loading and cleaning logic for the I-SEM arbitrage project.

Functions are copied from the submitted notebook and only reorganised into a module.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

def load_market_data(path: str | Path) -> pd.DataFrame:
    """Load market data from parquet or CSV.

    The original notebook converts MarketData_2022-2026.parquet to MarketData.csv and then
    reads the CSV. This function preserves the same supported inputs while avoiding a forced
    intermediate file in the repository.
    """
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        data = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        data = pd.read_csv(path)
    else:
        raise ValueError("Input data must be a .parquet or .csv file")
    if "Unnamed: 0" in data.columns:
        data = data.drop(columns="Unnamed: 0")
    return data


def clean_market_data(data: pd.DataFrame) -> pd.DataFrame:
    """Cleaning logic copied from the submitted notebook."""
    data = data.drop_duplicates()

    data["TradeDate"] = pd.to_datetime(data["TradeDate"], errors="coerce")
    data["Time"] = pd.to_datetime(data["Time"], format="%H:%M:%S", errors="coerce").dt.time
    data["StartDateTime"] = pd.to_datetime(data["StartDateTime"], errors="coerce")
    data = data.copy()

    data["ts"] = pd.to_datetime(data["TradeDate"].astype(str) + " " + data["Time"].astype(str), errors="coerce")
    data = data.sort_values("ts").reset_index(drop=True)
    data["hour"] = data["ts"].dt.hour
    data["minute"] = data["ts"].dt.minute
    data["hod"] = data["hour"] + data["minute"] / 60

    if "PredictedICFlow" in data.columns:
        base_cols = [c for c in data.columns if c not in ["ts", "PredictedICFlow"]]
        base = data.groupby("ts", as_index=False)[base_cols].first()
        icflow = data.groupby("ts", as_index=False)["PredictedICFlow"].mean()
        data = base.merge(icflow, on="ts", how="left")
        data = data.copy()

    if "EirgridActualWind" in data.columns:
        neg_mask = data["EirgridActualWind"] < 0
        data.loc[neg_mask, "EirgridActualWind"] = 0

    data = data.replace(" ", np.nan)

    must_have = [
        "ts",
        "TradeDate",
        "Time",
        "hour",
        "minute",
        "hod",
        "EirgridActualDemand",
        "EirgridActualWind",
        "NIV_Actual",
        "TotalPN",
        "Aggregated Forecast",
        "EirGridDemandFc_DAM",
        "EirGridDemandFc_IDA1",
    ]
    for c in ["PriceDAM", "PriceIDA1", "PriceImbalance"]:
        if c in data.columns:
            must_have.append(c)
    must_have = [c for c in must_have if c in data.columns]

    if "ISEMSOLAR_DAM" in data.columns:
        data["ISEMSOLAR_DAM"] = data["ISEMSOLAR_DAM"].fillna(0)
    if "ISEMSOLAR_IDA1" in data.columns:
        data["ISEMSOLAR_IDA1"] = data["ISEMSOLAR_IDA1"].fillna(0)

    if {"EirGridWindFc_DAM", "EmSys_U_ISEMWIND_DAM", "Meteo_ISEMWIND_DAM"}.issubset(data.columns):
        data["EirGridWindFc_DAM"] = (
            data["EirGridWindFc_DAM"]
            .fillna(data["EmSys_U_ISEMWIND_DAM"])
            .fillna(data["Meteo_ISEMWIND_DAM"])
            .fillna(0)
        )
    if {"EirGridWindFc_IDA1", "EmSys_U_ISEMWIND_IDA1", "Meteo_ISEMWIND_IDA1"}.issubset(data.columns):
        data["EirGridWindFc_IDA1"] = (
            data["EirGridWindFc_IDA1"]
            .fillna(data["EmSys_U_ISEMWIND_IDA1"])
            .fillna(data["Meteo_ISEMWIND_IDA1"])
            .fillna(0)
        )

    data1 = data.dropna(subset=must_have)
    data1 = data1.sort_values("ts").reset_index(drop=True).copy()

    IDA2_WINDOW = (11.0, 23.0)
    IDA3_WINDOW = (17.0, 23.0)
    ida2_should_exist = (data1["hod"] >= IDA2_WINDOW[0]) & (data1["hod"] < IDA2_WINDOW[1])
    ida3_should_exist = (data1["hod"] >= IDA3_WINDOW[0]) & (data1["hod"] < IDA3_WINDOW[1])

    if "ISEMSOLAR_IDA2" in data1.columns:
        data1.loc[~ida2_should_exist, "ISEMSOLAR_IDA2"] = np.nan
        mask1_ida2_fill = ida2_should_exist & data1["ISEMSOLAR_IDA2"].isna()
        data1.loc[mask1_ida2_fill, "ISEMSOLAR_IDA2"] = 0.0

    if "ISEMSOLAR_IDA3" in data1.columns:
        mask2_ida3_fill = ida3_should_exist & data1["ISEMSOLAR_IDA3"].isna()
        data1.loc[mask2_ida3_fill, "ISEMSOLAR_IDA3"] = 0.0

    if {"EirGridWindFc_IDA3", "EmSys_U_ISEMWIND_IDA3", "Meteo_ISEMWIND_IDA3"}.issubset(data1.columns):
        mask3_ida3_fill = ida3_should_exist & data1["EirGridWindFc_IDA3"].isna()
        data1.loc[mask3_ida3_fill, "EirGridWindFc_IDA3"] = (
            data1.loc[mask3_ida3_fill, "EmSys_U_ISEMWIND_IDA3"]
            .fillna(data1.loc[mask3_ida3_fill, "Meteo_ISEMWIND_IDA3"])
            .fillna(0.0)
        )

    data1 = data1.sort_values("ts").reset_index(drop=True)
    if "Time" in data1.columns:
        data1 = data1.drop(columns=["Time"])
    data1 = data1.reset_index(drop=True)
    return data1


def add_spreads_and_fundamentals(data1: pd.DataFrame) -> pd.DataFrame:
    """Create spreads and net-demand fundamentals exactly as in the notebook."""
    df = data1.copy()
    df = df.sort_values("ts").reset_index(drop=True)

    if "PriceDAM" in df.columns and "PriceIDA1" in df.columns:
        df["spr_IDA1_DAM"] = df["PriceIDA1"] - df["PriceDAM"]
    if "PriceIDA1" in df.columns and "PriceImbalance" in df.columns:
        df["spr_BM_IDA1"] = df["PriceImbalance"] - df["PriceIDA1"]
    if "PriceDAM" in df.columns and "PriceImbalance" in df.columns:
        df["spr_BM_DAM"] = df["PriceImbalance"] - df["PriceDAM"]
    if "PriceIDA1" in df.columns and "PriceIDA2" in df.columns:
        df["spr_IDA2_IDA1"] = df["PriceIDA2"] - df["PriceIDA1"]
    if "PriceIDA2" in df.columns and "PriceIDA3" in df.columns:
        df["spr_IDA3_IDA2"] = df["PriceIDA3"] - df["PriceIDA2"]
    if "PriceIDA2" in df.columns and "PriceImbalance" in df.columns:
        df["spr_BM_IDA2"] = df["PriceImbalance"] - df["PriceIDA2"]
    if "PriceIDA3" in df.columns and "PriceImbalance" in df.columns:
        df["spr_BM_IDA3"] = df["PriceImbalance"] - df["PriceIDA3"]
    if "PriceDAM" in df.columns and "PriceImbalance" in df.columns:
        df["spr_BM_DAM"] = df["PriceImbalance"] - df["PriceDAM"]

    for s in [c for c in df.columns if c.startswith("spr_")]:
        df[f"dir_{s}"] = np.where(df[s] >= 0, 1, -1)

    df["NetDemand_DAM"] = df["EirGridDemandFc_DAM"] - df["EirGridWindFc_DAM"] - df["ISEMSOLAR_DAM"]
    df["NetDemand_IDA1"] = df["EirGridDemandFc_IDA1"] - df["EirGridWindFc_IDA1"] - df["ISEMSOLAR_IDA1"]
    df["NetDemand_IDA2"] = df["EirGridDemandFc_IDA2"] - df["EirGridWindFc_IDA2"] - df["ISEMSOLAR_IDA2"]
    df["NetDemand_IDA3"] = df["EirGridDemandFc_IDA3"] - df["EirGridWindFc_IDA3"] - df["ISEMSOLAR_IDA3"]
    return df


