"""Feature engineering and transition-specific feature selection.

The transformations mirror the submitted notebook; this module is separated only for maintainability.
"""
from __future__ import annotations

from typing import Iterable, Optional
import numpy as np
import pandas as pd

from energy_arbitrage.config import ALWAYS_EXCLUDE, TARGET_PREFIXES

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering copied from the submitted notebook."""
    df = df.copy()
    df["dow"] = df["ts"].dt.dayofweek
    df["month"] = df["ts"].dt.month
    df["hour_sin"] = np.sin(2 * np.pi * df["hod"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hod"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_winter"] = df["month"].isin([11, 12, 1, 2, 3]).astype(int)
    df["is_peak"] = df["hod"].between(7, 21).astype(int)

    for sfx in ["DAM", "IDA1", "IDA2", "IDA3"]:
        cands = [
            c
            for c in [
                f"Meteo_ISEMWIND_{sfx}",
                f"EmSys_C_ISEMWIND_{sfx}",
                f"EmSys_U_ISEMWIND_{sfx}",
                f"EirGridWindFc_{sfx}",
            ]
            if c in df.columns
        ]
        if len(cands) >= 2:
            df[f"wind_consensus_{sfx}"] = df[cands].mean(axis=1)
            df[f"wind_uncertainty_{sfx}"] = df[cands].std(axis=1)
            df[f"wind_range_{sfx}"] = df[cands].max(axis=1) - df[cands].min(axis=1)
            if f"EirGridWindFc_{sfx}" in df.columns:
                df[f"wind_eirg_dev_{sfx}"] = df[f"EirGridWindFc_{sfx}"] - df[f"wind_consensus_{sfx}"]

    for sfx in ["DAM", "IDA1", "IDA2", "IDA3"]:
        m, e = f"Meteo_ISEMDEMAND_{sfx}", f"EirGridDemandFc_{sfx}"
        if m in df.columns and e in df.columns:
            df[f"demand_disagreement_{sfx}"] = df[m] - df[e]

    if "EirGridWindFc_IDA1" in df.columns and "EirGridWindFc_DAM" in df.columns:
        df["wind_revision_IDA1_DAM"] = df["EirGridWindFc_DAM"] - df["EirGridWindFc_IDA1"]
    if "EirGridWindFc_IDA2" in df.columns and "EirGridWindFc_IDA1" in df.columns:
        df["wind_revision_IDA2_IDA1"] = df["EirGridWindFc_IDA1"] - df["EirGridWindFc_IDA2"]
    if "EirGridWindFc_IDA3" in df.columns and "EirGridWindFc_IDA2" in df.columns:
        df["wind_revision_IDA3_IDA2"] = df["EirGridWindFc_IDA2"] - df["EirGridWindFc_IDA3"]

    if "EirGridDemandFc_IDA1" in df.columns and "EirGridDemandFc_DAM" in df.columns:
        df["demand_revision_IDA1_DAM"] = df["EirGridDemandFc_IDA1"] - df["EirGridDemandFc_DAM"]
    if "EirGridDemandFc_IDA2" in df.columns and "EirGridDemandFc_IDA1" in df.columns:
        df["demand_revision_IDA2_IDA1"] = df["EirGridDemandFc_IDA1"] - df["EirGridDemandFc_IDA2"]
    if "EirGridDemandFc_IDA3" in df.columns and "EirGridDemandFc_IDA2" in df.columns:
        df["demand_revision_IDA3_IDA2"] = df["EirGridDemandFc_IDA2"] - df["EirGridDemandFc_IDA3"]

    spread_cols = [c for c in df.columns if c.startswith("spr_")]
    lag_base = spread_cols + [
        c
        for c in [
            "PriceDAM",
            "PriceIDA1",
            "PriceIDA2",
            "PriceIDA3",
            "NetDemand_DAM",
            "NetDemand_IDA1",
            "NetDemand_IDA2",
            "NetDemand_IDA3",
            "TotalPN",
        ]
        if c in df.columns
    ]
    for col in lag_base:
        s = df[col]
        df[f"{col}_lag1d"] = s.shift(48)
        df[f"{col}_lag2d"] = s.shift(96)
        df[f"{col}_lag7d"] = s.shift(48 * 7)
        df[f"{col}_ma7d"] = s.shift(48).rolling(48 * 7, min_periods=24).mean()
        df[f"{col}_std7d"] = s.shift(48).rolling(48 * 7, min_periods=24).std()
        df[f"{col}_ma14d"] = s.shift(48).rolling(48 * 14, min_periods=48).mean()

    for sfx in ["DAM", "IDA1", "IDA2", "IDA3"]:
        d, w, s = f"EirGridDemandFc_{sfx}", f"EirGridWindFc_{sfx}", f"ISEMSOLAR_{sfx}"
        if all(c in df.columns for c in [d, w, s]):
            df[f"ren_pct_{sfx}"] = (df[w] + df[s]) / (df[d] + 1)

    for gb in ["GB DAM Epex", "GB DAM N2EX", "GB DAM HH Epex"]:
        if gb in df.columns:
            safe = gb.replace(" ", "_")
            df[f"GB_vs_IRL_{safe}"] = df[gb] - df["PriceDAM"]

    return df


def build_xcols(df: pd.DataFrame, allow_uppers: Iterable[str], extra_always: Optional[Iterable[str]] = None) -> list[str]:
    extra_always = list(extra_always or [])
    time_feats = [
        "hod",
        "hour",
        "dow",
        "month",
        "is_weekend",
        "is_winter",
        "is_peak",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
    ]
    lag_sfx = ("_lag1d", "_lag2d", "_lag7d", "_ma7d", "_std7d", "_ma14d")
    utility_kw = ["wind_consensus", "wind_uncertainty", "wind_range", "wind_eirg_dev", "demand_disagreement", "ren_pct"]
    feats = set()
    for c in df.columns:
        if c in ALWAYS_EXCLUDE:
            continue
        if any(c.startswith(p) for p in TARGET_PREFIXES):
            continue
        if c in time_feats or c in extra_always:
            feats.add(c)
            continue
        if any(c.startswith(kw) for kw in utility_kw):
            x = c
            for kw in utility_kw:
                x = x.replace(kw, "")
            if any(ap in x.upper() for ap in allow_uppers):
                feats.add(c)
                continue
        if any(c.endswith(s) for s in lag_sfx):
            base = c
            for s in lag_sfx:
                base = base.replace(s, "")
            if any(ap in base.upper() for ap in allow_uppers):
                feats.add(c)
                continue
        if any(ap in c.upper() for ap in allow_uppers):
            feats.add(c)
    return sorted(c for c in feats if c in df.columns)


