#!/usr/bin/env python
"""Create a synthetic I-SEM-like sample for structure/smoke testing only.

This is not Energia data and must not be used to reproduce the submitted numerical results.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def make_sample(n_rows: int = 2500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2025-01-01", periods=n_rows, freq="30min")
    hod = np.asarray(ts.hour + ts.minute / 60, dtype=float)
    daily = np.asarray(np.sin(2 * np.pi * hod / 24), dtype=float)
    evening = np.asarray(np.exp(-((hod - 18) / 3) ** 2), dtype=float)
    wind_base = 1500 + 550 * np.sin(2 * np.pi * np.arange(n_rows) / (48 * 7)) + rng.normal(0, 180, n_rows)
    wind_base = np.maximum(wind_base, 50)
    solar = np.asarray(np.maximum(0, 550 * np.sin(np.pi * (hod - 6) / 14)), dtype=float)
    demand = 4200 + 500 * daily + 650 * evening + rng.normal(0, 130, n_rows)
    net_demand = demand - wind_base - solar
    dam = 90 + 0.028 * net_demand + 8 * evening + rng.normal(0, 12, n_rows)
    ida1 = dam + rng.normal(0, 18, n_rows) - 0.006 * (wind_base - wind_base.mean())
    ida2 = ida1 + rng.normal(-4, 16, n_rows)
    ida3 = ida2 + rng.normal(-5, 15, n_rows)
    imbalance = dam + rng.normal(0, 34, n_rows) + 0.012 * (net_demand - net_demand.mean())

    ida2_window = (hod >= 11) & (hod < 23)
    ida3_window = (hod >= 17) & (hod < 23)
    ida2 = np.where(ida2_window, ida2, np.nan)
    ida3 = np.where(ida3_window, ida3, np.nan)

    df = pd.DataFrame({
        "TradeDate": ts.date.astype(str),
        "Time": ts.strftime("%H:%M:%S"),
        "StartDateTime": ts.astype(str),
        "PriceDAM": dam,
        "PriceIDA1": ida1,
        "PriceIDA2": ida2,
        "PriceIDA3": ida3,
        "PriceImbalance": imbalance,
        "EirgridActualDemand": demand + rng.normal(0, 45, n_rows),
        "EirgridActualWind": wind_base + rng.normal(0, 45, n_rows),
        "NIV_Actual": rng.normal(0, 180, n_rows),
        "TotalPN": demand + rng.normal(0, 80, n_rows),
        "ActualMeterData": demand + rng.normal(0, 75, n_rows),
        "Aggregated Forecast": demand + rng.normal(0, 100, n_rows),
        "AggregatedForecast": demand + rng.normal(0, 100, n_rows),
        "PredictedICFlow": rng.normal(0, 220, n_rows),
        "PumpStorage": rng.normal(30, 18, n_rows),
        "GB DAM Epex": dam + rng.normal(3, 10, n_rows),
        "GB DAM N2EX": dam + rng.normal(4, 11, n_rows),
        "GB DAM HH Epex": dam + rng.normal(2, 12, n_rows),
        "PostIDA1Flow": rng.normal(0, 180, n_rows),
        "PostIDA2Flow": rng.normal(0, 160, n_rows),
        "LoadForecast_MW": demand + rng.normal(0, 90, n_rows),
    })

    for sfx, bias in [("DAM", 0), ("IDA1", -15), ("IDA2", -25), ("IDA3", -30)]:
        df[f"EirGridDemandFc_{sfx}"] = demand + bias + rng.normal(0, 70, n_rows)
        df[f"Meteo_ISEMDEMAND_{sfx}"] = demand + bias + rng.normal(0, 95, n_rows)
        df[f"EirGridWindFc_{sfx}"] = wind_base + rng.normal(0, 100, n_rows)
        df[f"Meteo_ISEMWIND_{sfx}"] = wind_base + rng.normal(0, 130, n_rows)
        df[f"EmSys_C_ISEMWIND_{sfx}"] = wind_base + rng.normal(0, 115, n_rows)
        df[f"EmSys_U_ISEMWIND_{sfx}"] = wind_base + rng.normal(0, 120, n_rows)
        df[f"ISEMSOLAR_{sfx}"] = solar + rng.normal(0, 25, n_rows)
        df.loc[df[f"ISEMSOLAR_{sfx}"] < 0, f"ISEMSOLAR_{sfx}"] = 0

    # Reflect structural auction availability in sample forecasts/solar where useful.
    for c in ["EirGridDemandFc_IDA2", "EirGridWindFc_IDA2", "ISEMSOLAR_IDA2"]:
        df.loc[~ida2_window, c] = np.nan
    for c in ["EirGridDemandFc_IDA3", "EirGridWindFc_IDA3", "ISEMSOLAR_IDA3"]:
        df.loc[~ida3_window, c] = np.nan

    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=2500)
    parser.add_argument("--output", default="data/sample/isem_synthetic_sample.csv")
    args = parser.parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    make_sample(args.rows).to_csv(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
