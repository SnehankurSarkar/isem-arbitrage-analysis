"""Trading strategy and P&L evaluation logic.

The confidence filtering and magnitude-proportional sizing match the submitted notebook.
"""
from __future__ import annotations

from typing import Dict
import numpy as np
import pandas as pd

from energy_arbitrage.config import BEST_THRESH, CONF_THRESHOLDS
from energy_arbitrage.modeling import ModelResult

def evaluate_pnl(res: ModelResult, df: pd.DataFrame, conf_thresholds=CONF_THRESHOLDS, max_vol: float = 10.0) -> pd.DataFrame:
    spread_true = df.loc[res.oi_te, res.spr].to_numpy(dtype=float)
    Xte = res.Xte

    proba = res.clf.predict_proba(Xte)
    up_idx = list(res.clf.classes_).index(1)
    p_up = proba[:, up_idx]
    conf = np.abs(p_up - 0.5)
    pred_m = res.reg.predict(Xte)
    p25 = max(np.percentile(np.abs(pred_m), 25), 0.5)
    vol_raw = np.clip(np.abs(pred_m) / p25, 0.1, max_vol)

    rows = []
    for thresh in conf_thresholds:
        sign_p = np.sign(p_up - 0.5).copy()
        sign_p[conf <= thresh] = 0
        traded = sign_p != 0
        correct = (sign_p == np.sign(spread_true)) & traded
        pnl_mwh = np.where(correct, np.abs(spread_true), -np.abs(spread_true))
        pnl_eur = np.where(traded, pnl_mwh * vol_raw, 0.0)
        vol_exec = np.where(traded, vol_raw, 0.0)
        t_vol = vol_exec.sum()
        t_pnl = pnl_eur.sum()
        epm = t_pnl / t_vol if t_vol > 0 else np.nan
        hr = correct[traded].mean() if traded.any() else np.nan
        s_ratio = pnl_eur[traded].mean() / (pnl_eur[traded].std() + 1e-9) if traded.sum() > 1 else np.nan
        rows.append(
            {
                "transition": res.tag,
                "conf_thresh": thresh,
                "n_trades": int(traded.sum()),
                "hit_rate": round(hr, 4),
                "eur_per_mwh": round(epm, 4),
                "total_pnl_eur": round(t_pnl, 2),
                "sharpe_proxy": round(s_ratio, 4),
            }
        )
    return pd.DataFrame(rows)


def build_leaderboard(results: Dict[str, ModelResult], df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_trading = pd.concat([evaluate_pnl(r, df) for r in results.values()], ignore_index=True)

    best = all_trading.sort_values("eur_per_mwh", ascending=False).drop_duplicates("transition").reset_index(drop=True)
    metrics = pd.DataFrame(
        [
            {
                "transition": r.tag,
                "test_accuracy": round(r.acc, 4),
                "roc_auc": round(r.auc, 4),
                "cv_accuracy": round(r.cv_mean, 4),
                "cv_std": round(r.cv_std, 4),
                "reg_r2": round(r.r2, 4),
                "reg_mae": round(r.mae, 4),
            }
            for r in results.values()
        ]
    )
    leaderboard = best.merge(metrics, on="transition")
    leaderboard = leaderboard.sort_values("eur_per_mwh", ascending=False).reset_index(drop=True)
    return all_trading, leaderboard


def compute_pnl_series(res: ModelResult, df: pd.DataFrame, conf_threshold: float = BEST_THRESH) -> pd.DataFrame:
    spread_true = df.loc[res.oi_te, res.spr].to_numpy(dtype=float)
    proba = res.clf.predict_proba(res.Xte)
    up_i = list(res.clf.classes_).index(1)
    p_up = proba[:, up_i]
    conf = np.abs(p_up - 0.5)
    pred_m = res.reg.predict(res.Xte)
    p25 = max(np.percentile(np.abs(pred_m), 25), 0.5)
    vol = np.clip(np.abs(pred_m) / p25, 0.1, 10.0)
    sign_p = np.sign(p_up - 0.5)
    sign_p[conf <= conf_threshold] = 0
    correct = (sign_p == np.sign(spread_true)) & (sign_p != 0)
    pnl = np.where(sign_p != 0, np.where(correct, np.abs(spread_true), -np.abs(spread_true)) * vol, 0)
    out = df.loc[res.oi_te, ["ts"]].assign(pnl=pnl).sort_values("ts").reset_index(drop=True)
    out["cum_pnl"] = out["pnl"].cumsum()
    out["cum_max"] = out["cum_pnl"].cummax()
    out["drawdown"] = out["cum_pnl"] - out["cum_max"]
    return out


