"""Modelling logic for direction classification and spread-magnitude regression.

LightGBM is the primary model exactly as in the submitted notebook; gradient boosting is retained only as a fallback for environments without LightGBM.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, roc_auc_score, r2_score
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit

from energy_arbitrage.config import LAG_BURNIN, RANDOM_STATE, TRANSITIONS
from energy_arbitrage.features import build_xcols

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:  # pragma: no cover - fallback exists for environments without LightGBM
    lgb = None
    HAS_LGB = False


@dataclass
class ModelResult:
    tag: str
    spr: str
    clf: Pipeline
    reg: Pipeline
    Xte: pd.DataFrame
    yc_te: pd.Series
    yr_te: pd.Series
    oi_te: np.ndarray
    acc: float
    auc: float
    mae: float
    r2: float
    cv_mean: float
    cv_std: float
    report: str


def make_pre(X: pd.DataFrame) -> ColumnTransformer:
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("scl", StandardScaler())])
    return ColumnTransformer([("n", pipe, list(X.columns))], remainder="drop")


def make_clf():
    if HAS_LGB:
        return lgb.LGBMClassifier(
            n_estimators=800,
            learning_rate=0.03,
            max_depth=6,
            num_leaves=63,
            min_child_samples=30,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    return GradientBoostingClassifier(
        n_estimators=500,
        learning_rate=0.04,
        max_depth=5,
        min_samples_leaf=25,
        subsample=0.8,
        random_state=42,
    )


def make_reg():
    if HAS_LGB:
        return lgb.LGBMRegressor(
            n_estimators=800,
            learning_rate=0.03,
            max_depth=6,
            num_leaves=63,
            min_child_samples=30,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    return GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.04,
        max_depth=5,
        min_samples_leaf=25,
        subsample=0.8,
        random_state=42,
    )


def build_ds(df: pd.DataFrame, spr_col: str, xcols: list[str], win_expr: Optional[str] = None):
    mask = df.index >= LAG_BURNIN
    if win_expr:
        mask &= df.eval(win_expr)
    mask &= df[spr_col].notna()
    mask &= np.sign(df[spr_col]) != 0
    d = df.loc[mask, xcols].copy().reset_index(drop=True)
    yc = np.sign(df.loc[mask, spr_col]).astype(int).reset_index(drop=True)
    yr = df.loc[mask, spr_col].astype(float).reset_index(drop=True)
    orig = df.index[mask].to_numpy()
    return d, yc, yr, orig


def temporal_split(d: pd.DataFrame, yc: pd.Series, yr: pd.Series, orig: np.ndarray, frac: float = 0.20):
    n = len(d)
    cut = int(n * (1 - frac))
    return (
        d.iloc[:cut].copy(),
        d.iloc[cut:].copy(),
        yc.iloc[:cut].copy(),
        yc.iloc[cut:].copy(),
        yr.iloc[:cut].copy(),
        yr.iloc[cut:].copy(),
        orig[:cut],
        orig[cut:],
    )


def run_models(df: pd.DataFrame) -> Dict[str, ModelResult]:
    results: Dict[str, ModelResult] = {}

    for name, (spr, allow, extra, win) in TRANSITIONS.items():
        if spr not in df.columns or df[spr].notna().sum() < 500:
            continue

        xcols = build_xcols(df, [a.upper() for a in allow], extra)
        xcols = [c for c in xcols if c in df.columns]
        d, yc, yr, orig = build_ds(df, spr, xcols, win)
        Xtr, Xte, yc_tr, yc_te, yr_tr, yr_te, oi_tr, oi_te = temporal_split(d, yc, yr, orig)

        clf = Pipeline([("pre", make_pre(Xtr)), ("m", make_clf())])
        clf.fit(Xtr, yc_tr)
        yhat = clf.predict(Xte)
        acc = accuracy_score(yc_te, yhat)
        try:
            proba = clf.predict_proba(Xte)
            up_idx = list(clf.classes_).index(1)
            auc = roc_auc_score((yc_te == 1).astype(int), proba[:, up_idx])
        except Exception:
            auc = float("nan")
        report = classification_report(yc_te, yhat, digits=4)

        reg = Pipeline([("pre", make_pre(Xtr)), ("m", make_reg())])
        reg.fit(Xtr, yr_tr)
        yr_pred = reg.predict(Xte)
        mae = mean_absolute_error(yr_te, yr_pred)
        r2 = r2_score(yr_te, yr_pred)

        tscv = TimeSeriesSplit(n_splits=5)
        cv_accs = []
        for tr_i, te_i in tscv.split(d):
            p = Pipeline([("pre", make_pre(d.iloc[tr_i])), ("m", make_clf())])
            p.fit(d.iloc[tr_i], yc.iloc[tr_i])
            cv_accs.append(accuracy_score(yc.iloc[te_i], p.predict(d.iloc[te_i])))

        results[name] = ModelResult(
            tag=name,
            spr=spr,
            clf=clf,
            reg=reg,
            Xte=Xte,
            yc_te=yc_te,
            yr_te=yr_te,
            oi_te=oi_te,
            acc=acc,
            auc=auc,
            mae=mae,
            r2=r2,
            cv_mean=float(np.mean(cv_accs)),
            cv_std=float(np.std(cv_accs)),
            report=report,
        )

    return results


