"""End-to-end reproduction pipeline for local/private runs.

The private Energia dataset is not included in this repository. Place it under
``data/private/`` and run the CLI script to reproduce the original workflow.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from energy_arbitrage.data import load_market_data, clean_market_data, add_spreads_and_fundamentals
from energy_arbitrage.features import engineer_features
from energy_arbitrage.modeling import run_models
from energy_arbitrage.backtesting import build_leaderboard
from energy_arbitrage.visualization import save_figures


def run_reproduction_pipeline(data_path: str | Path, outputs_dir: str | Path = "outputs", figures_dir: str | Path = "reports/figures") -> tuple[pd.DataFrame, pd.DataFrame]:
    data = load_market_data(data_path)
    data1 = clean_market_data(data)
    df = add_spreads_and_fundamentals(data1)
    df = engineer_features(df)
    results = run_models(df)
    all_trading, leaderboard = build_leaderboard(results, df)

    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    all_trading.to_csv(outputs_dir / "trading_results_all_thresholds.csv", index=False)
    leaderboard.to_csv(outputs_dir / "leaderboard.csv", index=False)

    model_rows = []
    for r in results.values():
        model_rows.append(
            {
                "transition": r.tag,
                "spread": r.spr,
                "test_accuracy": r.acc,
                "roc_auc": r.auc,
                "cv_accuracy": r.cv_mean,
                "cv_std": r.cv_std,
                "reg_mae": r.mae,
                "reg_r2": r.r2,
                "test_size": len(r.Xte),
                "features": r.Xte.shape[1],
            }
        )
    pd.DataFrame(model_rows).to_csv(outputs_dir / "model_performance.csv", index=False)

    save_figures(results, df, leaderboard, figures_dir)
    return all_trading, leaderboard
