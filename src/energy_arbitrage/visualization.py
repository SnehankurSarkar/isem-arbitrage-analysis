"""Plotting functions for model and trading outputs.

These functions save figures to disk instead of displaying them interactively.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from energy_arbitrage.config import BEST_THRESH
from energy_arbitrage.modeling import ModelResult
from energy_arbitrage.backtesting import compute_pnl_series

def save_figures(results: Dict[str, ModelResult], df: pd.DataFrame, leaderboard: pd.DataFrame, figures_dir: str | Path) -> None:
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    cols = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, (name, res) in enumerate(results.items()):
        pnl_df = compute_pnl_series(res, df, BEST_THRESH)
        ax.plot(pnl_df["ts"], pnl_df["cum_pnl"], lw=1.8, color=cols[i % 10], label=f"{name}  final=€{pnl_df['cum_pnl'].iloc[-1]:,.0f}")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_title(f"Cumulative P&L — All Transitions (conf_threshold={BEST_THRESH}, magnitude sizing)", fontsize=11)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative P&L (€)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(figures_dir / "cumulative_pnl_all_transitions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    if "DAM→BM" in results:
        pnl_df = compute_pnl_series(results["DAM→BM"], df, BEST_THRESH)
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(pnl_df["ts"], pnl_df["drawdown"], color="red", lw=1.5)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title("Drawdown (€) — DAM → BM")
        ax.set_ylabel("Drawdown (€)")
        ax.set_xlabel("Date")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(figures_dir / "drawdown_DAM_BM.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    if not leaderboard.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        size_scale = np.sqrt(leaderboard["n_trades"])
        denom = size_scale.max() - size_scale.min()
        if denom == 0:
            size_scale = pd.Series(120, index=leaderboard.index)
        else:
            size_scale = 40 + 220 * (size_scale - size_scale.min()) / denom
        plot_df = leaderboard.sort_values("n_trades")
        for i, (_, row) in enumerate(plot_df.iterrows()):
            ax.scatter(row["test_accuracy"], row["eur_per_mwh"], s=size_scale.iloc[i], alpha=0.75, edgecolor="black", linewidth=0.6, color=cols[i % len(cols)], zorder=3)
            ax.annotate(row["transition"], (row["test_accuracy"], row["eur_per_mwh"]), fontsize=9, xytext=(6, 6), textcoords="offset points", weight="semibold")
        ax.axhline(0, color="red", lw=1.2, ls="--", alpha=0.7)
        ax.set_xlabel("Test Accuracy", fontsize=12, weight="semibold")
        ax.set_ylabel("EUR/MWh", fontsize=12, weight="semibold")
        ax.set_title("Model Performance: Accuracy vs Profitability\n(Bubble size ∝ number of trades)", fontsize=14, weight="bold", pad=15)
        ax.grid(alpha=0.25, linestyle="--", zorder=0)
        plt.tight_layout()
        fig.savefig(figures_dir / "accuracy_vs_profitability.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


