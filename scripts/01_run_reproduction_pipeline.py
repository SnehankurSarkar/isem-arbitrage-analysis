#!/usr/bin/env python
"""Run the faithful reproduction pipeline on a local market-data file.

Example:
    python scripts/01_run_reproduction_pipeline.py \
        --data data/private/MarketData_2022-2026.parquet \
        --outputs outputs \
        --figures reports/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from energy_arbitrage.pipeline import run_reproduction_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the exact I-SEM arbitrage reproduction pipeline.")
    parser.add_argument("--data", required=True, help="Path to MarketData_2022-2026.parquet or an equivalent CSV.")
    parser.add_argument("--outputs", default="outputs", help="Directory for CSV outputs.")
    parser.add_argument("--figures", default="reports/figures", help="Directory for generated figures.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_trading, leaderboard = run_reproduction_pipeline(
        data_path=Path(args.data),
        outputs_dir=Path(args.outputs),
        figures_dir=Path(args.figures),
    )
    print("\nTrading results at all confidence thresholds:")
    print(all_trading.to_string(index=False))
    print("\nFINAL LEADERBOARD (sorted by EUR/MWh profit):")
    print(leaderboard.to_string(index=False))


if __name__ == "__main__":
    main()
