"""
GridPrice — Evaluation & Backtesting Module

Provides
--------
- ExpandingWindowBacktester : walk-forward validation across the full dataset
- HourlyErrorAnalysis        : per-hour-of-day and per-day-of-week error breakdown
- HorizonErrorAnalysis      : error as a function of forecast horizon (1–24 h)
- ErrorDistributionPlot      : histogram + boxplot of forecast errors
- ForecastPlot               : 24 h overlay of forecast vs actual
- FeatureImportancePlot      : bar chart of XGBoost gain-based importance
- DailyMetricsExport        : save per-day MAE/RMSE/MAPE to CSV

CLI
---
``python notebooks/evaluation.py`` runs the full backtest on synthetic data
and saves all plots to ``reports/figures/`` and CSVs to ``reports/metrics/``.

Dependencies: matplotlib, seaborn (installed in .venv)
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Plot helpers (lazy) ─────────────────────────────────────────────────────

def _plt():
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    plt.rcParams.update({
        "figure.dpi": 120,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    })
    return plt, mdates


def _seaborn():
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    return sns


# Import backtester from the canonical location
from gridprice.backtest import BacktestConfig, ExpandingWindowBacktester

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="GridPrice Evaluation")
    parser.add_argument("--data-end", default="2024-06-01",
                        help="End date for synthetic data (YYYY-MM-DD)")
    parser.add_argument("--train-min", type=int, default=24 * 180,
                        help="Minimum training size in hours")
    parser.add_argument("--step", type=int, default=168,
                        help="Step size in hours between iterations")
    parser.add_argument("--n-estimators", type=int, default=50,
                        help="XGBoost n_estimators per iteration")
    parser.add_argument("--out", default="reports",
                        help="Output directory for reports")
    args = parser.parse_args()

    # ── Generate synthetic data ────────────────────────────────────────
    from datetime import date
    end_date = date.fromisoformat(args.data_end)
    start_date = date(end_date.year - 2, end_date.month, end_date.day)

    print(f"[GridPrice] Generating synthetic data: {start_date} → {end_date}")
    from gridprice.synthetic_data import SyntheticDataGenerator
    gen = SyntheticDataGenerator(bidding_zone="GR", seed=42, end_date=end_date)
    df = gen.generate()

    split = int(len(df) * 0.8)
    df_train_raw = df.iloc[:split]
    df_test_raw = df.iloc[split:]
    price_col = "price_gr"

    out_dir = Path(args.out)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── Train final model on 80 % ─────────────────────────────────────
    print("\n[GridPrice] Training final model on 80 % split …")
    from gridprice.models import XGBPriceModel, XGBConfig

    final_cfg = XGBConfig(
        n_estimators=args.n_estimators,
        max_depth=6,
        min_train_size=args.train_min,
    )
    model = XGBPriceModel(config=final_cfg, verbose=True)
    model.fit(df_train_raw)

    # ── Backtest on 20 % ─────────────────────────────────────────────
    print("\n[GridPrice] Running expanding-window backtest …")
    backtest_cfg = BacktestConfig(
        train_min=args.train_min,
        step=args.step,
        n_estimators=args.n_estimators,
        max_depth=6,
    )

    def factory():
        cfg = XGBConfig(n_estimators=args.n_estimators, max_depth=6,
                         min_train_size=args.train_min)
        return XGBPriceModel(config=cfg, verbose=False)

    bt = ExpandingWindowBacktester(config=backtest_cfg, model_factory=factory)
    bt.run(df, target_col=price_col, show_progress=True)

    # ── Plots ─────────────────────────────────────────────────────────
    print("\n[GridPrice] Generating plots …")
    bt.plot_per_horizon(fig_dir / "error_vs_horizon.png")
    bt.plot_per_hour(fig_dir / "error_per_hour.png")
    bt.plot_error_distribution(fig_dir / "error_distribution.png")
    bt.plot_daily_metrics(fig_dir / "daily_metrics.png")

    # Forecast sample
    bt.plot_forecast_sample(model, df_train_raw.iloc[-500:], df_test_raw,
                            price_col, n_days=3,
                            save_path=fig_dir / "forecast_sample_3d.png")

    # Feature importance
    bt.plot_feature_importance(model, top_n=20,
                               save_path=fig_dir / "feature_importance.png")

    # ── Export CSVs ──────────────────────────────────────────────────
    bt.export_metrics_csv(out_dir / "metrics")

    # ── Single-step vs recursive comparison ──────────────────────────
    print("\n[GridPrice] Single-step vs recursive evaluation …")
    single_metrics = model.evaluate(df_test_raw.iloc[:500])
    print(f"  Single-step:  MAE={single_metrics['MAE']:.2f}  "
          f"RMSE={single_metrics['RMSE']:.2f}  MAPE={single_metrics['MAPE']:.1f}%")

    # ── Summary ───────────────────────────────────────────────────────
    summary = bt.summary
    print("\n[GridPrice] Backtest summary:")
    print(summary[["MAE", "RMSE", "MAPE"]].describe().to_string())

    horizon_df = bt.per_horizon_error()
    print(f"\n  Horizon 1 h  MAE: {horizon_df[horizon_df['horizon']==1]['MAE'].values[0]:.2f}")
    print(f"  Horizon 24 h MAE: {horizon_df[horizon_df['horizon']==24]['MAE'].values[0]:.2f}")

    print(f"\n[GridPrice] Reports saved to {out_dir}/")
    print("  figures/   → PNG plots")
    print("  metrics/   → CSV metrics")
