"""
GridPrice — Expanding Window Backtester

Provides ExpandingWindowBacktester for walk-forward validation across
the full historical dataset.  Imported by notebooks/evaluation.py
and src/gridprice/pipeline.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from gridprice.models import XGBPriceModel

logger = logging.getLogger(__name__)


# ── Config ───────────────────────────────────────────────────────────────────

@dataclass
class BacktestConfig:
    """Settings for ExpandingWindowBacktester."""
    train_min: int = 24 * 180      # ~6 months minimum training size
    step: int = 168                 # advance 1 week per step
    horizon: int = 24               # forecast horizon
    n_estimators: int = 50          # keep low for backtesting speed
    max_depth: int = 6
    random_state: int = 42


# ── Expanding-window backtester ────────────────────────────────────────────────

class ExpandingWindowBacktester:
    """
    Walk-forward / expanding-window backtester for the XGBPriceModel.

    Runs the full model pipeline on a series of train/test splits,
    advancing by ``step`` hours each iteration.  Produces per-iteration
    metrics, daily aggregated metrics, and a concatenated prediction
    DataFrame.

    Parameters
    ----------
    config : BacktestConfig
    model_factory : callable → XGBPriceModel
        Factory that produces a fresh model for each iteration.
    """

    def __init__(
        self,
        config: Optional[BacktestConfig] = None,
        model_factory: Optional[Callable[[], "XGBPriceModel"]] = None,
    ) -> None:
        self.config = config or BacktestConfig()
        self.model_factory = model_factory or self._default_factory

        self.results_: List[Dict[str, Any]] = []
        self.all_predictions_: Optional[pd.DataFrame] = None
        self.daily_metrics_: Optional[pd.DataFrame] = None
        self._fitted: bool = False

    # ── Default model factory ─────────────────────────────────────────

    @staticmethod
    def _default_factory() -> "XGBPriceModel":
        from gridprice.models import XGBPriceModel, XGBConfig
        cfg = XGBConfig(
            n_estimators=50,
            max_depth=6,
            min_train_size=24 * 180,
        )
        return XGBPriceModel(config=cfg, verbose=False)

    # ── Run ────────────────────────────────────────────────────────────

    def run(
        self,
        df: pd.DataFrame,
        target_col: str,
        show_progress: bool = True,
    ) -> "ExpandingWindowBacktester":
        """
        Execute the expanding-window backtest.

        Parameters
        ----------
        df : DataFrame with DatetimeIndex
            Full dataset (raw, pre-feature).
        target_col : str
            Name of the price column in ``df``.
        show_progress : bool

        Returns
        -------
        self
        """
        from gridprice.models import compute_metrics

        cfg = self.config
        results: List[Dict[str, Any]] = []
        all_preds: List[Dict[str, Any]] = []
        n = len(df)

        if show_progress:
            print(f"[Backtester] {n} rows, step={cfg.step}h, "
                  f"horizon={cfg.horizon}h, n_estimators={cfg.n_estimators}")

        i = 0
        while True:
            train_end = cfg.train_min + i * cfg.step
            if train_end + cfg.horizon >= n:
                break

            train_df = df.iloc[:train_end]
            test_df = df.iloc[train_end:train_end + cfg.horizon]

            if len(train_df) < cfg.train_min or len(test_df) < cfg.horizon:
                i += 1
                continue

            model = self.model_factory()
            model.config.n_estimators = cfg.n_estimators
            model.config.max_depth = cfg.max_depth
            model.config.min_train_size = max(cfg.train_min - 24 * 7, 24 * 30)
            model.config.val_size = 168

            try:
                model.fit(train_df)
            except Exception as exc:
                logger.warning("Fit failed at step %d (%s). Skipping.", i, exc)
                i += 1
                continue

            # Single-step evaluation
            try:
                metrics = model.evaluate(test_df)
            except Exception:
                i += 1
                continue

            # Recursive forecast
            try:
                fc = model.forecast_24h(train_df, horizon=cfg.horizon)
                fc_ts = pd.Series(fc.values, index=test_df.index[:cfg.horizon])
            except Exception:
                fc_ts = pd.Series(dtype=float)

            results.append({
                "iteration": i,
                "train_end": train_df.index[-1],
                "test_start": test_df.index[0],
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                **metrics,
            })

            for h in range(min(cfg.horizon, len(fc_ts))):
                all_preds.append({
                    "iteration": i,
                    "timestamp": test_df.index[h],
                    "horizon": h + 1,
                    "y_true": (test_df[target_col].iloc[h]
                               if target_col in test_df.columns else np.nan),
                    "y_pred": fc_ts.iloc[h] if h < len(fc_ts) else np.nan,
                    "train_end": train_df.index[-1],
                })

            if show_progress and i % 5 == 0:
                print(f"  iter {i:3d}: train_end={train_df.index[-1].date()} "
                      f"MAE={metrics.get('MAE', -1):.2f}  "
                      f"RMSE={metrics.get('RMSE', -1):.2f}")

            i += 1

        self.results_ = results
        self.all_predictions_ = pd.DataFrame(all_preds)
        self._compute_daily_metrics()
        self._fitted = True

        if show_progress:
            total = self._summary_df()
            print(f"\n[Backtester] Done — {len(results)} iterations.")
            print(total[["MAE", "RMSE", "MAPE"]].describe().to_string())

        return self

    def _compute_daily_metrics(self) -> None:
        """Aggregate per-iteration metrics into daily averages."""
        if not self.results_:
            return
        df = pd.DataFrame(self.results_)
        df["date"] = pd.to_datetime(df["test_start"]).dt.date
        daily = df.groupby("date")[["MAE", "RMSE", "MAPE", "sMAPE"]].mean()
        self.daily_metrics_ = daily.reset_index()

    # ── Results ────────────────────────────────────────────────────────

    def _summary_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.results_)

    @property
    def summary(self) -> pd.DataFrame:
        """Per-iteration metrics as a DataFrame."""
        if not self._fitted:
            raise ValueError("Run backtest first.")
        return self._summary_df()

    @property
    def predictions(self) -> pd.DataFrame:
        """Long-format DataFrame of all horizon-level predictions."""
        if not self._fitted:
            raise ValueError("Run backtest first.")
        return self.all_predictions_

    @property
    def daily_metrics(self) -> pd.DataFrame:
        """Per-date averaged metrics."""
        if not self._fitted:
            raise ValueError("Run backtest first.")
        return self.daily_metrics_

    def per_hour_error(self) -> pd.DataFrame:
        """Mean MAE / RMSE / MAPE broken down by hour-of-day (0–23)."""
        if not self._fitted:
            raise ValueError("Run backtest first.")
        df = self.all_predictions_.copy()
        df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
        df["error"] = df["y_true"] - df["y_pred"]
        df["abs_error"] = df["error"].abs()
        return (
            df.groupby("hour")[["error", "abs_error"]]
            .agg(error_mean=("error", "mean"),
                 MAE=("abs_error", "mean"),
                 count=("error", "count"))
            .reset_index()
        )

    def per_dow_error(self) -> pd.DataFrame:
        """Mean MAE / RMSE broken down by day-of-week."""
        if not self._fitted:
            raise ValueError("Run backtest first.")
        df = self.all_predictions_.copy()
        df["dow"] = pd.to_datetime(df["timestamp"]).dt.dayofweek
        df["dow_name"] = df["dow"].map(
            {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu",
             4: "Fri", 5: "Sat", 6: "Sun"}
        )
        df["error"] = df["y_true"] - df["y_pred"]
        df["abs_error"] = df["error"].abs()
        return (
            df.groupby(["dow", "dow_name"])[["error", "abs_error"]]
            .agg(error_mean=("error", "mean"), MAE=("abs_error", "mean"),
                 count=("error", "count"))
            .reset_index()
        )

    def per_horizon_error(self) -> pd.DataFrame:
        """Mean absolute error as a function of forecast horizon (1–24 h)."""
        if not self._fitted:
            raise ValueError("Run backtest first.")
        df = self.all_predictions_.copy()
        df["error"] = df["y_true"] - df["y_pred"]
        df["abs_error"] = df["error"].abs()
        return (
            df.groupby("horizon")[["error", "abs_error"]]
            .agg(error_mean=("error", "mean"),
                 MAE=("abs_error", "mean"),
                 count=("error", "count"))
            .reset_index()
        )

    # ── Plots ─────────────────────────────────────────────────────────

    def plot_per_horizon(self, save_path: Optional[Path] = None) -> None:
        """Line plot of MAE vs forecast horizon (1–24 h)."""
        import matplotlib.pyplot as plt
        plt.rcParams.update({"figure.dpi": 120, "font.size": 9})
        horizon_df = self.per_horizon_error()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(horizon_df["horizon"], horizon_df["MAE"],
                marker="o", linewidth=2, label="MAE")
        ax.fill_between(horizon_df["horizon"], 0, horizon_df["MAE"], alpha=0.15)
        ax.set_xlabel("Forecast Horizon (hours)")
        ax.set_ylabel("MAE (EUR/MWh)")
        ax.set_title("Forecast Error vs Horizon")
        ax.set_xticks(range(1, 25))
        ax.grid(True, alpha=0.3)
        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
            print(f"[Backtester] Saved horizon plot → {save_path}")
        plt.show()

    def plot_per_hour(self, save_path: Optional[Path] = None) -> None:
        """Bar chart of MAE per hour-of-day."""
        import matplotlib.pyplot as plt
        plt.rcParams.update({"figure.dpi": 120, "font.size": 9})
        hour_df = self.per_hour_error()
        fig, ax = plt.subplots(figsize=(9, 4))
        colors = plt.cm.viridis(
            (hour_df["hour"] - hour_df["hour"].min())
            / max(hour_df["hour"].max() - hour_df["hour"].min(), 1)
        )
        ax.bar(hour_df["hour"], hour_df["MAE"], color=colors, edgecolor="white")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("MAE (EUR/MWh)")
        ax.set_title("Mean Absolute Error by Hour of Day")
        ax.set_xticks(range(24))
        ax.grid(True, axis="y", alpha=0.3)
        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
            print(f"[Backtester] Saved per-hour plot → {save_path}")
        plt.show()

    def plot_error_distribution(
        self,
        save_path: Optional[Path] = None,
        model_name: str = "XGBoost",
    ) -> None:
        """Histogram + boxplot of forecast errors."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({"figure.dpi": 120, "font.size": 9})

        df = self.all_predictions_.copy()
        df["error"] = df["y_true"] - df["y_pred"]
        errors = df["error"].dropna()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        sns.histplot(errors, kde=True, ax=ax1, color="steelblue", bins=50)
        ax1.axvline(0, color="red", linewidth=1, linestyle="--")
        ax1.set_title(f"{model_name} Error Distribution")
        ax1.set_xlabel("Forecast Error (EUR/MWh)")
        ax1.set_ylabel("Count")

        df["horizon_bin"] = pd.cut(
            df["horizon"],
            bins=[0, 6, 12, 18, 24],
            labels=["1–6h", "7–12h", "13–18h", "19–24h"]
        )
        sns.boxplot(data=df, x="horizon_bin", y="error", ax=ax2,
                    palette="viridis", showfliers=False)
        ax2.axhline(0, color="red", linewidth=1, linestyle="--")
        ax2.set_title("Error by Horizon Band")
        ax2.set_xlabel("Horizon Band")
        ax2.set_ylabel("Error (EUR/MWh)")

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
            print(f"[Backtester] Saved error distribution → {save_path}")
        plt.show()

    def plot_forecast_sample(
        self,
        model: Any,
        df_history: pd.DataFrame,
        df_future: pd.DataFrame,
        price_col: str,
        n_days: int = 3,
        save_path: Optional[Path] = None,
    ) -> None:
        """Overlay of recursive 24 h forecast vs actuals for a sample window."""
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        plt.rcParams.update({"figure.dpi": 120, "font.size": 9})

        horizon = 24 * n_days
        fc = model.forecast_24h(df_history, horizon=horizon)
        actual = df_future[price_col].iloc[:horizon]

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(actual.index, actual.values, "b-", linewidth=1.5,
                label="Actual", marker=".", markersize=3)
        ax.plot(fc.index, fc.values, "r--", linewidth=1.5,
                label="Forecast (recursive)", marker="x", markersize=3)
        ax.fill_between(actual.index, actual.values,
                         fc.reindex(actual.index).values,
                         alpha=0.15, color="gray", label="Error band")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price (EUR/MWh)")
        ax.set_title(f"Recursive {n_days}-Day Forecast vs Actual")
        ax.legend()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
        ax.xaxis.set_major_locator(mdates.DayLocator())
        plt.xticks(rotation=30)
        ax.grid(True, alpha=0.3)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
            print(f"[Backtester] Saved forecast sample → {save_path}")
        plt.show()

    def plot_feature_importance(
        self,
        model: Any,
        top_n: int = 20,
        save_path: Optional[Path] = None,
    ) -> None:
        """Bar chart of XGBoost gain-based feature importance."""
        import matplotlib.pyplot as plt
        plt.rcParams.update({"figure.dpi": 120, "font.size": 9})

        fi = model.feature_importance_df().head(top_n)
        fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
        colors = plt.cm.Blues(
            np.linspace(0.9, 0.4, len(fi))
        )
        ax.barh(fi["feature"][::-1], fi["importance"][::-1], color=colors[::-1])
        ax.set_xlabel("Importance (gain)")
        ax.set_title(f"Top {top_n} Feature Importance")
        ax.grid(True, axis="x", alpha=0.3)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
            print(f"[Backtester] Saved feature importance → {save_path}")
        plt.show()

    def plot_daily_metrics(self, save_path: Optional[Path] = None) -> None:
        """Time-series of daily MAE / RMSE."""
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import seaborn as sns
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({"figure.dpi": 120, "font.size": 9})

        dm = self.daily_metrics.copy()
        dm["date"] = pd.to_datetime(dm["date"])

        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        for ax, col in zip(axes, ["MAE", "RMSE"]):
            ax.plot(dm["date"], dm[col], marker=".", linewidth=1)
            rm = dm[col].rolling(7, min_periods=1).mean()
            ax.plot(dm["date"], rm, "r--", linewidth=1.5, label="7-day avg")
            ax.set_ylabel(f"{col} (EUR/MWh)")
            ax.set_title(f"Daily {col}")
            ax.grid(True, alpha=0.3)
            ax.legend()
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
            print(f"[Backtester] Saved daily metrics → {save_path}")
        plt.show()

    # ── Export ─────────────────────────────────────────────────────────

    def export_metrics_csv(self, output_dir: Path = Path("reports/metrics")) -> None:
        """Save per-iteration, daily, per-hour, per-horizon, per-dow metrics + predictions as CSV."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.summary.to_csv(output_dir / "backtest_iterations.csv", index=False)
        self.daily_metrics.to_csv(output_dir / "backtest_daily.csv", index=False)
        self.per_hour_error().to_csv(output_dir / "backtest_per_hour.csv", index=False)
        self.per_horizon_error().to_csv(output_dir / "backtest_per_horizon.csv", index=False)
        self.per_dow_error().to_csv(output_dir / "backtest_per_dow.csv", index=False)
        self.all_predictions_.to_csv(output_dir / "backtest_predictions.csv", index=False)

        print(f"[Backtester] Exported CSV metrics to {output_dir}/")
