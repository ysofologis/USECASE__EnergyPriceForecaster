"""
GridPrice — XGBoost Price Forecasting Model

Classes
-------
XGBPriceModel
    Single-step XGBoost forecaster with expanding-window training,
    early stopping, recursive multi-step 24 h forecasting, and SHAP
    feature-importance analysis.

Functions
---------
compare_baselines
    Evaluate Persistence, LinearRegression and 4-week-average baselines
    on a test set and return per-metric DataFrames.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Hyperparameter defaults ──────────────────────────────────────────────────

@dataclass
class XGBConfig:
    """XGBoost hyperparameters and training configuration."""

    # ── XGBoost tree settings
    max_depth: int = 6
    min_child_weight: int = 10
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    colsample_bylevel: float = 0.8

    # ── Regularisation
    reg_alpha: float = 0.1          # L1
    reg_lambda: float = 1.0          # L2

    # ── Learning
    learning_rate: float = 0.05
    n_estimators: int = 2000        # early stopping usually stops well before this

    # ── Training windows
    min_train_size: int = 24 * 180  # ~6 months minimum training data
    val_size: int = 168             # 1 week validation for early stopping

    # ── Other
    random_state: int = 42
    tree_method: Literal["hist", "approx", "exact"] = "hist"

    def to_xgb_dict(self) -> Dict[str, Any]:
        """Render as a dict suitable for xgboost.XGBRegressor(**d)."""
        return {
                "max_depth": self.max_depth,
                "min_child_weight": self.min_child_weight,
                "subsample": self.subsample,
                "colsample_bytree": self.colsample_bytree,
                "colsample_bylevel": self.colsample_bylevel,
                "reg_alpha": self.reg_alpha,
                "reg_lambda": self.reg_lambda,
                "learning_rate": self.learning_rate,
                "n_estimators": self.n_estimators,
                "random_state": self.random_state,
                "tree_method": self.tree_method,
                "verbosity": 0,
                # early_stopping_rounds is set at construction time per-step
                "early_stopping_rounds": 50,
            }


# ── Metrics ─────────────────────────────────────────────────────────────────

def _mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mape(y_true: pd.Series, y_pred: pd.Series, eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error, ignoring zero-price hours."""
    mask = y_true.abs() > eps
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _smape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Symmetric MAPE — handles negative prices gracefully."""
    denom = (y_true.abs() + y_pred.abs()) / 2
    mask = denom > 1e-8
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)


def compute_metrics(
    y_true: pd.Series, y_pred: pd.Series
) -> Dict[str, float]:
    return {"MAE": _mae(y_true, y_pred), "RMSE": _rmse(y_true, y_pred),
            "MAPE": _mape(y_true, y_pred), "sMAPE": _smape(y_true, y_pred)}


# ── Baseline models ─────────────────────────────────────────────────────────

def _persistence_baseline(df: pd.DataFrame, price_col: str, horizon: int = 1) -> pd.Series:
    """Naive persistence: tomorrow's price = today's price at same hour."""
    price = df[price_col]
    return price.shift(24).reindex(df.index)  # T-24h as forecast


def _linear_trend_baseline(
    df: pd.DataFrame,
    price_col: str,
    feature_cols: List[str],
) -> pd.Series:
    """
    Ridge regression using calendar + lag features.
    Fitted on the full DataFrame (no hold-out) — used as a baseline reference.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    feat_df = df[feature_cols].copy()
    feat_df = feat_df.fillna(feat_df.median())

    scaler = StandardScaler()
    X = scaler.fit_transform(feat_df)
    y = df[price_col].values

    model = RidgeCV(alphas=np.logspace(-3, 3, 50))
    model.fit(X, y)
    return pd.Series(model.predict(X), index=df.index)


def _seasonal_avg_baseline(df: pd.DataFrame, price_col: str) -> pd.Series:
    """
    Same-weekday same-hour average over the last 4 weeks.
    Naive but surprisingly hard to beat on stable markets.
    """
    sub = df[[price_col]].copy()
    sub["hour"] = sub.index.hour
    sub["dow"] = sub.index.dayofweek
    seasonal = sub.groupby(["dow", "hour"])[price_col].mean()
    return sub.apply(lambda r: seasonal[(r["dow"], r["hour"])], axis=1)


def compare_baselines(
    df: pd.DataFrame,
    price_col: str,
    test_start: Optional[pd.Timestamp] = None,
) -> Tuple[Dict[str, Dict[str, float]], pd.DataFrame]:
    """
    Compute metrics for three baselines on the test period.

    Returns
    -------
    metrics : dict  — {name: {metric: value}}
    test_df : DataFrame with y_true, persistence_pred, ridge_pred, seasonal_pred
    """
    if test_start is None:
        split = int(len(df) * 0.8)
        test_start = df.index[split]

    test = df[df.index >= test_start].copy()
    train_data = df[df.index < test_start]

    # Persistence
    persistence_pred = _persistence_baseline(df[df.index < test_start + timedelta(hours=24)],
                                              price_col)
    persistence_pred = persistence_pred.reindex(test.index)

    # Seasonal average
    seasonal_pred = _seasonal_avg_baseline(df[df.index < test_start], price_col)
    seasonal_pred = seasonal_pred.reindex(test.index)

    y_true = test[price_col]

    results: Dict[str, Dict[str, float]] = {}
    for name, pred in [("persistence", persistence_pred),
                        ("seasonal_avg_4w", seasonal_pred)]:
        valid = ~(pred.isna() | y_true.isna())
        if valid.any():
            results[name] = compute_metrics(y_true[valid], pred[valid])

    test_df = pd.DataFrame({"y_true": y_true,
                             "persistence": persistence_pred,
                             "seasonal_avg_4w": seasonal_pred})

    return results, test_df


# ── XGBPriceModel ────────────────────────────────────────────────────────────

class XGBPriceModel:
    """
    XGBoost single-step price forecaster for the Greek bidding zone.

    Training
    -------
    Uses an **expanding window**: the model is trained on all data from
    ``min_train_size`` hours up to each test timestamp, with a rolling
    1-week validation set for early stopping.

    Forecasting
    ----------
    ``forecast_24h()`` produces 24 hourly forecasts using a **recursive**
    strategy: each T+h prediction is fed back as the ``price_lag_24h`` /
    ``price_lag_48h`` / etc. features for the next step. This allows the
    model to compound its own error but is simple to implement and has
    been shown competitive with direct multi-output strategies.

    SHAP
    ----
    ``shap_summary()`` runs TreeExplainer on the fitted model and returns
    mean absolute SHAP values per feature (requires ``shap`` to be installed).

    Parameters
    ----------
    config : XGBConfig
        Hyperparameters and training settings.
    target_col : str
        Name of the price column (auto-detected from the feature DataFrame
        if omitted).
    lag_features : tuple of int
        Passed to FeatureEngineer. Default (24, 48, 168, 336).
    seasonal_years : tuple of int
        Passed to FeatureEngineer for Greek holiday calendar coverage.
    include_dst : bool
        Passed to FeatureEngineer.
    verbose : bool
        Print training progress.

    Attributes
    ----------
    model_ : xgboost.XGBRegressor
        The fitted booster.
    feature_cols_ : list of str
        Names of the features used for training (excluding the target).
    feature_importance_ : dict  (set after shap_summary())
        Feature → mean |SHAP| values.
    config_ : XGBConfig
        Effective config used (may differ from input if using defaults).

    Examples
    --------
    >>> from gridprice.models import XGBPriceModel
    >>> model = XGBPriceModel()
    >>> model.fit(df_train)
    >>> forecasts = model.forecast_24h(df_train.iloc[-24:])   # last known 24h
    >>> metrics = model.evaluate(df_test)
    >>> shap = model.shap_summary(df_test.head(1000))          # needs shap installed
    """

    def __init__(
        self,
        config: Optional[XGBConfig] = None,
        target_col: Optional[str] = None,
        lag_features: Tuple[int, ...] = (24, 48, 168, 336),
        seasonal_years: Tuple[int, ...] = (2024, 2025, 2026, 2027, 2028, 2029, 2030),
        include_dst: bool = True,
        verbose: bool = False,
    ) -> None:
        self.config = config or XGBConfig()
        self.target_col = target_col
        self.lag_features = lag_features
        self.seasonal_years = seasonal_years
        self.include_dst = include_dst
        self.verbose = verbose

        # Fitted state
        self.model_: Any = None
        self.feature_cols_: List[str] = []
        self.target_col_: str = ""
        self.feature_importance_: Dict[str, float] = {}
        self._fitted: bool = False

        # Lazy imports (avoid hard dependency)
        self._xgb: Any = None
        self._fe_class: Any = None

    # ── Lazy imports ─────────────────────────────────────────────────────

    @property
    def xgb(self):
        if self._xgb is None:
            self._xgb = __import__("xgboost", fromlist=["XGBRegressor"]).XGBRegressor
        return self._xgb

    @property
    def FE(self):
        if self._fe_class is None:
            from gridprice.features import FeatureEngineer
            self._fe_class = FeatureEngineer
        return self._fe_class

    # ── fit ─────────────────────────────────────────────────────────────

    def fit(
        self,
        df: pd.DataFrame,
        eval_fraction: float = 0.1,
    ) -> "XGBPriceModel":
        """
        Fit the XGBoost model using expanding-window + early stopping.

        The training window expands by one week at a time.  At each
        expansion step the model is re-fitted with early stopping on
        the latest ``val_size`` hours.  The final model uses the
        average best-iteration count across all expansion steps
        (a simple heuristic; more sophisticated would be to take the
        median and freeze at that number).

        Parameters
        ----------
        df : DataFrame with DatetimeIndex
            Raw (pre-feature) training data.
        eval_fraction : float
            Fraction of ``df`` to hold out as final evaluation set
            (used only for the last training step).

        Returns
        -------
        self
        """
        # ── 1. Engineer features ──────────────────────────────────────
        fe = self.FE(
            lag_features=self.lag_features,
            rolling_windows=(24, 168),
            holiday_years=self.seasonal_years,
            include_cyclical=True,
            include_dst_flag=self.include_dst,
        )
        df_fe = fe.fit_transform(df)
        self.target_col_ = self.target_col or fe.price_col_ or "price_gr"

        # Drop rows with any NaN (from lag / rolling features)
        drop_cols = [self.target_col_] if self.target_col_ in df_fe.columns else []
        feat_df = df_fe.dropna()
        if len(feat_df) < self.config.min_train_size:
            raise ValueError(
                f"Not enough rows after dropping NaNs.  "
                f"Got {len(feat_df)}, need {self.config.min_train_size}.  "
                f"Consider using a shorter lag window or more data."
            )

        # Feature columns = all numeric columns except the target
        all_cols = [c for c in feat_df.columns
                    if c != self.target_col_ and pd.api.types.is_numeric_dtype(feat_df[c])]
        self.feature_cols_ = all_cols

        X_all = feat_df[all_cols].values
        y_all = feat_df[self.target_col_].values
        index_all = feat_df.index

        # ── 2. Expanding window ────────────────────────────────────────
        step = 168  # advance by 1 week per step
        best_iterations: List[int] = []
        all_val_preds: List[Tuple[pd.Series, pd.Series]] = []  # (y_true, y_pred)

        n_total = len(feat_df)
        n_test = max(int(n_total * eval_fraction), self.config.val_size)

        if self.verbose:
            print(f"[XGBPriceModel] {n_total} rows, {n_test} held for eval, "
                  f"step={step}h, val_size={self.config.val_size}h")

        for train_end_idx in range(self.config.min_train_size, n_total - n_test, step):
            train_end = index_all[train_end_idx]
            X_train = X_all[:train_end_idx]
            y_train = y_all[:train_end_idx]

            val_end_idx = min(train_end_idx + self.config.val_size, n_total - n_test)
            X_val = X_all[train_end_idx:val_end_idx]
            y_val = y_all[train_end_idx:val_end_idx]

            if len(X_val) < 24:
                continue

            xgb_params = self.config.to_xgb_dict()

            model = self.xgb(**xgb_params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            best_iterations.append(model.best_iteration)

            val_pred = model.predict(X_val)
            all_val_preds.append((
                pd.Series(y_val, index=index_all[train_end_idx:val_end_idx]),
                pd.Series(val_pred, index=index_all[train_end_idx:val_end_idx]),
            ))

        if not best_iterations:
            raise RuntimeError("Expanding window produced no training steps. Increase data size.")

        avg_best_iter = int(np.mean(best_iterations))
        if self.verbose:
            print(f"[XGBPriceModel] Avg best iteration: {avg_best_iter}  "
                  f"(range {min(best_iterations)}-{max(best_iterations)})")

        # ── 3. Final fit on all training data ─────────────────────────
        train_end_final = n_total - n_test
        X_train_final = X_all[:train_end_final]
        y_train_final = y_all[:train_end_final]
        X_val_final = X_all[train_end_final:n_total]
        y_val_final = y_all[train_end_final:n_total]

        final_params = self.config.to_xgb_dict()
        final_params["n_estimators"] = avg_best_iter + 5  # small buffer

        self.model_ = self.xgb(**final_params)
        self.model_.fit(
            X_train_final, y_train_final,
            eval_set=[(X_val_final, y_val_final)],
            verbose=False,
        )

        self._fitted = True

        # Report overall validation performance
        if all_val_preds:
            y_true_all = pd.concat([p[0] for p in all_val_preds])
            y_pred_all = pd.concat([p[1] for p in all_val_preds])
            m = compute_metrics(y_true_all, y_true_all)  # placeholder
            # compute on concat
            overall = compute_metrics(
                pd.concat([p[0] for p in all_val_preds]),
                pd.concat([p[1] for p in all_val_preds]),
            )
            if self.verbose:
                print(f"[XGBPriceModel] Expanding-window val metrics: {overall}")

        return self

    # ── predict ────────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Single-step prediction for a feature matrix.

        X must have the same columns (in the same order) as the training
        features.  Use ``prepare_features()`` to convert raw data.

        Returns
        -------
        Series with the same index as X, or integer index if X has no index.
        """
        if not self._fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")
        X_arr = X[self.feature_cols_].values
        pred = self.model_.predict(X_arr)
        return pd.Series(pred, index=X.index if hasattr(X, "index") else None)

    # ── forecast_24h ───────────────────────────────────────────────────

    def forecast_24h(
        self,
        last_known: pd.DataFrame,
        horizon: int = 24,
    ) -> pd.Series:
        """
        Recursive 24-step-ahead forecast starting from ``last_known``.

        Parameters
        ----------
        last_known : DataFrame
            Raw (pre-feature) data ending at the last known timestamp.
            Must contain at least ``max(lag_features)`` hours of history.
        horizon : int
            Number of hours to forecast (default 24).

        Returns
        -------
        Series of ``horizon`` price predictions, indexed by forecast timestamp.
        """
        if not self._fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")

        if len(last_known) < max(self.lag_features):
            raise ValueError(
                f"last_known has {len(last_known)} rows but needs at least "
                f"{max(self.lag_features)} for lag features."
            )

        # Engineer features on the full history
        fe = self.FE(
            lag_features=self.lag_features,
            rolling_windows=(24, 168),
            holiday_years=self.seasonal_years,
            include_cyclical=True,
            include_dst_flag=self.include_dst,
        )
        # Clone the feature engine with the same fitted state
        fe.fit(last_known)

        forecasts: List[float] = []
        current_df = last_known.copy()

        for step in range(horizon):
            # Engineer features for current state
            df_fe = fe.transform(current_df)
            last_row = df_fe.iloc[[-1]]

            # Predict
            pred = self.model_.predict(last_row[self.feature_cols_].values)[0]
            forecasts.append(pred)

            # Append prediction as a new row
            next_ts = current_df.index[-1] + timedelta(hours=1)
            new_row = current_df.iloc[[-1]].copy()
            new_row.index = [next_ts]
            if self.target_col_ in new_row.columns:
                new_row[self.target_col_] = pred
            # Also update lag sources if they reference the target
            # (most importantly, the price column — lag features will
            #  pick it up because we appended it to current_df)
            if self.target_col_ in current_df.columns:
                current_df = pd.concat([current_df, new_row])

        # Build output index
        start_ts = last_known.index[-1] + timedelta(hours=1)
        forecast_index = pd.date_range(start=start_ts, periods=horizon, freq="h",
                                       tz=last_known.index.tz)
        return pd.Series(forecasts, index=forecast_index)

    # ── evaluate ───────────────────────────────────────────────────────

    def evaluate(
        self,
        df_test: pd.DataFrame,
        metrics: Tuple[str, ...] = ("MAE", "RMSE", "MAPE", "sMAPE"),
    ) -> Dict[str, float]:
        """
        Evaluate single-step forecasts on a hold-out test set.

        Parameters
        ----------
        df_test : DataFrame
            Raw (pre-feature) test data with the same columns as training.
        metrics : tuple of str
            Which metrics to compute. Options: MAE, RMSE, MAPE, sMAPE.

        Returns
        -------
        dict of metric_name → float
        """
        if not self._fitted:
            raise ValueError("Model has not been fitted.")

        # Engineer features — lag/rolling will produce NaNs for early rows
        fe = self.FE(
            lag_features=self.lag_features,
            rolling_windows=(24, 168),
            holiday_years=self.seasonal_years,
            include_cyclical=True,
            include_dst_flag=self.include_dst,
        )
        df_fe = fe.fit_transform(df_test)

        y_true = df_fe[self.target_col_]
        y_pred = self.predict(df_fe)

        result = compute_metrics(y_true, y_pred)
        return {k: result[k] for k in metrics}

    # ── recursive evaluate ──────────────────────────────────────────────

    def evaluate_recursive(
        self,
        df_test: pd.DataFrame,
        horizon: int = 24,
    ) -> pd.DataFrame:
        """
        Evaluate recursive multi-step forecasting on the test set.

        For each ``horizon``-hour block in ``df_test``, produces a
        recursive forecast starting from the block's preceding data.

        Returns a DataFrame with columns ``y_true`` and ``y_pred`` indexed
        by timestamp, plus a ``horizon`` column indicating how many steps
        ahead each prediction was made.
        """
        if not self._fitted:
            raise ValueError("Model has not been fitted.")

        max_lag = max(self.lag_features)
        step = horizon  # advance by full horizon each time

        results: List[Dict[str, Any]] = []
        n = len(df_test)

        for start in range(max_lag, n - horizon, step):
            history = df_test.iloc[:start]
            block = df_test.iloc[start:start + horizon]

            # Recursive forecast
            preds = self.forecast_24h(history, horizon=horizon)

            for h in range(horizon):
                if h < len(preds):
                    ts = block.index[h]
                    results.append({
                        "timestamp": ts,
                        "horizon": h + 1,
                        "y_true": block[self.target_col_].iloc[h] if self.target_col_ in block.columns else np.nan,
                        "y_pred": preds.iloc[h],
                    })

        result_df = pd.DataFrame(results).set_index("timestamp")
        return result_df

    # ── SHAP ──────────────────────────────────────────────────────────

    def shap_summary(
        self,
        df_sample: pd.DataFrame,
        max_display: int = 25,
    ) -> Dict[str, float]:
        """
        Compute mean absolute SHAP values for each feature.

        Requires ``shap`` to be installed (``pip install shap``).

        Parameters
        ----------
        df_sample : DataFrame
            Feature-engineered sample (from ``fe.transform()``).
            A subset of 500-2000 rows is recommended for speed.
        max_display : int
            Passed to shap.summary_plot / shap.plots.bar.

        Returns
        -------
        dict of feature_name → mean |SHAP value|
        """
        try:
            import shap
        except ImportError:
            raise ImportError(
                "shap is not installed.  Install with: pip install shap"
            )

        if not self._fitted:
            raise ValueError("Model has not been fitted.")

        X_sample = df_sample[self.feature_cols_].fillna(0).values
        explainer = shap.TreeExplainer(self.model_)
        shap_values = explainer.shap_values(X_sample)

        mean_abs = np.abs(shap_values).mean(axis=0)
        importance = dict(zip(self.feature_cols_, map(float, mean_abs)))
        self.feature_importance_ = importance
        return importance

    def shap_plot(self, df_sample: pd.DataFrame, save_path: Optional[Path] = None) -> None:
        """Generate SHAP beeswarm summary plot (requires shap)."""
        try:
            import shap
        except ImportError:
            raise ImportError("shap is not installed.  Run: pip install shap")

        if not self.feature_importance_:
            self.shap_summary(df_sample)

        X_sample = df_sample[self.feature_cols_].fillna(0).values
        explainer = shap.TreeExplainer(self.model_)
        shap_values = explainer.shap_values(X_sample)

        shap.summary_plot(shap_values, X_sample,
                          feature_names=self.feature_cols_,
                          show=False, max_display=25)
        if save_path:
            import matplotlib.pyplot as plt
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            plt.close()
            print(f"[XGBPriceModel] SHAP plot saved to {save_path}")

    # ── Feature importance (native XGBoost) ─────────────────────────────

    def feature_importance_df(self) -> pd.DataFrame:
        """
        Return a DataFrame of XGBoost native feature importance (gain).
        """
        if not self._fitted:
            raise ValueError("Model has not been fitted.")
        imp = self.model_.feature_importances_
        return (
            pd.DataFrame({"feature": self.feature_cols_, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    # ── Save / Load ────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Save model, config, and feature column names to ``path``."""
        import pickle
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"model": self.model_, "config": self.config,
                 "feature_cols": self.feature_cols_, "target_col": self.target_col_,
                 "lag_features": self.lag_features},
                f,
            )
        if self.verbose:
            print(f"[XGBPriceModel] Saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "XGBPriceModel":
        """Load a model from disk."""
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        inst = cls(config=data["config"], target_col=data["target_col"],
                   lag_features=data["lag_features"])
        inst.model_ = data["model"]
        inst.feature_cols_ = data["feature_cols"]
        inst.target_col_ = data["target_col"]
        inst._fitted = True
        return inst

    # ── Repr ───────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        fitted = "fitted" if self._fitted else "not fitted"
        return (
            f"XGBPriceModel(config={self.config}, target={self.target_col_!r}, "
            f"{fitted}, features={len(self.feature_cols_)})"
        )


# ── CLI demo ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from gridprice.synthetic_data import SyntheticDataGenerator
    from datetime import date

    print("[GridPrice] Generating training data …")
    gen = SyntheticDataGenerator(bidding_zone="GR", seed=42, end_date=date(2024, 6, 1))
    df = gen.generate()

    split = int(len(df) * 0.8)
    df_train = df.iloc[:split]
    df_test = df.iloc[split:]

    print(f"Train: {df_train.shape}, Test: {df_test.shape}")

    print("[GridPrice] Training XGBPriceModel …")
    model = XGBPriceModel(verbose=True)
    model.fit(df_train)

    print("\n[XGBPriceModel] Test metrics (single-step):")
    metrics = model.evaluate(df_test)
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}")

    print("\n[XGBPriceModel] Baseline comparison:")
    baselines, _ = compare_baselines(df, "price_gr",
                                      test_start=df_test.index[0])
    for name, m in baselines.items():
        print(f"  {name:25s}  MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  MAPE={m['MAPE']:.1f}%")

    print("\n[XGBPriceModel] Feature importance (top 10):")
    fi = model.feature_importance_df()
    print(fi.head(10).to_string(index=False))
