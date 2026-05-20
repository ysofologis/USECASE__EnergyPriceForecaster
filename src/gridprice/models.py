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
    """
    Hyperparameter settings for the XGBoost price forecasting model.

    Think of these as the "knobs" that control how the model learns.
    The defaults are sensible starting points — only change them if you
    have a reason and can measure whether the change helps.

    What each group of knobs does
    -----------------------------
    Tree settings (how complex each decision tree is):

    ``max_depth`` — How many decisions deep each tree can go.
        Default 6.  Higher = more complex patterns, but risks overfitting.
        Think of it like: "Ask at most 6 yes/no questions before making a guess."

    ``min_child_weight`` — How many data points must land in each leaf.
        Default 10.  Higher = more conservative (harder to split into tiny groups).

    ``subsample`` — What fraction of training rows each tree sees.
        Default 0.8 = 80 %.  Reduces overfitting by showing each tree a
        slightly different view of the data.

    ``colsample_bytree`` — What fraction of features each tree sees.
        Default 0.8 = 80 %.  Similar benefit to subsample but at the
        feature level.

    Regularization (how much to penalise complexity):

    ``reg_alpha`` (L1) — Encourages the model to ignore noisy features.
        Default 0.1.  Set higher if you suspect many irrelevant features.

    ``reg_lambda`` (L2) — Dampens large predictions.
        Default 1.0.  The standard regularization knob — almost always leave at 1.0.

    Learning (how fast the model adjusts):

    ``learning_rate`` — How much each tree can adjust the prediction.
        Default 0.05.  Smaller = slower learning, but less likely to overshoot.
        Typically paired with a larger ``n_estimators``.

    ``n_estimators`` — Maximum number of trees to train.
        Default 2000, but early stopping usually stops around 50–200.
        Set higher if ``learning_rate`` is low.

    Training windows (how much history to use):

    ``min_train_size`` — Minimum hours of history required.
        Default 4,320h (~6 months).  Fewer months = risk of not capturing
        seasonal patterns (e.g. winter/summer demand differences).

    ``val_size`` — Hours held out for early stopping.
        Default 168h (1 week).  The last week of training data is used to
        decide when to stop adding trees.

    Example
    -------
    >>> cfg = XGBConfig(max_depth=8, learning_rate=0.03, n_estimators=3000)
    >>> model = XGBPriceModel(config=cfg)
    """

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
    XGBoost price forecaster for the Greek electricity bidding zone.

    What this model does
    ---------------------
    Given the last few weeks of grid + weather data, it predicts what each
    of tomorrow's 24 hourly electricity prices will be (in EUR/MWh).

    It does NOT predict the future perfectly — it learns patterns from history:
    e.g. "prices tend to spike at 8am on weekday mornings in winter" or
    "high solar output on sunny afternoons drives prices down".

    Two modes of forecasting are supported:

    1. Single-step (evaluate)
       You have historical data including the actual price.  The model
       predicts the next hour.  Useful for measuring accuracy on known data.

    2. Recursive / 24h forecast (forecast_24h)
       You only have data up to today 23:00.  The model predicts hour T+1.
       That prediction is fed back as input to predict T+2, and so on.
       This simulates real-world conditions where tomorrow's "actual" prices
       are not yet known.

    Training strategy
    -----------------
    The model is retrained regularly (ideally weekly) on all available history.
    It uses an **expanding window**: the training set grows as new data arrives.
    A **validation hold-out** (last week) is used for early stopping, which
    prevents the model from overfitting — learning noise instead of signal.

    Parameters
    ----------
    config : XGBConfig
        Hyperparameter settings (tree depth, learning rate, regularization, etc.).
        Defaults to sensible values; only override if you have a reason.
    target_col : str
        Name of the price column in the input DataFrame.
        Auto-detected if omitted (tries "price_gr" first).
    lag_features : tuple of int
        How far back to look for price patterns, in hours.
        Default (24, 48, 168, 336) means:
        - 24h ago  — same hour yesterday
        - 48h ago  — same hour two days ago
        - 168h ago — same hour last week
        - 336h ago — same hour two weeks ago
    seasonal_years : tuple of int
        Which calendar years to include in the Greek holiday table.
    include_dst : bool
        Whether to add Daylight Saving Time transition flags.
        Greece switches clocks in March and October — these shifts confuse
        naive hourly models, so explicit DST features help.
    verbose : bool
        Print training progress messages.

    Attributes
    ----------
    model_ : xgboost.XGBRegressor
        The fitted XGBoost model.  Access this to inspect individual trees
        or use the model outside this class.
    feature_cols_ : list of str
        Names of the columns the model was trained on.  Any new input DataFrame
        must have these exact columns.
    feature_importance_ : dict
        Mean absolute SHAP values per feature.  Set by ``shap_summary()``.
        Higher value = more influential on the model's predictions.
    config_ : XGBConfig
        The hyperparameter configuration used (may differ from input defaults).

    Example usage
    -------------
    >>> from gridprice.models import XGBPriceModel
    >>> from gridprice.synthetic_data import SyntheticDataGenerator
    >>> from datetime import date
    >>>
    >>> # 1. Generate training data (replace with real ENTSO-E data in production)
    >>> df = SyntheticDataGenerator(bidding_zone="GR", seed=42, end_date=date(2024, 6, 1)).generate()
    >>>
    >>> split = int(len(df) * 0.8)
    >>> df_train, df_test = df.iloc[:split], df.iloc[split:]
    >>>
    >>> # 2. Train the model
    >>> model = XGBPriceModel(verbose=True)
    >>> model.fit(df_train)          # ~2 min on synthetic data
    >>>
    >>> # 3. Evaluate on held-out test data (single-step)
    >>> metrics = model.evaluate(df_test)
    >>> print(f"MAE: {metrics['MAE']:.2f} EUR/MWh")
    >>>
    >>> # 4. Produce a 24h forecast (recursive — how you'd use it in production)
    >>> forecast = model.forecast_24h(df_train.iloc[-500:])
    >>> print(forecast.head())
    >>>
    >>> # 5. Inspect which features matter most
    >>> fi = model.feature_importance_df()
    >>> print(fi.head(5))
    >>>
    >>> # 6. Save to disk
    >>> model.save("models/xgb_model.pkl")
    >>> loaded = XGBPriceModel.load("models/xgb_model.pkl")  # recreate from disk
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
        Train (or retrain) the XGBoost model on historical price data.

        What happens inside
        -------------------
        The training process has three phases:

        Phase 1 — Feature engineering
            Raw grid + weather data is transformed into 51 numeric features:
            calendar (hour, day-of-week, holiday), cyclical encodings (sin/cos of hour),
            lag features (price 24h/48h/168h ago), rolling statistics (24h/168h means),
            weather variables, and grid-level variables (residual load, renewable share).
            NaN rows (from lag windows at the start of the series) are dropped.

        Phase 2 — Expanding window
            The model is trained multiple times, each time expanding the training
            window by one week.  At each step the last ``val_size`` hours are
            held out as a validation set and early stopping is applied.  The number
            of boosting rounds that performed best on validation is recorded.
            This is a standard technique to (a) prevent overfitting and (b) pick
            a reasonable number of trees without manual tuning.

            Why expand?  Rather than training on just the last 6 months, we train
            on 6 months, then 6 months + 1 week, then 6 months + 2 weeks, and so on.
            This simulates how the model would have performed if it had been
            deployed at each point in history — giving a realistic picture of
            what accuracy to expect in production.

        Phase 3 — Final retrain
            All recorded best-iteration counts are averaged.  A final model is
            trained on all training data using that average + 5 as the tree count.
            This final model is what gets used for forecasting.

        Parameters
        ----------
        df : DataFrame with DatetimeIndex
            Raw (pre-feature) hourly data.  Must contain at minimum:
            - ``price_gr`` (or the configured target column): hourly price in EUR/MWh
            - ``load_gr``: hourly total electricity load in MW
            - Weather columns: ``temperature_2m_c``, ``wind_speed_10m_ms``,
              ``cloud_cover``, ``solar_irradiance_wm2``
            Columns not present are silently ignored (the model will just have
            fewer features).

        eval_fraction : float
            Fraction of ``df`` to hold out for the final validation step.
            Default 0.1 = last 10 % of the series.
            Must contain at least ``val_size`` hours (168h = 1 week by default).

        Returns
        -------
        self (allows method chaining: ``model = XGBPriceModel().fit(df)``)

        Raises
        ------
        ValueError
            If ``df`` has fewer than ``min_train_size`` rows after dropping NaNs
            (i.e. too little history to build lag features).
        RuntimeError
            If the expanding window produces zero valid training steps
            (usually means the dataset is too short).

        Example
        -------
        >>> model = XGBPriceModel(verbose=True)
        >>> model.fit(df_train)          # ~2 min on 2 years of hourly data
        [XGBPriceModel] 17520 rows, 1752 held for eval, step=168h, val_size=168h
        [XGBPriceModel] Avg best iteration: 87 (range 42-156)
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
        Predict the next hour's price for a feature matrix (single-step).

        This is a **batch prediction**: all rows in ``X`` are predicted
        independently.  It does NOT chain predictions — each row is treated
        as if the correct features are already available.  For chained
        24-step-ahead forecasting, use ``forecast_24h()`` instead.

        Parameters
        ----------
        X : DataFrame
            Feature matrix.  Must have exactly the columns the model was
            trained on (``model.feature_cols_``) in the same order.
            Typically produced by passing raw data through a FeatureEngineer:
            ``df_fe = FeatureEngineer().fit_transform(df)``.
            Use ``df_fe.dropna()`` to remove rows with NaN (lag warm-up period).

        Returns
        -------
        Series of predicted prices in EUR/MWh, indexed the same as ``X``.
        NaN values in ``X`` produce NaN predictions.

        Example
        -------
        >>> # Engineer features for test data
        >>> fe = FeatureEngineer(lag_features=(24, 48, 168, 336))
        >>> df_fe = fe.fit_transform(df_test).dropna()
        >>>
        >>> # Batch predict — no chaining, each row independent
        >>> preds = model.predict(df_fe)
        >>> print(preds.head())
        2024-05-20 00:00:00+03:00     87.34
        2024-05-20 01:00:00+03:00     81.12
        ...
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
        Produce N hours of price forecasts starting from the last known timestamp.

        This is the **production method** — how the model would be used in a
        daily pipeline.  It uses a **recursive (autoregressive) strategy**:

        Step-by-step example (for horizon=3):
        ┌─────────────────────────────────────────────────────────────┐
        │ last_known: data up to Mon 23:00                            │
        │                                                             │
        │  1. Engineer features using data up to Mon 23:00             │
        │  2. Predict Tue 00:00  ← [forecast_1]                       │
        │  3. Append [forecast_1] as if it were the actual price     │
        │  4. Engineer features using data up to Tue 00:00             │
        │  5. Predict Tue 01:00  ← [forecast_2]                      │
        │  6. Append [forecast_2] ...                                 │
        │  7. Predict Tue 02:00  ← [forecast_3]                      │
        └─────────────────────────────────────────────────────────────┘

        Why not predict all 24 hours at once?
        Direct multi-output (train one model to output 24 values) would avoid
        error compounding but requires significantly more engineering.
        Recursive is simpler, well-studied, and competitive in practice.
        Error does compound — hour 24 is typically less accurate than hour 1 —
        but the model is still useful if this degradation is quantified.

        Parameters
        ----------
        last_known : DataFrame with DatetimeIndex
            Raw hourly data ending at the last timestamp you actually know.
            Must contain at least ``max(lag_features)`` hours (default: 336h = 2 weeks)
            to build lag and rolling features correctly.
            Must contain the target price column (``price_gr`` or equivalent).

        horizon : int
            How many hours ahead to forecast.  Default 24 (full day-ahead market).
            Can be set lower (e.g. 6) for intraday use or higher (e.g. 48) if needed.

        Returns
        -------
        Series of ``horizon`` price predictions in EUR/MWh.
        Index: DatetimeIndex starting at ``last_known.index[-1] + 1 hour``,
        same timezone as ``last_known``.

        Raises
        ------
        ValueError
            If ``last_known`` has fewer rows than ``max(lag_features)``.
        ValueError
            If called before ``fit()``.

        Example
        -------
        >>> # You have data up to yesterday 23:00 — now predict today 00:00 to 23:00
        >>> forecast = model.forecast_24h(df_train.iloc[-500:])
        >>> print(forecast)
        2024-06-02 00:00:00+03:00     87.34
        2024-06-02 01:00:00+03:00     81.12
        ...
        2024-06-02 23:00:00+03:00     92.05
        Length: 24

        >>> # Or for an intraday 6-hour ahead forecast:
        >>> intraday = model.forecast_24h(df_train.iloc[-100:], horizon=6)
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
        Measure how accurate the model is on a held-out test period.

        This is a **single-step evaluation**: for each hour in the test set,
        the model sees all features INCLUDING the actual price (lag features
        are computed using actuals, not predictions).  This gives the most
        optimistic accuracy estimate — in real recursive forecasting, errors
        compound because the model doesn't have perfect lag features.

        Use ``evaluate_recursive()`` to measure realistic performance where
        the model must rely on its own previous predictions.

        What the metrics mean
        --------------------
        MAE  (Mean Absolute Error)
            Average prediction error in EUR/MWh.  E.g. MAE = 5 means
            the model's forecast is off by 5 EUR/MWh on average.
            The most interpretable metric.

        RMSE (Root Mean Squared Error)
            Like MAE but squares errors before averaging — heavily penalises
            large mistakes.  If RMSE ≫ MAE, the model occasionally makes
            big mistakes (e.g. misses a price spike).

        MAPE (Mean Absolute Percentage Error)
            Average percentage error relative to the actual price.
            E.g. MAPE = 15% means predictions are off by 15% on average.
            Problematic when prices are near zero or negative.

        sMAPE (Symmetric MAPE)
            Like MAPE but symmetric — equally penalises over- and under-prediction
            regardless of the actual price level.  Better behaved for energy
            prices which can swing from 0 to 300+ EUR/MWh.

        Parameters
        ----------
        df_test : DataFrame with DatetimeIndex
            Raw (pre-feature) test data.  Must contain the target price column.
            Must be long enough to build lag/rolling features:
            ``len(df_test)`` should be at least ``max(lag_features)`` + 1.

        metrics : tuple of str
            Which metrics to compute.  Subset of {"MAE", "RMSE", "MAPE", "sMAPE"}.
            All four are computed by default.

        Returns
        -------
        dict of metric_name → float value.
        NaN is returned for MAPE/sMAPE if the test set contains zero prices.

        Example
        -------
        >>> # Hold out the last 20% of data for evaluation
        >>> split = int(len(df) * 0.8)
        >>> df_train, df_test = df.iloc[:split], df.iloc[split:]
        >>>
        >>> model.fit(df_train)
        >>> metrics = model.evaluate(df_test)
        >>>
        >>> print(f"MAE:   {metrics['MAE']:.2f} EUR/MWh")
        >>> print(f"RMSE:  {metrics['RMSE']:.2f} EUR/MWh")
        >>> print(f"sMAPE: {metrics['sMAPE']:.1f}%")
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
        Measure accuracy of recursive multi-step forecasting on the test set.

        This is the **realistic accuracy test**.  Unlike ``evaluate()`` which
        uses actual prices for lag features, this method forces the model to
        use its own previous predictions as inputs — exactly as it would in
        production.

        How it works
        ------------
        The test set is split into non-overlapping blocks of ``horizon`` hours.
        For each block:

        1. History before the block is taken as ``last_known``.
        2. ``forecast_24h(history, horizon)`` is called — producing a
           recursive forecast for the entire block.
        3. Each predicted hour is compared to the actual price.

        This gives you per-hour accuracy broken down by how many steps ahead
        the forecast was made (h=1 is usually most accurate, h=24 is least).

        Parameters
        ----------
        df_test : DataFrame with DatetimeIndex
            Raw (pre-feature) test data.  Must contain the target price column.
            Must be at least ``max(lag_features) + horizon`` hours long.

        horizon : int
            Number of hours to forecast per block.  Default 24.
            The step between block starts is also ``horizon`` (non-overlapping blocks).

        Returns
        -------
        DataFrame with one row per forecast hour, indexed by timestamp.

        Columns
        -------
        horizon  : int   — 1 to N, how many hours ahead this prediction was made
        y_true   : float — actual price (EUR/MWh) at this hour
        y_pred   : float — model's recursive prediction (EUR/MWh)

        Use this to answer questions like:
        - "On average, how wrong is the 24-hour-ahead forecast?"
        - "Do errors grow linearly with horizon, or plateau after h=12?"

        Example
        -------
        >>> results = model.evaluate_recursive(df_test, horizon=24)
        >>>
        >>> # Average error at each horizon
        >>> results["error"] = results["y_true"] - results["y_pred"]
        >>> print(results.groupby("horizon")["error"].abs().mean())
        horizon
        1      1.23
        6      2.87
        12     4.15
        24     6.42
        Name: error, dtype: float64
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
        Compute SHAP (SHapley Additive exPlanations) feature importance.

        SHAP explains each individual prediction by assigning each feature
        a "contribution score" — how much that feature pushed the prediction
        up or down relative to the average prediction.  This method averages
        the absolute contribution scores across all rows in ``df_sample``,
        giving an overall ranking of which features matter most.

        Think of it as: "On average across these predictions, which features
        had the biggest influence on the price forecast?"

        Requires ``shap`` package: ``pip install shap``.

        Parameters
        ----------
        df_sample : DataFrame
            Feature-engineered sample rows.  Must have the same columns
            as ``self.feature_cols_``.  Use 500–2000 rows for a good
            balance between coverage and speed.  Example:
            ``df_sample = FeatureEngineer().fit_transform(df_test).dropna().iloc[:1000]``

        max_display : int
            How many features to show in the beeswarm plot (set by ``shap_plot()``).

        Returns
        -------
        dict of feature_name → mean |SHAP value| (higher = more important).

        The result is also stored in ``self.feature_importance_`` for later use.

        Example
        -------
        >>> fi = model.shap_summary(df_sample)
        >>> for feat, score in sorted(fi.items(), key=lambda x: -x[1])[:5]:
        ...     print(f"  {feat:30s} {score:.4f}")
          price_roll_zscore_168h          12.3421
          price_vs_7d_mean                 8.2154
          renewable_share                   5.8732
          temperature_2m_c                 4.1023
          hour_of_week                     3.9871
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
        """
        Generate and display a SHAP beeswarm summary plot.

        The beeswarm plot shows every prediction as a dot per feature:
        - Horizontal position = SHAP value (how much this feature moved the prediction)
        - Color = feature value (red = high, blue = low)
        - Clustering at the top = high-impact features

        This is the standard way to understand *why* the model makes certain
        predictions, beyond just knowing *what* the predictions are.

        Requires ``shap`` package: ``pip install shap``.

        Parameters
        ----------
        df_sample : DataFrame
            Feature-engineered sample rows (same as for ``shap_summary()``).
        save_path : Path, optional
            If provided, saves the plot to this path instead of displaying it.

        Example
        -------
        >>> model.shap_plot(df_sample)                       # display inline
        >>> model.shap_plot(df_sample, save_path="reports/figures/shap.png")
        """
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
        Return XGBoost's native feature importance scores (gain-based).

        "Gain" measures how much each feature improves the model's accuracy
        (reduction in prediction error) when used to split a tree node.
        This is XGBoost's built-in metric — simpler than SHAP, faster to compute,
        but less precise for understanding individual predictions.

        Use this for a quick overview; use ``shap_summary()`` when you need
        rigorous, per-prediction explanations.

        Returns
        -------
        DataFrame with columns:
        - ``feature`` : feature name
        - ``importance`` : gain-based importance score (higher = more important)
        Sorted descending by importance.

        Example
        -------
        >>> fi = model.feature_importance_df()
        >>> print(fi.head(10).to_string(index=False))
          feature               importance
          price_roll_zscore_168h     0.182
          price_vs_7d_mean          0.134
          renewable_share           0.098
          temperature_2m_c          0.072
          hour_of_week              0.065
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
        """
        Persist the trained model to disk using pickle.

        Saves everything needed to recreate the model: the fitted XGBoost
        booster, the hyperparameter config, the list of feature column names,
        and the target column name.

        Files are typically ~1–5 MB depending on the number of trees.

        Parameters
        ----------
        path : Path (or str)
            Destination file path.  The parent directory is created if it
            doesn't exist.  Convention: ``models/xgb_model.pkl`` or
            ``models/xgb_2024-06-01.pkl`` for dated backups.

        Note
        ----
        The model does NOT save the feature engineer's state.  Lag features,
        rolling windows, and holiday calendars are recomputed at prediction time
        from the raw data.  This means: (a) you don't need to version the
        feature logic separately, and (b) predictions will always use the
        current version of the feature engineering code.

        Example
        -------
        >>> from pathlib import Path
        >>> model.save(Path("models/xgb_2024-06-01.pkl"))
        >>> # Later, in a different process:
        >>> loaded = XGBPriceModel.load(Path("models/xgb_2024-06-01.pkl"))
        >>> loaded.forecast_24h(df)
        """
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
        """
        Load a model from a pickle file previously saved by ``save()``.

        Returns a fully reconstructed ``XGBPriceModel`` instance:
        the model is already fitted, so you can call ``forecast_24h()`` or
        ``evaluate()`` immediately without calling ``fit()`` again.

        Parameters
        ----------
        path : Path (or str)
            Path to the ``.pkl`` file saved by ``save()``.

        Returns
        -------
        XGBPriceModel
            A reconstructed instance with ``.model_`` and ``.feature_cols_`` restored.

        Example
        -------
        >>> from pathlib import Path
        >>> model = XGBPriceModel.load(Path("models/xgb_model.pkl"))
        >>> print(f"Loaded model with {len(model.feature_cols_)} features")
        Loaded model with 51 features
        >>> model.forecast_24h(df_recent)  # ready to use immediately
        """
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
