"""
GridPrice — Baseline Models

Implements the simplest possible forecasting benchmarks:
- Persistence: today's price = tomorrow's price
- Historical average: same-hour average over last N weeks
- Linear regression with calendar + lag features

These baselines establish the floor that our ML models must beat.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass
class BaselineMetrics:
    """Container for baseline evaluation results."""
    model_name: str
    mae: float
    rmse: float
    mape: float
    description: str


def persistence_forecast(
    prices: pd.Series,
    n_ahead: int = 24,
) -> np.ndarray:
    """
    Persistence baseline: the price at hour T becomes the forecast for T+24.
    Returns an array of length n_ahead.
    """
    if len(prices) < n_ahead:
        return np.full(n_ahead, np.nan)
    return prices.iloc[-n_ahead:].values.copy()


def historical_average_forecast(
    prices: pd.Series,
    n_weeks: int = 4,
) -> np.ndarray:
    """
    Historical average: for each hour of the day, take the average
    over the last `n_weeks` weeks at that same hour.
    Returns 24 values.
    """
    if len(prices) < 24 * n_weeks:
        return np.full(24, np.nan)

    recent = prices.iloc[-(24 * n_weeks):]
    # Group by hour of day
    hourly_means = recent.groupby(recent.index.hour).mean()
    return hourly_means.values  # shape (24,)


def build_linear_features(
    df: pd.DataFrame,
    target_col: str,
    lags: Tuple[int, ...] = (24, 48, 168),
    rolling_windows: Tuple[int, ...] = (24, 168),
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build feature matrix for linear regression baseline.
    
    Features:
    - Lagged prices (t-24, t-48, t-168)
    - Rolling mean & std over recent windows
    - Hour of day, day of week, month (one-hot)
    - Public holiday flag (TBD: needs holiday data)
    
    Returns (X, y) with aligned index, no NaN rows.
    """
    df = df.copy()
    y = df[target_col].copy()

    # Lag features
    for lag in lags:
        df[f"lag_{lag}h"] = df[target_col].shift(lag)

    # Rolling statistics
    for w in rolling_windows:
        df[f"roll_mean_{w}h"] = df[target_col].rolling(w).mean()
        df[f"roll_std_{w}h"] = df[target_col].rolling(w).std()

    # Calendar features (one-hot)
    df["hour"] = df.index.hour
    df["dow"] = df.index.dayofweek
    df["month"] = df.index.month

    # One-hot encode
    df = pd.get_dummies(df, columns=["hour", "dow", "month"], prefix=["h", "dow", "mon"])

    # Drop rows with NaN (from lag/roll)
    df = df.dropna()
    y = y.loc[df.index]

    # Separate features
    exclude = {target_col}
    X = df.drop(columns=list(exclude & set(df.columns)))

    return X, y


def train_linear_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> LinearRegression:
    """Train a simple linear regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_baseline(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
    description: str = "",
) -> BaselineMetrics:
    """Compute standard regression metrics."""
    # Align lengths
    min_len = min(len(y_true), len(y_pred))
    y_t = y_true.iloc[:min_len].values
    y_p = y_pred[:min_len]

    mae = mean_absolute_error(y_t, y_p)
    rmse = np.sqrt(mean_squared_error(y_t, y_p))

    # MAPE — avoid division by zero
    nonzero = y_t != 0
    mape = (
        np.mean(np.abs((y_t[nonzero] - y_p[nonzero]) / y_t[nonzero])) * 100
        if nonzero.any()
        else np.inf
    )

    return BaselineMetrics(
        model_name=model_name,
        mae=float(mae),
        rmse=float(rmse),
        mape=float(mape),
        description=description or model_name,
    )


def run_all_baselines(
    prices: pd.Series,
    train_split: float = 0.8,
) -> Dict[str, BaselineMetrics]:
    """
    Run persistence, historical average, and linear regression baselines
    on the given price series. Returns dict of model_name → metrics.
    """
    n = len(prices)
    split_idx = int(n * train_split)
    train = prices.iloc[:split_idx]
    test = prices.iloc[split_idx:]

    metrics = {}

    # 1. Persistence
    pred_persist = persistence_forecast(train, n_ahead=len(test))
    metrics["persistence"] = evaluate_baseline(
        test, pred_persist, "persistence",
        "Today's price = tomorrow's forecast",
    )

    # 2. Historical average
    pred_hist = historical_average_forecast(train, n_weeks=4)
    # Tile to test length
    pred_hist_tiled = np.tile(pred_hist, int(np.ceil(len(test) / 24)))[:len(test)]
    metrics["historical_avg_4wk"] = evaluate_baseline(
        test, pred_hist_tiled, "historical_avg_4wk",
        "Average of last 4 weeks for each hour",
    )

    # 3. Linear regression
    try:
        X, y = build_linear_features(
            prices.to_frame(name="price"), target_col="price"
        )
        # Split preserving time order
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]

        # Drop any remaining NaNs
        valid_train = y_train.notna() & ~X_train.isna().any(axis=1)
        valid_test = y_test.notna() & ~X_test.isna().any(axis=1)

        model = train_linear_baseline(
            X_train[valid_train], y_train[valid_train]
        )
        y_pred_lr = model.predict(X_test[valid_test])

        metrics["linear_regression"] = evaluate_baseline(
            y_test[valid_test],
            y_pred_lr,
            "linear_regression",
            "Linear regression with lag + calendar features",
        )
    except Exception as e:
        print(f"[WARN] Linear regression failed: {e}")

    return metrics


def print_metrics_table(metrics: Dict[str, BaselineMetrics]) -> None:
    """Pretty-print baseline metrics."""
    print(f"{'Model':<25} {'MAE':>8} {'RMSE':>8} {'MAPE(%)':>8}")
    print("-" * 55)
    for name, m in metrics.items():
        print(f"{m.description:<25} {m.mae:>8.2f} {m.rmse:>8.2f} {m.mape:>8.2f}")


if __name__ == "__main__":
    # Quick test with synthetic data
    np.random.seed(42)
    idx = pd.date_range("2025-01-01", "2025-04-01", freq="h", tz=None)
    # Simulate daily + weekly pattern with noise
    daily = np.tile(np.sin(np.linspace(0, 2 * np.pi, 24)) * 30 + 50, len(idx) // 24 + 1)[:len(idx)]
    weekly = np.tile(np.tile([10, 5, 0, -5, -10, 0, 5], 24), len(idx) // 168 + 1)[:len(idx)]
    noise = np.random.normal(0, 8, len(idx))
    prices = pd.Series(daily + weekly + noise, index=idx, name="price")

    metrics = run_all_baselines(prices)
    print_metrics_table(metrics)
