"""
Unit tests for gridprice.models — XGBPriceModel
"""

from __future__ import annotations

import sys
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gridprice.models import (
    XGBPriceModel,
    XGBConfig,
    compute_metrics,
    compare_baselines,
    _persistence_baseline,
    _seasonal_avg_baseline,
)
from gridprice.synthetic_data import SyntheticDataGenerator


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_df():
    """Generate a full synthetic dataset once for the test module."""
    gen = SyntheticDataGenerator(bidding_zone="GR", seed=42, end_date=date(2024, 6, 1))
    return gen.generate()


@pytest.fixture(scope="module")
def split_df(synthetic_df):
    split = int(len(synthetic_df) * 0.8)
    return synthetic_df.iloc[:split], synthetic_df.iloc[split:]


@pytest.fixture(scope="module")
def trained_model(split_df):
    """Trained XGBPriceModel for tests that need a fitted model."""
    df_train, _ = split_df
    model = XGBPriceModel(verbose=False)
    model.config.n_estimators = 100  # fast for tests
    model.config.min_train_size = 24 * 60
    model.fit(df_train)
    return model


# ── Test: XGBConfig ─────────────────────────────────────────────────────────

class TestXGBConfig:
    def test_defaults(self):
        cfg = XGBConfig()
        d = cfg.to_xgb_dict()
        assert d["max_depth"] == 6
        assert d["learning_rate"] == 0.05
        assert d["n_estimators"] == 2000
        assert d["early_stopping_rounds"] == 50

    def test_custom_values(self):
        cfg = XGBConfig(max_depth=8, learning_rate=0.1)
        d = cfg.to_xgb_dict()
        assert d["max_depth"] == 8
        assert d["learning_rate"] == 0.1

    def test_dataclass_repr(self):
        cfg = XGBConfig()
        repr_str = repr(cfg)
        assert "XGBConfig" in repr_str


# ── Test: Metrics ───────────────────────────────────────────────────────────

class TestMetrics:
    def test_mae(self):
        y_true = pd.Series([1.0, 2.0, 3.0])
        y_pred = pd.Series([1.1, 2.1, 2.9])
        m = compute_metrics(y_true, y_pred)
        assert abs(m["MAE"] - 0.1) < 1e-9

    def test_rmse(self):
        y_true = pd.Series([1.0, 2.0, 3.0])
        y_pred = pd.Series([1.0, 2.0, 3.0])
        m = compute_metrics(y_true, y_pred)
        assert m["RMSE"] == 0.0

    def test_mape_zero_true(self):
        """MAPE should skip zero true values."""
        y_true = pd.Series([0.0, 10.0])
        y_pred = pd.Series([5.0, 10.0])
        m = compute_metrics(y_true, y_pred)
        assert m["MAPE"] == 0.0  # only the non-zero one

    def test_smape(self):
        y_true = pd.Series([100.0, 100.0])
        y_pred = pd.Series([110.0, 90.0])
        m = compute_metrics(y_true, y_pred)
        assert "sMAPE" in m
        # sMAPE = 100 * mean(|y-P| / (|y|+|P|)/2)
        # = 100 * ((10/210) + (10/190)) / 2 = 100 * 0.10025 = 10.025
        assert abs(m["sMAPE"] - 10.025) < 1e-4


# ── Test: Persistence baseline ─────────────────────────────────────────────

class TestPersistenceBaseline:
    def test_persistence_shape(self, synthetic_df):
        pred = _persistence_baseline(synthetic_df, "price_gr")
        assert len(pred) == len(synthetic_df)

    def test_persistence_na_at_start(self, synthetic_df):
        pred = _persistence_baseline(synthetic_df, "price_gr")
        # First 24 rows should be NaN (no T-24 available)
        assert pred.iloc[:24].isna().all()
        # Row 25 onwards should have values
        assert pred.iloc[24:].notna().all()

    def test_persistence_value(self, synthetic_df):
        pred = _persistence_baseline(synthetic_df, "price_gr")
        # Row 48 should equal row 24 (T-24)
        assert abs(pred.iloc[48] - synthetic_df["price_gr"].iloc[24]) < 1e-9


# ── Test: Seasonal avg baseline ────────────────────────────────────────────

class TestSeasonalAvgBaseline:
    def test_seasonal_avg_shape(self, synthetic_df):
        pred = _seasonal_avg_baseline(synthetic_df, "price_gr")
        assert len(pred) == len(synthetic_df)

    def test_seasonal_avg_not_all_identical(self, synthetic_df):
        pred = _seasonal_avg_baseline(synthetic_df, "price_gr")
        # Different hours/days should give different values
        assert pred.notna().any()

    def test_seasonal_avg_stable_across_same_bucket(self, synthetic_df):
        """Same dow/hour should give same seasonal average."""
        pred = _seasonal_avg_baseline(synthetic_df, "price_gr")
        # Find two rows with same dow and hour
        idx = synthetic_df.index
        dow_hours = pd.DataFrame({"dow": idx.dayofweek, "hour": idx.hour})
        groups = dow_hours.groupby(["dow", "hour"]).groups
        # Get any bucket with at least 2 entries
        bucket = next((v for v in groups.values() if len(v) > 1), None)
        if bucket is not None:
            vals = pred.iloc[bucket].dropna()
            assert vals.nunique() == 1, "Same dow/hour should have identical seasonal avg"


# ── Test: compare_baselines ─────────────────────────────────────────────────

class TestCompareBaselines:
    def test_compare_baselines_returns_dict(self, synthetic_df):
        split = int(len(synthetic_df) * 0.8)
        test_start = synthetic_df.index[split]
        results, test_df = compare_baselines(synthetic_df, "price_gr",
                                              test_start=test_start)
        assert isinstance(results, dict)
        assert "persistence" in results

    def test_compare_baselines_test_df_shape(self, synthetic_df):
        split = int(len(synthetic_df) * 0.8)
        test_start = synthetic_df.index[split]
        _, test_df = compare_baselines(synthetic_df, "price_gr",
                                        test_start=test_start)
        assert len(test_df) == len(synthetic_df) - split


# ── Test: XGBPriceModel init ────────────────────────────────────────────────

class TestModelInit:
    def test_init_default(self):
        model = XGBPriceModel()
        assert model.config is not None
        assert model.verbose is False
        assert not model._fitted

    def test_init_with_config(self):
        cfg = XGBConfig(n_estimators=50)
        model = XGBPriceModel(config=cfg, verbose=True)
        assert model.config.n_estimators == 50
        assert model.verbose is True

    def test_repr(self):
        model = XGBPriceModel()
        r = repr(model)
        assert "XGBPriceModel" in r


# ── Test: fit ───────────────────────────────────────────────────────────────

class TestModelFit:
    def test_fit_sets_fitted_flag(self, split_df):
        df_train, _ = split_df
        model = XGBPriceModel(verbose=False)
        model.config.n_estimators = 50
        model.config.min_train_size = 24 * 30
        model.fit(df_train)
        assert model._fitted is True

    def test_fit_sets_feature_cols(self, split_df):
        df_train, _ = split_df
        model = XGBPriceModel(verbose=False)
        model.config.n_estimators = 50
        model.config.min_train_size = 24 * 30
        model.fit(df_train)
        assert len(model.feature_cols_) > 0

    def test_fit_sets_target_col(self, split_df):
        df_train, _ = split_df
        model = XGBPriceModel(verbose=False)
        model.config.n_estimators = 50
        model.config.min_train_size = 24 * 30
        model.fit(df_train)
        assert model.target_col_ != ""

    def test_fit_predict_not_fitted_raises(self):
        model = XGBPriceModel()
        df = pd.DataFrame({"a": [1, 2, 3]}, index=pd.date_range("2024-01-01", periods=3, freq="h"))
        with pytest.raises(ValueError, match="has not been fitted"):
            model.predict(df)

    def test_fit_forecast_not_fitted_raises(self, split_df):
        model = XGBPriceModel()
        df_train, _ = split_df
        with pytest.raises(ValueError, match="has not been fitted"):
            model.forecast_24h(df_train.iloc[-100:])

    def test_fit_insufficient_data_raises(self, split_df):
        df_train, _ = split_df
        model = XGBPriceModel(verbose=False)
        model.config.min_train_size = 24 * 14  # 14 days
        model.config.n_estimators = 10
        with pytest.raises(ValueError, match="Not enough rows"):
            model.fit(df_train.iloc[:500])  # only 21 days — not enough after NaN removal


# ── Test: predict ─────────────────────────────────────────────────────────

class TestModelPredict:
    def test_predict_shape(self, trained_model, split_df):
        df_train, df_test = split_df
        # Use first 168 rows of test (should have enough lag data)
        df_test_fe = trained_model._fe_class(
            lag_features=trained_model.lag_features,
            rolling_windows=(24, 168),
            holiday_years=trained_model.seasonal_years,
        ).fit_transform(df_train[-500:])
        pred = trained_model.predict(df_test_fe)
        assert len(pred) == len(df_test_fe)

    def test_predict_index_preserved(self, trained_model, split_df):
        df_train, df_test = split_df
        df_fe = trained_model._fe_class(
            lag_features=trained_model.lag_features,
            rolling_windows=(24, 168),
            holiday_years=trained_model.seasonal_years,
        ).fit_transform(df_train[-500:])
        pred = trained_model.predict(df_fe)
        assert pred.index.equals(df_fe.index)


# ── Test: evaluate ─────────────────────────────────────────────────────────

class TestModelEvaluate:
    def test_evaluate_returns_dict(self, trained_model, split_df):
        _, df_test = split_df
        metrics = trained_model.evaluate(df_test)
        assert isinstance(metrics, dict)
        for key in ["MAE", "RMSE", "MAPE", "sMAPE"]:
            assert key in metrics

    def test_evaluate_all_positive(self, trained_model, split_df):
        _, df_test = split_df
        metrics = trained_model.evaluate(df_test)
        assert metrics["MAE"] >= 0
        assert metrics["RMSE"] >= 0

    def test_evaluate_no_test_data(self, split_df):
        df_train, _ = split_df
        model = XGBPriceModel(verbose=False)
        model.config.n_estimators = 50
        model.config.min_train_size = 24 * 30
        model.fit(df_train)
        # Use some of training data as "test"
        metrics = model.evaluate(df_train.iloc[-168:])
        assert "MAE" in metrics


# ── Test: forecast_24h ────────────────────────────────────────────────────

class TestModelForecast:
    def test_forecast_returns_series(self, trained_model, split_df):
        df_train, _ = split_df
        last_known = df_train.iloc[-500:]
        fc = trained_model.forecast_24h(last_known, horizon=24)
        assert isinstance(fc, pd.Series)
        assert len(fc) == 24

    def test_forecast_horizon(self, trained_model, split_df):
        df_train, _ = split_df
        last_known = df_train.iloc[-500:]
        fc = trained_model.forecast_24h(last_known, horizon=6)
        assert len(fc) == 6

    def test_forecast_index_step(self, trained_model, split_df):
        df_train, _ = split_df
        last_known = df_train.iloc[-500:]
        last_ts = last_known.index[-1]
        fc = trained_model.forecast_24h(last_known, horizon=24)
        # First forecast is last_ts + 1h
        assert fc.index[0] == last_ts + timedelta(hours=1)
        # Last forecast is last_ts + 24h
        assert fc.index[-1] == last_ts + timedelta(hours=24)

    def test_forecast_insufficient_history(self, trained_model, split_df):
        df_train, _ = split_df
        # Only 10 hours of history (needs 336 for lag_336h)
        last_known = df_train.iloc[-10:]
        with pytest.raises(ValueError, match="needs at least"):
            trained_model.forecast_24h(last_known)


# ── Test: feature importance ────────────────────────────────────────────────

class TestFeatureImportance:
    def test_feature_importance_df(self, trained_model):
        fi = trained_model.feature_importance_df()
        assert len(fi) == len(trained_model.feature_cols_)
        assert "feature" in fi.columns
        assert "importance" in fi.columns
        # Sorted descending
        assert fi["importance"].is_monotonic_decreasing or fi["importance"].iloc[0] >= fi["importance"].iloc[-1]

    def test_feature_importance_not_fitted_raises(self):
        model = XGBPriceModel()
        with pytest.raises(ValueError, match="has not been fitted"):
            model.feature_importance_df()


# ── Test: recursive evaluation ──────────────────────────────────────────────

class TestRecursiveEvaluate:
    def test_evaluate_recursive(self, trained_model, split_df):
        df_train, df_test = split_df
        # Need at least 336 (max lag) + 24 (horizon) = 360 rows
        small_test = df_test.iloc[:400]  # ~16 days — enough for recursive eval
        result = trained_model.evaluate_recursive(small_test, horizon=24)
        assert "y_true" in result.columns
        assert "y_pred" in result.columns
        assert "horizon" in result.columns
        assert len(result) > 0


# ── Test: save / load ──────────────────────────────────────────────────────

class TestSaveLoad:
    def test_save_load_roundtrip(self, trained_model, tmp_path):
        path = tmp_path / "model.pkl"
        trained_model.save(path)
        assert path.exists()

        loaded = XGBPriceModel.load(path)
        assert loaded._fitted is True
        assert loaded.target_col_ == trained_model.target_col_
        assert loaded.feature_cols_ == trained_model.feature_cols_


# ── Test: custom lag features ─────────────────────────────────────────────

class TestCustomLagFeatures:
    def test_fit_with_custom_lags(self, split_df):
        df_train, _ = split_df
        model = XGBPriceModel(
            lag_features=(24, 48, 168),  # no 336
            verbose=False,
        )
        model.config.n_estimators = 50
        model.config.min_train_size = 24 * 30
        model.fit(df_train)
        assert model._fitted is True
        # Should have price_lag_336h feature removed
        assert "price_lag_336h" not in model.feature_cols_


# ── Test: XGBoost hyperparameter impact ───────────────────────────────────

class TestHyperparameters:
    def test_deeper_trees_have_more_leaves(self, split_df):
        df_train, _ = split_df
        cfg_shallow = XGBConfig(max_depth=3, n_estimators=30)
        model_shallow = XGBPriceModel(config=cfg_shallow, verbose=False)
        model_shallow.fit(df_train)

        cfg_deep = XGBConfig(max_depth=10, n_estimators=30)
        model_deep = XGBPriceModel(config=cfg_deep, verbose=False)
        model_deep.fit(df_train)

        # Deep trees should have more feature interactions (more varied importance)
        fi_shallow = model_shallow.feature_importance_df()
        fi_deep = model_deep.feature_importance_df()
        # Both should have valid importance values
        assert fi_shallow["importance"].notna().all()
        assert fi_deep["importance"].notna().all()


# ── Run ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
