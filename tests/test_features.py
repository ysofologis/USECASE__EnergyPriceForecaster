"""
Unit tests for gridprice.features — FeatureEngineer
"""

from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gridprice.features import (
    FeatureEngineer,
    engineer_features,
    _detect_price_col,
    _detect_load_col,
    _detect_weather_cols,
    _detect_gen_cols,
    _detect_flow_cols,
    _GREEK_HOLIDAYS,
    _dst_transition_dates,
)


# ── Fixtures ────────────────────────────────────────────────────────────────

def make_df(
    start: str = "2025-03-30", periods: int = 100, freq: str = "h", tz: str = "CET"
) -> pd.DataFrame:
    """Minimal DataFrame with all expected columns."""
    idx = pd.date_range(start, periods=periods, freq=freq, tz=tz)
    df = pd.DataFrame(
        {
            "price_eur_mwh": np.linspace(30, 80, periods) + np.random.randn(periods) * 2,
            "load_mw": np.full(periods, 5000.0),
            "gen_gr_solar": np.full(periods, 800.0),
            "gen_gr_wind_onshore": np.full(periods, 400.0),
            "temperature_2m_c": np.full(periods, 18.0),
            "wind_speed_100m_km_h": np.full(periods, 15.0),
            "cloud_cover": np.full(periods, 50.0),
            "solar_radiation": np.full(periods, 500.0),
        },
        index=idx,
    )
    return df


# ── Test: Greek holiday set ─────────────────────────────────────────────────

class TestGreekHolidays:
    def test_easter_2025(self):
        assert date(2025, 4, 20) in _GREEK_HOLIDAYS

    def test_easter_2026(self):
        assert date(2026, 4, 19) in _GREEK_HOLIDAYS

    def test_independence_day(self):
        assert date(2025, 3, 25) in _GREEK_HOLIDAYS
        assert date(2026, 3, 25) in _GREEK_HOLIDAYS

    def test_politechnio(self):
        assert date(2025, 11, 17) in _GREEK_HOLIDAYS
        assert date(2026, 11, 17) in _GREEK_HOLIDAYS

    def test_clean_monday(self):
        # Clean Monday 2025 = Mar 3
        assert date(2025, 3, 3) in _GREEK_HOLIDAYS


# ── Test: DST helpers ───────────────────────────────────────────────────────

class TestDST:
    def test_spring_forward_march(self):
        spring, _ = _dst_transition_dates(2025)
        assert spring.month == 3

    def test_autumn_back_october(self):
        _, autumn = _dst_transition_dates(2025)
        assert autumn.month == 10

    def test_spring_later_than_autumn(self):
        spring, autumn = _dst_transition_dates(2025)
        assert spring < autumn


# ── Test: Column detection ──────────────────────────────────────────────────

class TestColumnDetection:
    def test_detect_price_col(self):
        cols = pd.Index(["price_eur_mwh", "load_mw", "gen_gr_solar"])
        assert _detect_price_col(cols) == "price_eur_mwh"

    def test_detect_price_col_case_insensitive(self):
        cols = pd.Index(["PRICE_EUR_MWH", "load_mw"])
        assert _detect_price_col(cols) == "PRICE_EUR_MWH"

    def test_detect_load_col(self):
        cols = pd.Index(["load_mw", "load_forecast_mw"])
        assert _detect_load_col(cols) == "load_mw"

    def test_detect_weather_cols(self):
        cols = pd.Index(["temperature_2m_c", "wind_speed_100m_km_h", "cloud_cover", "foo"])
        detected = _detect_weather_cols(cols)
        assert "temperature" in detected
        assert "wind_speed" in detected
        assert "cloud_cover" in detected

    def test_detect_gen_cols(self):
        cols = pd.Index(["gen_gr_solar", "gen_gr_wind_onshore", "gen_gr_gas"])
        detected = _detect_gen_cols(cols, zone="GR")
        assert "solar" in detected
        assert "wind" in detected
        assert "gas" in detected


# ── Test: fit_transform shape ───────────────────────────────────────────────

class TestFitTransform:
    def test_shape_increases(self):
        df = make_df(periods=200)
        fe = FeatureEngineer()
        df_fe = fe.fit_transform(df)
        assert df_fe.shape[0] == df.shape[0]
        assert df_fe.shape[1] > df.shape[1]

    def test_price_col_detected(self):
        df = make_df(periods=100)
        fe = FeatureEngineer()
        fe.fit(df)
        assert fe.price_col_ == "price_eur_mwh"

    def test_load_col_detected(self):
        df = make_df(periods=100)
        fe = FeatureEngineer()
        fe.fit(df)
        assert fe.load_col_ == "load_mw"

    def test_gen_cols_detected(self):
        df = make_df(periods=100)
        fe = FeatureEngineer()
        fe.fit(df)
        assert "solar" in fe.gen_cols_
        assert "wind" in fe.gen_cols_

    def test_weather_cols_detected(self):
        df = make_df(periods=100)
        fe = FeatureEngineer()
        fe.fit(df)
        assert "temperature" in fe.weather_cols_
        assert "wind_speed" in fe.weather_cols_

    def test_holiday_set_populated(self):
        df = make_df(periods=100)
        fe = FeatureEngineer()
        fe.fit(df)
        assert len(fe.holiday_set_) > 50  # multiple years × holidays

    def test_fit_is_idempotent(self):
        df = make_df(periods=100)
        fe = FeatureEngineer()
        fe.fit(df)
        fe.fit(df)  # should not raise
        assert fe.price_col_ == "price_eur_mwh"


# ── Test: Calendar features ─────────────────────────────────────────────────

class TestCalendarFeatures:
    def test_hour_column(self):
        df = make_df(periods=48, start="2025-03-30")
        fe = FeatureEngineer()
        df_fe = fe.fit_transform(df)
        assert "hour" in df_fe.columns
        assert set(df_fe["hour"].unique()).issubset(set(range(24)))

    def test_day_of_week_values(self):
        df = make_df(periods=168, start="2025-03-30")
        fe = FeatureEngineer()
        df_fe = fe.fit_transform(df)
        assert df_fe["day_of_week"].min() >= 0
        assert df_fe["day_of_week"].max() <= 6

    def test_is_weekend_sunday(self):
        idx = pd.date_range("2025-03-30", periods=48, freq="h", tz="CET")  # Sunday
        df = make_df(periods=48, start="2025-03-30")
        fe = FeatureEngineer()
        df_fe = fe.fit_transform(df)
        # Sunday = dayofweek 6
        assert (df_fe.loc[df_fe["day_of_week"] == 6, "is_weekend"] == 1).all()

    def test_is_holiday_on_known_holiday(self):
        # Clean Monday 2025 = Mar 3
        df = make_df(periods=24, start="2025-03-03")
        fe = FeatureEngineer()
        df_fe = fe.fit_transform(df)
        assert (df_fe["is_holiday"] == 1).all()

    def test_is_holiday_off_on_normal_day(self):
        # Wednesday Feb 19 2025 is not a holiday
        df = make_df(periods=24, start="2025-02-19")
        fe = FeatureEngineer()
        df_fe = fe.fit_transform(df)
        assert (df_fe["is_holiday"] == 0).all()

    def test_month_column(self):
        df = make_df(periods=100)
        fe = FeatureEngineer()
        df_fe = fe.fit_transform(df)
        assert 1 <= df_fe["month"].min() <= 12
        assert 1 <= df_fe["month"].max() <= 12


# ── Test: Lag features ──────────────────────────────────────────────────────

class TestLagFeatures:
    def test_lag_24_exists(self):
        df = make_df(periods=50)
        fe = FeatureEngineer()
        df_fe = fe.fit_transform(df)
        assert "price_lag_24h" in df_fe.columns

    def test_lag_168_exists(self):
        df = make_df(periods=200)
        fe = FeatureEngineer()
        df_fe = fe.fit_transform(df)
        assert "price_lag_168h" in df_fe.columns

    def test_first_168_rows_have_nan_lag(self):
        df = make_df(periods=200)
        fe = FeatureEngineer()
        df_fe = fe.fit_transform(df)
        assert df_fe["price_lag_168h"].iloc[:167].isna().all()
        # After lag-168, should have values
        assert df_fe["price_lag_168h"].iloc[168:200].notna().all()

    def test_weekly_delta_exists(self):
        df = make_df(periods=200)
        fe = FeatureEngineer()
        df_fe = fe.fit_transform(df)
        assert "price_weekly_delta" in df_fe.columns


# ── Test: Rolling features ──────────────────────────────────────────────────

class TestRollingFeatures:
    def test_rolling_mean_exists(self):
        df = make_df(periods=50)
        fe = FeatureEngineer()
        df_fe = fe.fit_transform(df)
        assert "price_roll_mean_24h" in df_fe.columns

    def test_rolling_zscore_exists(self):
        df = make_df(periods=50)
        fe = FeatureEngineer()
        df_fe = fe.fit_transform(df)
        assert "price_roll_zscore_24h" in df_fe.columns


# ── Test: Weather features ─────────────────────────────────────────────────

class TestWeatherFeatures:
    def test_temperature_raw_exists(self):
        df = make_df(periods=50)
        fe = FeatureEngineer()
        df_fe = fe.fit_transform(df)
        assert "weather_temperature_raw" in df_fe.columns

    def test_wind_roll6h_exists(self):
        df = make_df(periods=50)
        fe = FeatureEngineer()
        df_fe = fe.fit_transform(df)
        assert "weather_wind_speed_roll6h" in df_fe.columns

    def test_weather_anomaly_exists(self):
        df = make_df(periods=50)
        fe = FeatureEngineer()
        df_fe = fe.fit_transform(df)
        assert "weather_temperature_anomaly" in df_fe.columns


# ── Test: Grid features ────────────────────────────────────────────────────

class TestGridFeatures:
    def test_residual_load_exists(self):
        df = make_df(periods=50)
        fe = FeatureEngineer()
        df_fe = fe.fit_transform(df)
        assert "residual_load" in df_fe.columns

    def test_residual_load_value(self):
        df = make_df(periods=50)
        fe = FeatureEngineer()
        df_fe = fe.fit_transform(df)
        # residual_load = load - solar - wind = 5000 - 800 - 400 = 3800
        np.testing.assert_allclose(df_fe["residual_load"].dropna(), 3800.0, rtol=1e-9)

    def test_renewable_share_exists(self):
        df = make_df(periods=50)
        fe = FeatureEngineer()
        df_fe = fe.fit_transform(df)
        assert "renewable_share" in df_fe.columns
        # renewable_share = (800 + 400) / 5000 = 0.24
        np.testing.assert_allclose(df_fe["renewable_share"].dropna(), 0.24, rtol=1e-9)


# ── Test: Cyclical features ────────────────────────────────────────────────

class TestCyclicalFeatures:
    def test_hour_sin_cos_exists(self):
        df = make_df(periods=48)
        fe = FeatureEngineer()
        df_fe = fe.fit_transform(df)
        assert "hour_sin" in df_fe.columns
        assert "hour_cos" in df_fe.columns

    def test_hour_sin_cos_bounded(self):
        df = make_df(periods=48)
        fe = FeatureEngineer()
        df_fe = fe.fit_transform(df)
        assert df_fe["hour_sin"].between(-1, 1).all()
        assert df_fe["hour_cos"].between(-1, 1).all()

    def test_cyclical_disabled(self):
        df = make_df(periods=48)
        fe = FeatureEngineer(include_cyclical=False)
        df_fe = fe.fit_transform(df)
        assert "hour_sin" not in df_fe.columns


# ── Test: DST flag ─────────────────────────────────────────────────────────

class TestDSTFlag:
    def test_dst_flag_exists(self):
        df = make_df(periods=100)
        fe = FeatureEngineer()
        df_fe = fe.fit_transform(df)
        assert "is_dst_spring" in df_fe.columns
        assert "is_dst_autumn" in df_fe.columns
        assert "is_dst_transition" in df_fe.columns

    def test_dst_spring_transition(self):
        # Last Sunday March 2025 = March 30
        df = make_df(periods=48, start="2025-03-29")
        fe = FeatureEngineer()
        df_fe = fe.fit_transform(df)
        # Should have at least some spring flag
        assert "is_dst_spring" in df_fe.columns


# ── Test: transform (without fit) raises ────────────────────────────────────

class TestTransformRaises:
    def test_transform_without_fit_raises(self):
        fe = FeatureEngineer()
        df = make_df(periods=50)
        with pytest.raises(ValueError, match="has not been fitted"):
            fe.transform(df)


# ── Test: engineer_features convenience function ───────────────────────────

class TestEngineerFeaturesFunction:
    def test_one_shot_works(self):
        df = make_df(periods=100)
        df_fe = engineer_features(df)
        assert df_fe.shape[1] > df.shape[1]


# ── Test: transform is consistent with fit ───────────────────────────────────

class TestTransformConsistency:
    def test_transform_does_not_relearn(self):
        df = make_df(periods=200)
        fe = FeatureEngineer()
        df_train = fe.fit_transform(df)

        # New data with same columns
        df_test = make_df(periods=50, start="2025-06-01")
        df_test_fe = fe.transform(df_test)

        # Should have the same new feature columns
        new_cols = set(df_train.columns) - set(df.columns)
        assert new_cols.issubset(set(df_test_fe.columns))


# ── Test: edge cases ───────────────────────────────────────────────────────

class TestEdgeCases:
    def test_short_df(self):
        df = make_df(periods=5)
        fe = FeatureEngineer()
        df_fe = fe.fit_transform(df)
        assert df_fe.shape[0] == 5

    def test_no_price_col_still_works(self):
        df = make_df(periods=50)
        df = df.drop(columns=["price_eur_mwh"])
        fe = FeatureEngineer()
        df_fe = fe.fit_transform(df)  # should not raise
        assert "hour" in df_fe.columns  # calendar features still added

    def test_no_weather_cols_still_works(self):
        df = make_df(periods=50)
        df = df.drop(columns=["temperature_2m_c", "wind_speed_100m_km_h"])
        fe = FeatureEngineer()
        df_fe = fe.fit_transform(df)  # should not raise
        assert "hour" in df_fe.columns

    def test_feature_names(self):
        df = make_df(periods=50)
        fe = FeatureEngineer()
        fe.fit(df)
        names = fe.feature_names()
        assert "hour" in names
        assert "price_lag_24h" in names
        assert "price_roll_mean_24h" in names


# ── Run ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
