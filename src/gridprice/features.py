"""
GridPrice — Feature Engineering Pipeline

Generates all features used by the forecasting models:

  Lag features      — price_T-24h, price_T-48h, price_T-168h, etc.
  Rolling stats     — 7-day and 30-day rolling mean / std of price
  Calendar features — hour, day-of-week, month, weekend, Greek holidays
  Cyclical encoding — sin/cos for hour and day-of-week (better for trees)
  Weather features  — temperature, wind, cloud, solar irradiance (raw + rolling)
  Grid features     — residual load = load - solar - wind
  DST flag          — clocks-forward / clocks-back transitions
  Open-Meteo prefix — detected automatically

The class follows a scikit-learn fit_transform / transform pattern so that
features computed on training data are applied consistently to new data.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


# ── Greek national holidays (Orthodox + secular) ────────────────────────
# Covers 2024-2030. Add more years as needed.
_GREEK_HOLIDAYS: Set[date] = {
    # 2024
    date(2024, 1, 1),   date(2024, 1, 6),   date(2024, 2, 12),  # Clean Monday
    date(2024, 3, 25),  date(2024, 3, 31),  date(2024, 5, 1),
    date(2024, 5, 3),   date(2024, 8, 15),  date(2024, 10, 28),
    date(2024, 12, 25), date(2024, 12, 26),
    # Orthodox Easter 2024 (May 5)
    date(2024, 5, 5),   date(2024, 5, 6),
    # 2025
    date(2025, 1, 1),   date(2025, 1, 6),   date(2025, 3, 3),   # Clean Monday
    date(2025, 3, 25),  date(2025, 4, 18),  date(2025, 4, 21),
    date(2025, 5, 1),   date(2025, 8, 15),  date(2025, 10, 28),  date(2025, 11, 17),
    date(2025, 12, 25), date(2025, 12, 26),
    # Orthodox Easter 2025 (Apr 20)
    date(2025, 4, 20),  date(2025, 4, 21),
    # 2026
    date(2026, 1, 1),   date(2026, 1, 6),   date(2026, 2, 23),  # Clean Monday
    date(2026, 3, 25),  date(2026, 4, 19),  date(2026, 4, 20),
    date(2026, 5, 1),   date(2026, 8, 15),  date(2026, 10, 28),  date(2026, 11, 17),
    date(2026, 12, 25), date(2026, 12, 26),
    # Orthodox Easter 2026 (Apr 19)
    date(2026, 4, 19),  date(2026, 4, 20),
    # 2027
    date(2027, 1, 1),   date(2027, 1, 6),   date(2027, 3, 15),  # Clean Monday
    date(2027, 3, 25),  date(2027, 5, 9),   date(2027, 5, 10),
    date(2027, 5, 1),   date(2027, 8, 15),  date(2027, 10, 28),  date(2027, 11, 17),
    date(2027, 12, 25), date(2027, 12, 26),
    # Orthodox Easter 2027 (May 9)
    date(2027, 5, 9),   date(2027, 5, 10),
    # 2028
    date(2028, 1, 1),   date(2028, 1, 6),   date(2028, 3, 6),   # Clean Monday
    date(2028, 3, 25),  date(2028, 4, 30),  date(2028, 5, 1),
    date(2028, 5, 1),   date(2028, 8, 15),  date(2028, 10, 28),  date(2028, 11, 17),
    date(2028, 12, 25), date(2028, 12, 26),
    # Orthodox Easter 2028 (Apr 30)
    date(2028, 4, 30),  date(2028, 5, 1),
    # 2029
    date(2029, 1, 1),   date(2029, 1, 6),   date(2029, 2, 19),  # Clean Monday
    date(2029, 3, 25),  date(2029, 4, 15),  date(2029, 4, 16),
    date(2029, 5, 1),   date(2029, 8, 15),  date(2029, 10, 28),  date(2029, 11, 17),
    date(2029, 12, 25), date(2029, 12, 26),
    # Orthodox Easter 2029 (Apr 15)
    date(2029, 4, 15),  date(2029, 4, 16),
    # 2030
    date(2030, 1, 1),   date(2030, 1, 6),   date(2030, 3, 11),  # Clean Monday
    date(2030, 3, 25),  date(2030, 5, 5),   date(2030, 5, 6),
    date(2030, 5, 1),   date(2030, 8, 15),  date(2030, 10, 28),  date(2030, 11, 17),
    date(2030, 12, 25), date(2030, 12, 26),
    # Orthodox Easter 2030 (May 5)
    date(2030, 5, 5),   date(2030, 5, 6),
}


# ── DST transition dates (CET/CEST change) ───────────────────────────────
# Last Sunday of March (02:00 → 03:00 CET→CEST, clock forward)
# Last Sunday of October (03:00 → 02:00 CEST→CET, clock back)
def _dst_transition_dates(year: int) -> Tuple[datetime, datetime]:
    """Return (spring_forward, autumn_back) for a given year."""
    import calendar
    mar = calendar.monthcalendar(year, 3)
    oct_ = calendar.monthcalendar(year, 10)
    # Last Sunday of March
    last_sun_mar = max(week[calendar.SUNDAY] for week in mar)
    # Last Sunday of October
    last_sun_oct = max(week[calendar.SUNDAY] for week in oct_)
    spring_forward = datetime(year, 3, last_sun_mar, 3, 0, 0)
    autumn_back = datetime(year, 10, last_sun_oct, 4, 0, 0)  # 4 = before fall-back
    return spring_forward, autumn_back


# ── Column-name autodetection helpers ────────────────────────────────────

def _detect_price_col(columns: pd.Index) -> Optional[str]:
    for col in columns:
        if "price" in col.lower():
            return str(col)
    return None


def _detect_load_col(columns: pd.Index) -> Optional[str]:
    for col in columns:
        if col.lower().startswith("load") and "forecast" not in col.lower():
            return str(col)
    return None


def _detect_gen_cols(columns: pd.Index, zone: str = "gr") -> Dict[str, str]:
    """Return {fuel: col_name} for solar, wind, other renewables."""
    found: Dict[str, str] = {}
    zone_prefix = f"gen_{zone.lower()}_"
    for col in columns:
        col_lower = col.lower()
        if not col_lower.startswith(zone_prefix):
            continue
        fuel = col_lower.replace(zone_prefix, "")
        if "solar" in fuel and "solar" not in found:
            found["solar"] = str(col)
        elif "wind" in fuel and "wind" not in found:
            found["wind"] = str(col)
        elif "gas" in fuel and "gas" not in found:
            found["gas"] = str(col)
        elif "hydro" in fuel and "hydro" not in found:
            found["hydro"] = str(col)
    return found


def _detect_weather_cols(columns: pd.Index) -> Dict[str, str]:
    """Detect Open-Meteo weather columns by common naming patterns."""
    found: Dict[str, str] = {}
    for col in columns:
        cl = col.lower()
        # Open-Meteo standard names
        if cl in ("temperature_2m_c", "temperature_2m", "temperature",
                  "temperature_c", "temp_c", "temperature_celsius"):
            found["temperature"] = str(col)
        elif cl in ("wind_speed_100m_km_h", "wind_speed_100m", "wind_speed_10m_ms",
                    "wind_speed", "wind_speed_ms"):
            found["wind_speed"] = str(col)
        elif cl in ("cloud_cover", "cloudcover", "cloud_fraction", "total_cloud_cover"):
            found["cloud_cover"] = str(col)
        elif cl in ("solar_radiation", "solar_radiance", "shortwave_radiation",
                    "solar_irradiance", "solar_irradiance_wm2"):
            found["solar_irradiance"] = str(col)
        elif cl in ("precipitation", "precip"):
            found["precipitation"] = str(col)
    return found


def _detect_flow_cols(columns: pd.Index) -> Optional[str]:
    for col in columns:
        if "flow" in col.lower() and "import" not in col.lower():
            return str(col)
    return None


# ── Core FeatureEngineer ─────────────────────────────────────────────────

class FeatureEngineer:
    """
    Generate features for electricity price forecasting.

    The class operates in two phases:

    fit_transform(X)  — computes per-row features on training data.
                        Also records column names and fit statistics
                        (rolling means/stds, holiday sets) for use in transform().

    transform(X)      — applies the same transformations to new data,
                        using the statistics learned in fit_transform().
                        Lag/rolling features that fall outside the observed
                        range are set to NaN.

    Parameters
    ----------
    lag_features : tuple of int
        Lag sizes in hours. Default (24, 48, 168, 336).
    rolling_windows : tuple of int
        Rolling window sizes in hours. Default (24, 168).
    holiday_years : tuple of int
        Which calendar years to include in the Greek holiday set.
        Default (2024, 2025, 2026, 2027, 2028, 2029, 2030).
    include_cyclical : bool
        If True (default), add sin/cos encoding for hour-of-day and
        day-of-week — these help tree-based models by encoding
        the circular nature of time.
    include_dst_flag : bool
        If True (default), flag hours that fall within ±1 day of a
        DST transition (clocks forward or backward).

    Notes
    -----
    Price and weather column names are auto-detected from the DataFrame
    columns. For Open-Meteo data the standard column names are recognised:
    ``temperature_2m_c``, ``wind_speed_100m_km_h``, ``cloud_cover``,
    ``solar_radiation``, ``precipitation``.

    The ``residual_load`` feature requires load, solar, and wind columns
    to be present. If any are missing it is silently skipped.

    Examples
    --------
    >>> from gridprice.features import FeatureEngineer
    >>> fe = FeatureEngineer()
    >>> X_train_fe = fe.fit_transform(df_train)
    >>> X_test_fe  = fe.transform(df_test)
    """

    def __init__(
        self,
        lag_features: Tuple[int, ...] = (24, 48, 168, 336),
        rolling_windows: Tuple[int, ...] = (24, 168),
        holiday_years: Tuple[int, ...] = (2024, 2025, 2026, 2027, 2028, 2029, 2030),
        include_cyclical: bool = True,
        include_dst_flag: bool = True,
    ) -> None:
        self.lag_features = lag_features
        self.rolling_windows = rolling_windows
        self.holiday_years = holiday_years
        self.include_cyclical = include_cyclical
        self.include_dst_flag = include_dst_flag

        # Fitted state — set in fit()
        self.price_col_: Optional[str] = None
        self.load_col_: Optional[str] = None
        self.gen_cols_: Dict[str, str] = {}
        self.weather_cols_: Dict[str, str] = {}
        self.flow_col_: Optional[str] = None
        self.zone_: str = "gr"

        # Rolling stats for price — used in transform()
        self.rolling_means_: Dict[int, float] = {}
        self.rolling_stds_: Dict[int, float] = {}

        # DST transition dates observed in training data
        self.dst_spring_dates_: Set[datetime] = set()
        self.dst_autumn_dates_: Set[datetime] = set()

        # Holiday dates observed in training data
        self.holiday_set_: Set[date] = set()

        # Columns that were present during fit
        self.fitted_columns_: List[str] = []

    # ── fit ─────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        """
        Learn column names and per-column statistics from the DataFrame.

        Does NOT add features — use fit_transform() to do both in one call.
        """
        cols = df.columns

        # Detect columns
        self.price_col_ = _detect_price_col(cols)
        self.load_col_ = _detect_load_col(cols)
        self.gen_cols_ = _detect_gen_cols(cols, zone=self.zone_)
        self.weather_cols_ = _detect_weather_cols(cols)
        self.flow_col_ = _detect_flow_cols(cols)

        # Rolling statistics for price
        if self.price_col_:
            price = df[self.price_col_]
            for w in self.rolling_windows:
                self.rolling_means_[w] = float(price.rolling(w, min_periods=1).mean().iloc[-1])
                self.rolling_stds_[w] = float(price.rolling(w, min_periods=1).std().iloc[-1])

        # DST transitions in the date range
        years_in_data = set(df.index.year)
        for year in self.holiday_years:
            if year in years_in_data or not years_in_data:
                spring, autumn = _dst_transition_dates(year)
                self.dst_spring_dates_.add(spring)
                self.dst_autumn_dates_.add(autumn)

        # Holiday set for this year's subset
        for year in self.holiday_years:
            for d in _GREEK_HOLIDAYS:
                if d.year == year:
                    self.holiday_set_.add(d)

        # Remember which columns were present (for transform alignment)
        self.fitted_columns_ = list(cols)
        return self

    # ── transform ────────────────────────────────────────────────────────

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature transformations to df.

        Uses column names and statistics learned in fit().
        Rows whose lag/rolling windows extend before the start of the
        fitted data will have those features set to NaN.
        """
        if not self.fitted_columns_:
            raise ValueError("FeatureEngineer has not been fitted. Call fit() or fit_transform() first.")

        result = df.copy()
        idx = pd.to_datetime(result.index)

        # ── 1. Calendar features ────────────────────────────────────────
        result = self._add_calendar_features(result, idx)

        # ── 2. Lag features ────────────────────────────────────────────
        result = self._add_lag_features(result)

        # ── 3. Rolling statistics ──────────────────────────────────────
        result = self._add_rolling_features(result)

        # ── 4. Weather features ────────────────────────────────────────
        result = self._add_weather_features(result)

        # ── 5. Grid features ───────────────────────────────────────────
        result = self._add_grid_features(result)

        # ── 6. DST flag ────────────────────────────────────────────────
        result = self._add_dst_flag(result, idx)

        # ── 7. Cyclical encoding ───────────────────────────────────────
        result = self._add_cyclical_features(result, idx)

        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience: fit() then transform() in one call."""
        self.fit(df)
        return self.transform(df)

    # ── Individual feature groups ────────────────────────────────────────

    def _add_calendar_features(
        self, result: pd.DataFrame, idx: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """Add hour, day-of-week, month, is_weekend, is_holiday."""
        result["hour"] = idx.hour
        result["day_of_week"] = idx.dayofweek          # 0=Mon, 6=Sun
        result["month"] = idx.month
        result["is_weekend"] = (idx.dayofweek >= 5).astype(int)

        # Greek holiday flag — work with tz-naive dates directly
        dates_naive = idx.normalize()  # DatetimeIndex (tz-naive after normalization)

        result["is_holiday"] = pd.Series(
            [ts.date() in self.holiday_set_ for ts in dates_naive],
            index=result.index,
            dtype=int,
        )

        # Is holiday Eve? (lower activity the day before major holidays)
        # Shift dates forward by 1 day using Timedelta
        eve_dates_naive = dates_naive + pd.Timedelta(days=1)
        result["is_holiday_eve"] = pd.Series(
            [ts.date() in self.holiday_set_ for ts in eve_dates_naive],
            index=result.index,
            dtype=int,
        )

        # Day of year (for seasonal signal)
        result["day_of_year"] = idx.dayofyear
        # Week of year
        result["week_of_year"] = idx.isocalendar().week.astype(int)

        # Part-of-day buckets (4 bins: night, morning, afternoon, evening)
        hour = idx.hour
        result["part_of_day"] = pd.cut(
            hour,
            bins=[-1, 6, 12, 18, 24],
            labels=[0, 1, 2, 3],          # 0=night, 1=morning, 2=afternoon, 3=evening
        ).astype(int)

        return result

    def _add_lag_features(self, result: pd.DataFrame) -> pd.DataFrame:
        """Add price lag features: T-24h, T-48h, T-168h, T-336h."""
        if not self.price_col_:
            return result

        price = result[self.price_col_]

        for lag in self.lag_features:
            col_name = f"price_lag_{lag}h"
            if col_name not in result.columns:
                result[col_name] = price.shift(lag)

        # Same-hour-last-week change
        if "price_lag_168h" in result.columns:
            prev_week = result["price_lag_168h"]
            result["price_weekly_delta"] = price - prev_week

        # Price return (log-change)
        if "price_lag_24h" in result.columns:
            prev = result["price_lag_24h"]
            safe_prev = prev.replace(0, np.nan)
            result["price_pct_change_24h"] = (price - safe_prev) / safe_prev.abs()

        return result

    def _add_rolling_features(self, result: pd.DataFrame) -> pd.DataFrame:
        """Add rolling mean and std of price over configured windows."""
        if not self.price_col_:
            return result

        price = result[self.price_col_]

        for w in self.rolling_windows:
            roll_mean = price.rolling(w, min_periods=1).mean()
            roll_std = price.rolling(w, min_periods=1).std()

            # Rolling mean — add as feature
            result[f"price_roll_mean_{w}h"] = roll_mean

            # Z-score relative to rolling window
            roll_std_safe = roll_std.replace(0, np.nan)
            result[f"price_roll_zscore_{w}h"] = (price - roll_mean) / roll_std_safe

        # Price level relative to 7-day rolling mean
        if "price_roll_mean_168h" in result.columns:
            result["price_vs_7d_mean"] = (
                price - result["price_roll_mean_168h"]
            )

        return result

    def _add_weather_features(self, result: pd.DataFrame) -> pd.DataFrame:
        """Add raw weather + rolling means."""
        for feat_name, col_name in self.weather_cols_.items():
            col = result[col_name]

            # Raw feature (keep original column)
            result[f"weather_{feat_name}_raw"] = col

            # Rolling 6-hour mean (smooths forecast noise)
            result[f"weather_{feat_name}_roll6h"] = (
                col.rolling(6, min_periods=1).mean()
            )

            # Rolling 24-hour mean
            result[f"weather_{feat_name}_roll24h"] = (
                col.rolling(24, min_periods=1).mean()
            )

            # Deviation from 24h rolling mean (captures temperature anomalies)
            result[f"weather_{feat_name}_anomaly"] = (
                col - result.get(f"weather_{feat_name}_roll24h", col)
            )

        # Solar duck-curve feature: estimated solar generation from irradiance
        if "solar_irradiance" in self.weather_cols_:
            irr_col = self.weather_cols_["solar_irradiance"]
            solar_estimated = result[irr_col] / 1000  # normalise 0-1
            result["solar_estimated_cf"] = solar_estimated.clip(0, 1)

        return result

    def _add_grid_features(self, result: pd.DataFrame) -> pd.DataFrame:
        """Compute residual load and net import features."""
        # Residual load = load - solar - wind
        if self.load_col_ and "solar" in self.gen_cols_ and "wind" in self.gen_cols_:
            load = result[self.load_col_]
            solar = result[self.gen_cols_["solar"]]
            wind = result[self.gen_cols_["wind"]]
            result["residual_load"] = load - solar - wind

            # Residual load percentile (contextualises current demand level)
            rl = result["residual_load"]
            result["residual_load_pct"] = rl.rank(pct=True)

            # Renewable penetration ratio
            renewables = solar + wind
            result["renewable_share"] = (renewables / load).clip(0, 2)

            # Solar/wind ratio to residual load
            result["solar_to_residual"] = (solar / rl.replace(0, np.nan)).clip(-2, 5)

        # Net import (flow column)
        if self.flow_col_:
            result["net_import"] = result[self.flow_col_]

        # Load-based features
        if self.load_col_:
            load = result[self.load_col_]
            result["load_roll24h"] = load.rolling(24, min_periods=1).mean()
            result["load_pct_of_7d"] = load / result["load_roll24h"].replace(0, np.nan)

        return result

    def _add_dst_flag(self, result: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
        """Flag hours near DST transitions."""
        if not self.include_dst_flag:
            return result

        result["is_dst_spring"] = 0
        result["is_dst_autumn"] = 0

        # Ensure DST transition dates are timezone-aware and match the index tz
        tz = idx.tz

        for spring_naive in self.dst_spring_dates_:
            spring = pd.Timestamp(spring_naive).tz_localize(tz) if tz else spring_naive
            for dt in idx:
                if abs((dt - spring).total_seconds()) <= 86400:
                    result.loc[result.index == dt, 'is_dst_spring'] = 1

        for autumn_naive in self.dst_autumn_dates_:
            autumn = pd.Timestamp(autumn_naive).tz_localize(tz) if tz else autumn_naive
            for dt in idx:
                if abs((dt - autumn).total_seconds()) <= 86400:
                    result.loc[result.index == dt, 'is_dst_autumn'] = 1

                result.loc[result.index == dt, "is_dst_autumn"] = 1

        # Combined DST flag
        result["is_dst_transition"] = (
            (result["is_dst_spring"] == 1) | (result["is_dst_autumn"] == 1)
        ).astype(int)

        return result

    def _add_cyclical_features(
        self, result: pd.DataFrame, idx: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """Add sin/cos encoding for hour-of-day and day-of-week."""
        if not self.include_cyclical:
            return result

        hour = idx.hour + idx.minute / 60  # fractional hour

        # Hour of week (0-167) — encodes both time-of-day AND day-of-week
        hour_of_week = idx.dayofweek * 24 + hour
        result["hour_of_week"] = hour_of_week

        # Cyclical hour-of-day
        result["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        result["hour_cos"] = np.cos(2 * np.pi * hour / 24)

        # Cyclical day-of-week
        dow = idx.dayofweek
        result["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        result["dow_cos"] = np.cos(2 * np.pi * dow / 7)

        # Cyclical day-of-year (seasonal)
        doy = idx.dayofyear
        result["doy_sin"] = np.sin(2 * np.pi * doy / 365)
        result["doy_cos"] = np.cos(2 * np.pi * doy / 365)

        # Cyclical month
        month = idx.month
        result["month_sin"] = np.sin(2 * np.pi * month / 12)
        result["month_cos"] = np.cos(2 * np.pi * month / 12)

        return result

    # ── Feature list ────────────────────────────────────────────────────

    def feature_names(self) -> List[str]:
        """
        Return the names of all features that this engineer produces.

        Only valid after fit() has been called.
        """
        if not self.fitted_columns_:
            raise ValueError("FeatureEngineer has not been fitted.")
        # Return a representative set based on configuration
        names: List[str] = []
        # Calendar
        names += [
            "hour", "day_of_week", "month", "is_weekend",
            "is_holiday", "is_holiday_eve", "day_of_year",
            "week_of_year", "part_of_day",
        ]
        # Cyclical
        if self.include_cyclical:
            names += [
                "hour_of_week", "hour_sin", "hour_cos",
                "dow_sin", "dow_cos",
                "doy_sin", "doy_cos",
                "month_sin", "month_cos",
            ]
        # Lag
        for lag in self.lag_features:
            names.append(f"price_lag_{lag}h")
        names += ["price_weekly_delta", "price_pct_change_24h"]
        # Rolling
        for w in self.rolling_windows:
            names += [f"price_roll_mean_{w}h", f"price_roll_zscore_{w}h"]
        names.append("price_vs_7d_mean")
        # Weather
        for feat in ["temperature", "wind_speed", "cloud_cover", "solar_irradiance", "precipitation"]:
            names += [f"weather_{feat}_raw", f"weather_{feat}_roll6h", f"weather_{feat}_roll24h"]
        names.append("solar_estimated_cf")
        # Grid
        names += ["residual_load", "residual_load_pct", "renewable_share",
                  "solar_to_residual", "net_import", "load_roll24h", "load_pct_of_7d"]
        # DST
        if self.include_dst_flag:
            names += ["is_dst_spring", "is_dst_autumn", "is_dst_transition"]
        return names

    # ── Pretty repr ────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"FeatureEngineer(lag_features={self.lag_features}, "
            f"rolling_windows={self.rolling_windows}, "
            f"include_cyclical={self.include_cyclical}, "
            f"include_dst_flag={self.include_dst_flag})"
        )


# ── Convenience functions ─────────────────────────────────────────────────

def engineer_features(
    df: pd.DataFrame,
    lag_features: Tuple[int, ...] = (24, 48, 168, 336),
    rolling_windows: Tuple[int, ...] = (24, 168),
    holiday_years: Tuple[int, ...] = (2024, 2025, 2026, 2027, 2028, 2029, 2030),
    include_cyclical: bool = True,
    include_dst_flag: bool = True,
) -> pd.DataFrame:
    """
    One-shot feature engineering.

    Creates and fits a FeatureEngineer on df and returns the transformed
    DataFrame with all features added.

    Use this when you don't need the fitted transformer for later use.
    For train/test pipelines, use the FeatureEngineer class directly.
    """
    fe = FeatureEngineer(
        lag_features=lag_features,
        rolling_windows=rolling_windows,
        holiday_years=holiday_years,
        include_cyclical=include_cyclical,
        include_dst_flag=include_dst_flag,
    )
    return fe.fit_transform(df)


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Demo with synthetic data
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from gridprice.synthetic_data import SyntheticDataGenerator
    from datetime import date

    gen = SyntheticDataGenerator(
        bidding_zone="GR",
        seed=42,
        end_date=date(2026, 3, 1),
    )
    df = gen.generate()
    print(f"Input shape: {df.shape}")

    fe = FeatureEngineer()
    df_fe = fe.fit_transform(df)

    # Show new columns
    original_cols = set(df.columns)
    new_cols = [c for c in df_fe.columns if c not in original_cols]
    print(f"\nNew features added ({len(new_cols)}):")
    for c in sorted(new_cols):
        print(f"  {c}")

    # Show a sample
    print(f"\nOutput shape: {df_fe.shape}")
    print(df_fe[new_cols[:8]].tail(3))
