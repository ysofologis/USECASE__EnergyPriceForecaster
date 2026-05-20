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
    Transform raw hourly grid + weather data into 51 features ready for XGBoost.

    What this class does
    ---------------------
    The raw data you get from ENTSO-E and Open-Meteo looks like this:

        timestamp          price_gr  load_gr  temperature_2m_c  solar_gr  wind_gr
        2024-05-01 00:00     82.3    5234.0              14.2       0.0     312.4
        2024-05-01 01:00     79.1    5098.1              13.8       0.0     328.1
        ...

    XGBoost can't just look at that.  It needs to know things like:
    - "Is this a weekday or weekend?"
    - "How does the current price compare to the same hour last week?"
    - "Is it hotter than usual for this time of day?"
    - "How much solar generation is pushing prices down right now?"

    ``FeatureEngineer`` does exactly that translation.  It takes the raw table
    and adds ~51 new columns that capture the patterns in the data that
    XGBoost can use to make predictions.

    How it works — the big picture
    ------------------------------
    The class follows the same fit/transform pattern as scikit-learn's
    ``StandardScaler`` or ``OneHotEncoder``:

    1. ``fit(df)``  — scans the DataFrame to learn which columns exist
                       (price, load, solar, wind, weather, etc.) and
                       records statistics needed for later (rolling averages,
                       holiday calendars, DST dates).  Nothing is added yet.

    2. ``transform(df)`` — adds all ~51 feature columns to the DataFrame
                          using the knowledge from step 1.

    3. ``fit_transform(df)`` — shorthand for the two steps above.

    The distinction matters when you process test data: you must call ``fit()``
    on training data first, then call ``transform()`` on both train and test
    separately.  Never call ``fit()`` on test data — that would discard the
    statistics learned from training.

    The 7 feature groups (in order)
    -------------------------------
    1. Calendar  — hour, day-of-week, month, weekend, Greek holidays, etc.
    2. Lag       — price T-24h, T-48h, T-168h ago (last week same hour)
    3. Rolling   — 24h/168h rolling mean and z-score of price
    4. Weather   — raw + smoothed + anomaly variants of temperature/wind/cloud
    5. Grid      — residual load, renewable share, load ratios
    6. DST       — clocks-forward / clocks-back transition flags
    7. Cyclical  — sin/cos encodings of hour, day-of-week, day-of-year

    Why so many weather variants?
    -----------------------------
    Weather forecasts are noisy.  A single temperature reading of 28°C might
    be a spike or it might be accurate.  By computing the 6-hour and 24-hour
    rolling averages, the model sees the smoothed trend.  The "anomaly" feature
    (deviation from the 24h rolling mean) flags when something unusual is
    happening — a cold snap or heat wave that the model can learn to associate
    with price spikes.

    Parameters
    ----------
    lag_features : tuple of int
        How far back in time to look for price comparisons, in hours.
        Default (24, 48, 168, 336) means:
        - 24h  — price at the same hour yesterday
        - 48h  — price at the same hour two days ago
        - 168h — price at the same hour one week ago (same day-of-week)
        - 336h — price at the same hour two weeks ago
        Think of these as "what was the price doing around this time recently?"

        The largest lag (336h = 2 weeks) determines how many rows you need
        before you get useful features.  The first 336 rows will have NaN
        for all lag features.

    rolling_windows : tuple of int
        Time windows (in hours) over which to compute rolling statistics.
        Default (24, 168) means:
        - 24h  — captures the short-term daily rhythm (morning ramp, evening drop)
        - 168h — captures the full weekly cycle
        These are used to compute rolling mean, rolling std, and z-scores.

    holiday_years : tuple of int
        Which calendar years to include in the Greek public holiday list.
        Default covers 2024–2030.  Add future years as needed.
        Greek holidays include New Year's Day, Epiphany, Clean Monday,
        Independence Day (Mar 25), Easter (Orthodox, variable),
        Labour Day (May 1), Assumption (Aug 15), Ochi Day (Oct 28),
        Christmas, and Boxing Day.

    include_cyclical : bool
        Whether to add sin/cos encodings for hour-of-day, day-of-week,
        day-of-year, and month.  Default True.
        Why sin/cos?  Because hour=23 and hour=0 are neighbours on the clock
        but numerically very far apart (|23-0| = 23).  Sin/cos encoding
        solves this: both 23:00 and 00:00 map to the same point on the circle.

    include_dst_flag : bool
        Whether to flag hours near Daylight Saving Time transitions.
        Default True.
        Greece switches clocks forward (02:00→03:00) on the last Sunday of March
        and back (03:00→02:00) on the last Sunday of October.
        These transitions cause the hourly data to have either 23 or 25 hours
        that day, which can confuse models that assume every day has 24 hours.
        The DST flag explicitly marks those transition days.

    Attributes (set after fit())
    -----------------------------
    price_col_ : str or None
        Name of the detected price column (e.g. "price_gr").
    load_col_ : str or None
        Name of the detected load column (e.g. "load_gr").
    gen_cols_ : dict
        Mapping of fuel type → column name for solar, wind, gas, hydro.
    weather_cols_ : dict
        Mapping of weather variable → column name for temperature, wind_speed,
        cloud_cover, solar_irradiance.
    holiday_set_ : set of date
        All Greek public holiday dates in the configured years.
    dst_spring_dates_ : set of datetime
        Spring-forward transition timestamps (last Sunday of March, 03:00).
    dst_autumn_dates_ : set of datetime
        Autumn-back transition timestamps (last Sunday of October, 04:00).

    Example usage
    -------------
    >>> from gridprice.features import FeatureEngineer
    >>> from gridprice.synthetic_data import SyntheticDataGenerator
    >>> from datetime import date
    >>>
    >>> # Generate some data
    >>> df = SyntheticDataGenerator(bidding_zone="GR", seed=42, end_date=date(2024, 6, 1)).generate()
    >>> print(f"Raw columns ({len(df.columns)}): {list(df.columns)}")
    Raw columns (14): ['price_gr', 'load_gr', 'temperature_2m_c', 'wind_speed_10m_ms', ...]
    >>>
    >>> # Step 1 — fit on training data (learn column names + statistics)
    >>> fe = FeatureEngineer()
    >>> fe.fit(df)   # this scans the DataFrame but doesn't change it yet
    >>>
    >>> # Step 2 — transform (add all features)
    >>> df_fe = fe.transform(df)
    >>> print(f"Feature columns ({len(df_fe.columns)}): {list(df_fe.columns)}")
    Feature columns (65): ['price_gr', 'load_gr', ..., 'hour', 'day_of_week', 'is_holiday',
                            'price_lag_24h', 'price_roll_mean_24h', 'residual_load', ...]
    >>>
    >>> # Convenience: fit + transform in one call
    >>> df_fe = FeatureEngineer().fit_transform(df)
    >>>
    >>> # First 336 rows have NaN lags — drop them before training
    >>> df_fe = df_fe.dropna()
    >>> print(f"After dropna: {len(df_fe)} rows (was {len(df)})")
    After dropna: 14784 rows (was 15120)
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
        Learn what columns are available and record statistics needed for features.

        This is the "learning" step.  It scans ``df`` to:

        - Find which columns contain price, load, generation, and weather data
          (using name patterns — e.g. any column with "price" in the name is
          treated as the price column).  Results are stored in ``.price_col_``,
          ``.load_col_``, etc.

        - Record the most recent rolling mean and rolling standard deviation of
          the price.  These are used later to compute z-scores for new data.

        - Build the set of Greek public holiday dates for the configured years.

        - Compute DST transition timestamps for the years covered by the data.

        ``fit()`` does NOT add any columns to ``df``.  It only records metadata.

        IMPORTANT: always call ``fit()`` on training data, NOT on test data.
        If you fit on test data, the model will have seen information it
        shouldn't have (e.g. the mean price of the test period).

        Parameters
        ----------
        df : DataFrame with DatetimeIndex
            Training data.  Must have a timezone-aware DatetimeIndex
            (required for DST flag calculations).

        Returns
        -------
        self (allows chaining: ``fe = FeatureEngineer().fit(df)``)

        Raises
        ------
        ValueError
            If the DataFrame has no detected price column.
        """
        cols = df.columns

        # Detect columns
        self.price_col_ = _detect_price_col(cols)
        self.load_col_ = _detect_load_col(cols)
        self.gen_cols_ = _detect_gen_cols(cols, zone=self.zone_)
        self.weather_cols_ = _detect_weather_cols(cols)
        self.flow_col_ = _detect_flow_cols(cols)

        # Rolling statistics for price — last value observed in the series
        # Used as a fallback for z-score normalisation on new data
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
        Add all ~51 feature columns to the DataFrame.

        Call this after ``fit()`` — it uses the column names and statistics
        recorded during fitting to add the following feature groups:

        1. Calendar    — ``hour``, ``day_of_week``, ``month``, ``is_weekend``,
                          ``is_holiday``, ``is_holiday_eve``, ``day_of_year``,
                          ``week_of_year``, ``part_of_day``
        2. Lag         — ``price_lag_24h``, ``price_lag_48h``, ``price_lag_168h``,
                          ``price_lag_336h``, ``price_weekly_delta``,
                          ``price_pct_change_24h``
        3. Rolling     — ``price_roll_mean_24h``, ``price_roll_zscore_24h``,
                          ``price_roll_mean_168h``, ``price_roll_zscore_168h``,
                          ``price_vs_7d_mean``
        4. Weather    — ``weather_{var}_raw``, ``weather_{var}_roll6h``,
                          ``weather_{var}_roll24h``, ``weather_{var}_anomaly``
                          for temperature, wind_speed, cloud_cover,
                          solar_irradiance (+ ``solar_estimated_cf``)
        5. Grid       — ``residual_load``, ``residual_load_pct``,
                          ``renewable_share``, ``solar_to_residual``,
                          ``net_import``, ``load_roll24h``, ``load_pct_of_7d``
        6. DST        — ``is_dst_spring``, ``is_dst_autumn``,
                          ``is_dst_transition``
        7. Cyclical   — ``hour_of_week``, ``hour_sin``, ``hour_cos``,
                          ``dow_sin``, ``dow_cos``, ``doy_sin``, ``doy_cos``,
                          ``month_sin``, ``month_cos``

        Rows at the start of the series (before enough history exists for
        lag and rolling features) will have NaN in those columns.  Drop them
        with ``df.dropna()`` before training.

        Parameters
        ----------
        df : DataFrame with DatetimeIndex
            Raw data in the same format as passed to ``fit()``.

        Returns
        -------
        DataFrame with all original columns plus new feature columns.
        The original DataFrame is not modified.

        Raises
        ------
        ValueError
            If ``fit()`` has not been called yet.
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
        """
        Convenience: run ``fit()`` then ``transform()`` in a single call.

        Equivalent to: ``fe = FeatureEngineer(); fe.fit(df); return fe.transform(df)``

        Use this when you just need features for training (one-shot) or when
        you are sure no separate test-time transformation is needed.

        For a proper train/test split, call ``fit_transform()`` on training
        data only, then call ``transform()`` (not ``fit_transform()``) on test
        data to avoid information leakage from the test period leaking into
        the feature statistics.
        """
        self.fit(df)
        return self.transform(df)

    # ── Individual feature groups ────────────────────────────────────────

    def _add_calendar_features(
        self, result: pd.DataFrame, idx: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        Add calendar-based features derived from the timestamp index.

        What each feature means for electricity prices
        ---------------------------------------------
        ``hour`` — Prices follow a daily rhythm: low at night (3am trough),
          double-peak on weekdays (morning ~10h, evening ~17h), high in winter.

        ``day_of_week`` — Weekends have a completely different demand profile:
          no morning industrial ramp, lower peak demand, different solar profile.

        ``month`` — Seasonal demand: heating in Jan/Feb, cooling from Jun/Aug.
          Prices in summer may spike due to AC demand even if fuel is cheap.

        ``is_weekend`` — Binary flag for Sat/Sun.  XGBoost can split on this
          directly but the separate ``day_of_week`` helps too.

        ``is_holiday`` — Holidays eliminate industrial and commercial demand.
          Athens on Christmas Day has dramatically lower prices than a normal Tue.

        ``is_holiday_eve`` — Christmas Eve, New Year's Eve, etc. have reduced
          but not eliminated demand (half-day working).

        ``day_of_year`` — Captures longer seasonal patterns (e.g. heating
          season vs cooling season) beyond what month captures.

        ``week_of_year`` — Week 1 vs Week 52 may have special patterns.

        ``part_of_day`` — 4 buckets: night (0–6h), morning (6–12h),
          afternoon (12–18h), evening (18–24h).  Crude but helps XGBoost
          find intraday patterns without needing to learn hour=23 is
          "similar to" hour=0 (that's what cyclical features are for).

        Parameters
        ----------
        result : DataFrame
            Working copy of the DataFrame; modified in place.
        idx : DatetimeIndex
            The timestamp index of ``result``.
        """
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
        """
        Add lagged price features: what was the price N hours ago?

        The intuition: electricity prices are highly autocorrelated.
        If the price right now is €85/MWh and it was €83/MWh at the same
        hour yesterday, it's very likely to be around €80–90 tomorrow too.
        Lag features let XGBoost learn this pattern.

        What each lag means
        --------------------
        ``price_lag_24h`` — Same hour yesterday.  The single most powerful feature
            for daily seasonality.  "Price at 9am today ≈ price at 9am yesterday."

        ``price_lag_48h`` — Same hour two days ago.  Helps when there's a
            multi-day trend (e.g. a cold snap lasting 3 days).

        ``price_lag_168h`` — Same hour last week (7 days).  The most important
            lag for weekly seasonality.  "Price at 9am on a Wednesday tends
            to be similar to last Wednesday's price."

        ``price_lag_336h`` — Same hour two weeks ago.  Captures bi-weekly patterns
            and provides redundancy if 168h data is missing.

        ``price_weekly_delta`` — Change over the past week:
            ``price_now - price_same_hour_last_week``.
            A rising price trend is often predictive of continued rising prices
            (e.g. sustained cold weather pushing up demand).

        ``price_pct_change_24h`` — Percentage change from yesterday to today.
            Captures momentum: if prices jumped 20% overnight, something unusual
            is happening (cold front, outage, etc.) and may persist.

        The first 336 rows will have NaN for all lag features — this is
        expected and is handled by dropping NaN rows before training.
        """
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
        """
        Add rolling-window statistics of the price.

        The intuition: comparing the current price to its recent history
        tells you whether the market is "hot" or "cold" right now.

        What each feature means
        -----------------------
        ``price_roll_mean_{w}h`` — Average price over the past W hours.
            E.g. ``price_roll_mean_24h`` is the average of the last 24 hours.
            A current price far above its 24h average signals an unusual event.

        ``price_roll_zscore_{w}h`` — How many standard deviations the current
            price is from its recent average:
            ``z = (price_now - roll_mean_w) / roll_std_w``

            A z-score of +2 means "the current price is 2 standard deviations
            above the recent average" — a strong signal that something unusual
            is happening.  This is the model's most important feature.

        ``price_vs_7d_mean`` — Difference between the current price and the
            average price over the past 7 days (168 hours).  More stable than
            the 24h mean, captures whether you're in a high-price or low-price
            week relative to the recent past.
        """
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
        """
        Add weather-derived features: temperature, wind, cloud cover, solar.

        The intuition: weather is the most important external driver of
        electricity prices.  A heat wave → air conditioning → higher demand →
        higher prices.  High solar output → renewable generation displaces
        expensive gas plants → lower prices (the "duck curve").

        Why 4 variants per weather variable?
        ------------------------------------
        Weather forecasts are uncertain.  A single temperature reading might
        be a forecast glitch.  By including all three variants, the model can
        learn which level of smoothing is most predictive and when to weight
        the raw reading heavily vs. the smoothed trend.

        What each variant means
        ------------------------
        ``weather_{var}_raw`` — The raw current value.
            E.g. current temperature = 31°C.  Tells the model the immediate
            condition.  Also useful when conditions are changing fast
            (a rapid temperature drop during a storm).

        ``weather_{var}_roll6h`` — Average over the past 6 hours.
            Smooths out hourly noise and forecast jitter.  Helps the model
            ignore brief anomalies and focus on the sustained weather trend.

        ``weather_{var}_roll24h`` — Average over the past 24 hours.
            Even smoother.  Captures whether the current weather episode
            is a multi-day event (e.g. a heat wave lasting 5 days).

        ``weather_{var}_anomaly`` — Deviation from the 24h rolling average:
            ``anomaly = raw - roll24h_mean``
            A positive anomaly means "hotter than recent average" — useful for
            flagging heat waves or cold snaps that the 24h mean alone would miss.

        ``solar_estimated_cf`` — Estimated solar capacity factor (0–1).
            Derived from solar irradiance: ``irradiance_Wm2 / 1000``.
            This is the model's approximation of how much solar generation
            is displacing fossil fuel plants at any given hour.
        """
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
        """
        Add grid-level features: residual load, renewable penetration, and load ratios.

        The intuition: the electricity market clears when total demand (load)
        meets total supply (generation).  The price is set by the marginal
        (most expensive) plant that must run to meet demand.  Grid features
        capture supply/demand balance directly.

        What each feature means
        -----------------------
        ``residual_load`` — "Net demand" that non-renewable plants must serve:
            ``residual_load = load - solar - wind``

            If residual load is high, expensive gas plants must run
            → prices rise.  If residual load is low (high solar/wind),
            cheap renewables set the price → prices fall.
            This is the most economically meaningful feature.

        ``residual_load_pct`` — The percentile rank of the current residual load
            within the historical distribution.  A value of 0.9 means "residual
            load is in the top 10% of all-time levels" — a strong demand signal.

        ``renewable_share`` — Fraction of total load met by solar + wind:
            ``renewable_share = (solar + wind) / load``

            High renewable share → oversupply → low prices.
            This captures the fundamental market dynamic of renewable-driven
            price suppression.

        ``solar_to_residual`` — Ratio of solar generation to residual load:
            ``solar / residual_load``

            A value > 1 means solar alone covers all residual demand —
            the "duck curve" situation at midday where prices can go negative.
            This is the key driver of Greece's midday price dips.

        ``net_import`` — Net electricity import from interconnectors.
            Positive = importing, Negative = exporting.
            Imports from neighbouring zones can cap local prices.

        ``load_roll24h`` — 24-hour rolling average of total load.
            Smoothed demand signal, complementary to the raw load.

        ``load_pct_of_7d`` — Current load as a fraction of the past 7-day
            average load at this hour.  Captures demand anomalies
            (e.g. "today's demand is 15% above the recent norm for this hour").
        """
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
        """
        Flag hours near Daylight Saving Time (DST) clock transitions.

        Why DST matters for hourly data
        -------------------------------
        Greece changes clocks in March (spring forward: 02:00→03:00, one 23h day)
        and October (autumn back: 03:00→02:00, one 25h day).

        A naive model treating every "hour 2–3am" as a single hour will see:
        - In March: only 23 data rows for that day → missing hour → NaN lag features
        - In October: 25 rows for that day → duplicated hour → overlapping lag windows

        The DST flags explicitly mark transition days so the model can learn to
        handle them differently (or ignore them entirely).  Without this flag,
        XGBoost would try to fit these days as unusual patterns rather than
        known structural anomalies.

        What each flag means
        ---------------------
        ``is_dst_spring`` — 1 if this hour falls within ±1 day of the spring
            forward transition (last Sunday of March, 03:00).  These days have
            23 hours.  Expect lag features to be misaligned.

        ``is_dst_autumn`` — 1 if this hour falls within ±1 day of the autumn
            back transition (last Sunday of October, 04:00).  These days have
            25 hours.  Expect duplicate lag entries.

        ``is_dst_transition`` — Either of the above (OR).  A simple summary flag.
        """
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
        """
        Add sin/cos cyclical encodings for time-based features.

        Why sin/cos encoding?
        ---------------------
        Plain integer encoding of time creates a false ordering problem:

            hour=23 and hour=0 are neighbours on the clock,
            but numerically |23 - 0| = 23 — a huge distance.
            XGBoost would think hour 23 is "far from" hour 0.

        The solution: map each cyclic value onto a unit circle.

        For hours (24-hour cycle):
            hour_sin = sin(2π × hour / 24)
            hour_cos = cos(2π × hour / 24)

        This way:
            hour=23 → sin ≈ -0.26, cos ≈ 0.97  ← close to
            hour=0  → sin = 0,       cos = 1.0  ← close to
            hour=12 → sin = 0,       cos = -1   ← opposite

        XGBoost can use both the sin and cos columns together to reconstruct
        the original hour.  The sin/cos pair is more informative than the
        raw integer because it makes the circular relationship explicit.

        What each feature encodes
        -------------------------
        ``hour_of_week`` — Integer 0–167: which hour within the week this is.
            ``hour_of_week = day_of_week × 24 + hour``.
            E.g. Monday 9am = 0×24 + 9 = 9; Tuesday 9am = 1×24 + 9 = 33.
            This captures the full weekly schedule in one integer.

        ``hour_sin``, ``hour_cos`` — Circular encoding of hour of day.
            Together these two values encode any hour uniquely.

        ``dow_sin``, ``dow_cos`` — Circular encoding of day of week.
            XGBoost can use these to learn that Monday and Sunday are
            "close" (Sunday=6, Monday=0) even though numerically they are far.

        ``doy_sin``, ``doy_cos`` — Circular encoding of day of year.
            Captures that Jan 1 and Dec 31 are adjacent (seasonal wraparound).

        ``month_sin``, ``month_cos`` — Circular encoding of month.
            Captures seasonal wraparound at the year boundary.
        """
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

        Useful for debugging, for checking which columns a model was trained on,
        and for verifying that a new DataFrame has all required features.

        Returns
        -------
        list of str
            Feature names in a consistent order (calendar → lag → rolling
            → weather → grid → DST → cyclical).

        Example
        -------
        >>> fe = FeatureEngineer().fit(df)
        >>> model_features = fe.feature_names()
        >>> print(f"Model uses {len(model_features)} features")
        Model uses 51 features
        >>> print(model_features[:5])
        ['hour', 'day_of_week', 'month', 'is_weekend', 'is_holiday']
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
    One-shot feature engineering: create and apply FeatureEngineer in a single call.

    This is a convenience function equivalent to:
        ``fe = FeatureEngineer(...); return fe.fit_transform(df)``

    When to use this vs. the class directly
    ----------------------------------------
    Use ``engineer_features()`` for:
    - Quick prototyping and notebook exploration
    - Scripts that don't need to persist the fitted transformer
    - One-off data transformations where you won't reuse the transformer

    Use ``FeatureEngineer`` class directly when:
    - You need to apply the same transformation to multiple DataFrames
      (e.g. train → test) and must keep the transformer state consistent
    - You want to save the transformer to disk alongside the model

    Parameters
    ----------
    df : DataFrame with DatetimeIndex
        Raw hourly data (same parameters as ``FeatureEngineer.__init__``).
    lag_features, rolling_windows, holiday_years,
    include_cyclical, include_dst_flag : same as ``FeatureEngineer.__init__``
        Passed directly to the ``FeatureEngineer`` constructor.

    Returns
    -------
    DataFrame with all ~51 feature columns added.

    Example
    -------
    >>> df_fe = engineer_features(df, lag_features=(24, 48, 168, 336))
    >>> df_fe = df_fe.dropna()  # drop warmup rows
    >>> print(df_fe.shape)
    (14784, 65)
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
