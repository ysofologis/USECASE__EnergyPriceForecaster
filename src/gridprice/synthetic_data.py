"""
GridPrice — Synthetic Data Generator

Produces realistic synthetic data mimicking the Greek electricity market
for development and testing until the ENTSO-E API key arrives.

Data characteristics modelled:
- Daily price pattern (low at night, peaks at midday and evening)
- Weekly pattern (lower weekend prices)
- Seasonal pattern (higher summer due to AC load, mild winter)
- Weather-driven effects (solar generation depresses midday prices)
- Random price spikes (transmission constraints, forced outages)
- Occasional negative prices (high solar + low demand)
- Correlation: temperature → load → price
"""

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class SyntheticDataGenerator:
    """
    Generate realistic synthetic electricity market data.

    Parameters
    ----------
    bidding_zone : str
        Zone label (e.g. "GR" for Greece)
    seed : int
        Random seed for reproducibility
    start_date : date
        First day of generated data
    end_date : date
        Last day of generated data (inclusive)
    """

    # Typical Greek base load patterns (MW) — rough order of magnitude
    BASE_LOAD_WEEKDAY = 4500  # MW
    BASE_LOAD_WEEKEND = 4000  # MW

    # Typical Greek renewable capacity (MW)
    SOLAR_CAPACITY = 5500   # MW installed
    WIND_CAPACITY = 5000    # MW installed

    def __init__(
        self,
        bidding_zone: str = "GR",
        seed: int = 42,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ):
        self.zone = bidding_zone
        self.rng = np.random.default_rng(seed)

        # Default: last 2 years
        if end_date is None:
            end_date = date.today()
        if start_date is None:
            start_date = end_date - timedelta(days=730)  # ~2 years

        self.start_date = start_date
        self.end_date = end_date

    # ── Weather ──────────────────────────────────────────────────────────

    def _generate_weather(self, timestamps: pd.DatetimeIndex) -> pd.DataFrame:
        """Generate synthetic hourly weather data for Athens/GR."""

        hours = np.arange(len(timestamps))
        # Temperature: seasonal sine + diurnal + noise
        # Greek summer ~35°C, winter ~10°C
        day_of_year = timestamps.dayofyear.to_numpy()
        seasonal_temp = 22 + 12 * np.sin(2 * np.pi * (day_of_year - 100) / 365)
        diurnal_temp = -4 * np.cos(2 * np.pi * timestamps.hour.to_numpy() / 24)
        temp = seasonal_temp + diurnal_temp + self.rng.normal(0, 2, len(timestamps))

        # Wind speed: higher in winter, random diurnal
        wind_base = 4 + 2 * np.sin(2 * np.pi * (day_of_year - 60) / 365)
        wind_diurnal = 1.5 * np.cos(2 * np.pi * (timestamps.hour.to_numpy() - 14) / 24)
        wind = wind_base + wind_diurnal + self.rng.exponential(1.5, len(timestamps))
        wind = np.clip(wind, 0, 25)

        # Cloud cover: 0 (clear) to 1 (overcast) — seasonal + random
        cloud_base = 0.3 + 0.2 * np.sin(2 * np.pi * (day_of_year - 350) / 365)
        cloud = np.clip(cloud_base + self.rng.beta(1, 3, len(timestamps)), 0, 1)

        # Solar irradiance (W/m²): only during daylight
        solar_irradiance = 1000 * np.maximum(
            0,
            np.cos(2 * np.pi * (timestamps.hour.to_numpy() - 12) / 24),
        ) * (1 - 0.6 * cloud) * (
            1 + 0.3 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        )
        solar_irradiance = np.clip(solar_irradiance, 0, 1000)

        return pd.DataFrame({
            "temperature_c": np.round(temp, 1),
            "wind_speed_ms": np.round(wind, 1),
            "cloud_cover": np.round(cloud, 2),
            "solar_irradiance_wm2": np.round(solar_irradiance, 0),
        }, index=timestamps)

    # ── Generation ───────────────────────────────────────────────────────

    def _generate_generation(
        self, timestamps: pd.DatetimeIndex, weather: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate synthetic generation by fuel type (MW)."""

        hours = np.arange(len(timestamps))
        day_of_year = timestamps.dayofyear.to_numpy()
        is_weekend = timestamps.dayofweek >= 5
        hour_of_day = timestamps.hour.to_numpy()

        # Solar: depends on irradiance + capacity factor
        solar_cf = weather["solar_irradiance_wm2"].values / 1000
        solar_noise = self.rng.normal(0, 0.03, len(timestamps))
        solar = self.SOLAR_CAPACITY * np.clip(solar_cf + solar_noise, 0, 0.85)
        solar = np.round(solar, 0)

        # Wind: depends on wind speed (cubic relationship capped)
        wind_cf = 0.4 * np.minimum(weather["wind_speed_ms"].values / 12, 1.0)
        wind_noise = self.rng.beta(1, 3, len(timestamps)) * 0.1
        wind = self.WIND_CAPACITY * np.clip(wind_cf + wind_noise, 0, 0.95)
        wind = np.round(wind, 0)

        # Gas: baseload + peaking — fills the gap
        # Higher when renewables are low
        gas = 1800 - 0.3 * solar - 0.2 * wind + self.rng.normal(0, 100, len(timestamps))
        # Evening peak
        gas += 300 * np.exp(-((hour_of_day - 20) ** 2) / 8)
        gas = np.clip(gas, 200, 3000)
        gas = np.round(gas, 0)

        # Lignite (coal): base generation, slowly declining
        lignite = 800 + self.rng.normal(0, 50, len(timestamps))
        lignite = np.clip(lignite, 300, 1200)

        # Hydro: flexible, used for balancing
        hydro = 300 + 100 * np.sin(2 * np.pi * (hour_of_day - 8) / 24)
        hydro += 50 * np.sin(2 * np.pi * (day_of_year - 120) / 365)  # more in spring
        hydro += self.rng.normal(0, 30, len(timestamps))
        hydro = np.clip(hydro, 50, 800)

        return pd.DataFrame({
            f"gen_{self.zone.lower()}_solar": solar,
            f"gen_{self.zone.lower()}_wind_onshore": wind,
            f"gen_{self.zone.lower()}_gas": gas,
            f"gen_{self.zone.lower()}_lignite": lignite,
            f"gen_{self.zone.lower()}_hydro": hydro,
        }, index=timestamps)

    # ── Load ─────────────────────────────────────────────────────────────

    def _generate_load(
        self, timestamps: pd.DatetimeIndex, weather: pd.DataFrame
    ) -> pd.Series:
        """Generate synthetic total load (MW)."""

        hour = timestamps.hour.to_numpy()
        day_of_week = timestamps.dayofweek.to_numpy()
        day_of_year = timestamps.dayofyear.to_numpy()
        is_weekend = day_of_week >= 5

        # Base load (float for accumulation)
        load = np.where(is_weekend, self.BASE_LOAD_WEEKEND, self.BASE_LOAD_WEEKDAY)
        load = load.astype(float)

        # Diurnal pattern (peak ~11:00 and ~20:00, trough ~04:00)
        diurnal = (
            800 * np.sin(np.pi * (hour - 4) / 14) * (hour >= 4) * (hour < 18)
            + 600 * np.sin(np.pi * (hour - 18) / 10) * (hour >= 18)
        )
        load += diurnal

        # Temperature effect: AC in summer, heating in winter
        temp = weather["temperature_c"].values
        load += 50 * np.maximum(0, temp - 28)  # AC load above 28°C
        load += 40 * np.maximum(0, 8 - temp)   # Heating below 8°C

        # Seasonal: higher summer (AC), lower spring/autumn
        seasonal = 200 * np.sin(2 * np.pi * (day_of_year - 180) / 365)
        load += seasonal

        # Random noise
        load += self.rng.normal(0, 80, len(timestamps))

        load = np.round(np.clip(load, 2000, 9000), 0)
        return pd.Series(load, index=timestamps, name=f"load_{self.zone.lower()}")

    # ── Price ────────────────────────────────────────────────────────────

    def _generate_price(
        self, timestamps: pd.DatetimeIndex, load: pd.Series, gen: pd.DataFrame
    ) -> pd.Series:
        """
        Generate synthetic day-ahead prices (EUR/MWh).

        Price is driven by:
        - residual load (load - solar - wind)
        - gas price as marginal fuel
        - scarcity pricing (high residual load → price spikes)
        - negative prices when solar is abundant and demand is low
        """

        residual_load = (
            load.values
            - gen[f"gen_{self.zone.lower()}_solar"].values
            - gen[f"gen_{self.zone.lower()}_wind_onshore"].values
        )

        # Gas price proxy (EUR/MWh) — recent historical range ~20-80
        gas_price = 40 + 15 * np.sin(2 * np.pi * timestamps.dayofyear.to_numpy() / 365)
        gas_price += self.rng.normal(0, 5, len(timestamps))

        # Price = gas_price * (residual_load / avg_load) ^ elasticity
        avg_residual = np.mean(residual_load)
        elasticity = 2.5  # higher = more price spikes
        price = gas_price * np.clip(residual_load / avg_residual, 0.3, 2.5) ** elasticity

        # Negative prices when solar is high and demand is low
        solar_high = gen[f"gen_{self.zone.lower()}_solar"].values > 3000
        demand_low = load.values < 3500
        negative_mask = solar_high & demand_low
        price[negative_mask] = -self.rng.exponential(10, negative_mask.sum())

        # Random price spikes (transmission constraints, outages)
        spike_prob = 0.005  # ~0.5% of hours
        spike_mask = self.rng.random(len(timestamps)) < spike_prob
        spike_magnitude = self.rng.exponential(80, spike_mask.sum()) + 50
        price[spike_mask] += spike_magnitude

        # Noise
        price += self.rng.normal(0, 5, len(timestamps))

        price = np.round(np.clip(price, -50, 500), 2)
        return pd.Series(price, index=timestamps, name=f"price_{self.zone.lower()}")

    # ── Public API ───────────────────────────────────────────────────────

    def generate(self) -> pd.DataFrame:
        """Generate a complete synthetic dataset."""

        timestamps = pd.date_range(
            self.start_date,
            self.end_date + timedelta(days=1),  # inclusive
            freq="h",
            tz="CET",
            inclusive="left",
        )

        print(f"[GEN] Generating synthetic data: {len(timestamps)} hours")
        print(f"[GEN] Range: {timestamps[0]} → {timestamps[-1]}")

        weather = self._generate_weather(timestamps)
        generation = self._generate_generation(timestamps, weather)
        load = self._generate_load(timestamps, weather)
        price = self._generate_price(timestamps, load, generation)

        df = pd.concat([price, load, generation, weather], axis=1)
        df.index.name = "timestamp"

        # Remove timezone for consistency with real data
        df.index = df.index.tz_convert(None)

        print(f"[GEN] Shape: {df.shape}")
        print(f"[GEN] Price range: {df.iloc[:, 0].min():.1f} → {df.iloc[:, 0].max():.1f} EUR/MWh")
        print(f"[GEN] Price mean: {df.iloc[:, 0].mean():.1f} EUR/MWh")

        return df

    def generate_and_save(
        self,
        output_dir: Optional[Path] = None,
        filename: Optional[str] = None,
    ) -> pd.DataFrame:
        """Generate synthetic data and save as parquet + CSV sample."""

        df = self.generate()

        if output_dir is None:
            output_dir = Path("data/raw")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save as parquet
        parquet_path = output_dir / (filename or f"synthetic_{self.zone.lower()}.parquet")
        df.to_parquet(parquet_path)
        print(f"[SAVED] Parquet → {parquet_path} ({parquet_path.stat().st_size / 1e6:.1f} MB)")

        # Save a CSV sample (last 30 days for quick inspection)
        csv_path = output_dir / (filename or f"synthetic_{self.zone.lower()}_sample.csv")
        df.tail(720).to_csv(csv_path)  # 30 days
        print(f"[SAVED] CSV sample → {csv_path}")

        return df


def describe_synthetic_data(df: pd.DataFrame) -> None:
    """Print summary statistics for the synthetic dataset."""
    price_col = [c for c in df.columns if c.startswith("price_")][0]
    load_col = [c for c in df.columns if c.startswith("load_")][0]

    print("\n" + "=" * 60)
    print("SYNTHETIC DATA SUMMARY")
    print("=" * 60)

    print(f"\nTime range: {df.index.min()} → {df.index.max()}")
    print(f"Total hours: {len(df):,}")
    print(f"Date range: {len(df) / 24:.0f} days")

    print(f"\n--- Price ({price_col}) ---")
    print(f"Mean:      {df[price_col].mean():.2f} EUR/MWh")
    print(f"Median:    {df[price_col].median():.2f} EUR/MWh")
    print(f"Std:       {df[price_col].std():.2f} EUR/MWh")
    print(f"Min:       {df[price_col].min():.2f} EUR/MWh")
    print(f"Max:       {df[price_col].max():.2f} EUR/MWh")
    print(f"Negatives: {(df[price_col] < 0).sum()} hours ({(df[price_col] < 0).mean() * 100:.1f}%)")
    print(f"Spikes (>100): {(df[price_col] > 100).sum()} hours ({(df[price_col] > 100).mean() * 100:.1f}%)")

    print(f"\n--- Load ({load_col}) ---")
    print(f"Mean: {df[load_col].mean():.0f} MW")
    print(f"Peak: {df[load_col].max():.0f} MW")
    print(f"Min:  {df[load_col].min():.0f} MW")

    gen_cols = [c for c in df.columns if c.startswith("gen_")]
    if gen_cols:
        print(f"\n--- Generation (MW, mean) ---")
        for col in gen_cols:
            print(f"  {col.split('_')[-1]:<12}: {df[col].mean():>8.0f} MW")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    from pathlib import Path

    # Generate 2 years of Greek synthetic data
    gen = SyntheticDataGenerator(
        bidding_zone="GR",
        seed=42,
        end_date=date.today(),
    )
    df = gen.generate_and_save(output_dir=Path("data/raw"))
    describe_synthetic_data(df)
