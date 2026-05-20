"""
GridPrice — Daily Pipeline

Classes
-------
DailyPipeline
    Orchestrates a single daily run:
    1. Fetch latest load / generation / weather data
    2. Engineer features
    3. Load (or retrain) the model
    4. Produce 24 h ahead forecast
    5. Persist results to parquet

CLI
---
``python -m src.gridprice.pipeline --config config.yaml``
or individual targets:
  ``python -m src.gridprice.pipeline fetch   --zone GR``
  ``python -m src.gridprice.pipeline train   --config config.yaml``
  ``python -m src.gridprice.pipeline predict --date 2024-06-01``

Environment variables (override config.yaml):
  ENTOSO_API_KEY, OPEN_METEO_LAT, OPEN_METEO_LON, BIDDING_ZONE,
  MODEL_PATH, DATA_DIR, REPORTS_DIR
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


# ── Config ───────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "bidding_zone": "GR",
    "entsoe": {
        "api_key": os.getenv("ENTSOE_API_KEY", ""),
        "area_type": "BZN",
        "area_id": "10YGR-HTSO-----Y",   # Greek bidding zone
    },
    "open_meteo": {
        "latitude": 37.9838,
        "longitude": 23.7275,
        "forecast_days": 7,
    },
    "data_dir": "data",
    "raw_dir": "data/raw",
    "interim_dir": "data/interim",
    "processed_dir": "data/processed",
    "models_dir": "models",
    "reports_dir": "reports",
    "model": {
        "path": "models/xgb_model.pkl",
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.05,
        "min_train_size": 24 * 180,
        "val_size": 168,
    },
    "pipeline": {
        "backtest_days": 365,
        "retrain_threshold_days": 30,  # retrain if model is older than this
        "notify_on_error": False,
    },
}


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file, falling back to defaults.

    Environment variables always take precedence over the YAML file.
    """
    config = DEFAULT_CONFIG.copy()
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            user = yaml.safe_load(f) or {}
        config = _deep_merge(config, user)

    # Environment overrides
    for key, env_var in [
        ("entsoe.api_key", "ENTSOE_API_KEY"),
        ("open_meteo.latitude", "OPEN_METEO_LAT"),
        ("open_meteo.longitude", "OPEN_METEO_LON"),
        ("bidding_zone", "BIDDING_ZONE"),
        ("models_dir", "MODEL_DIR"),
        ("reports_dir", "REPORTS_DIR"),
        ("data_dir", "DATA_DIR"),
    ]:
        value = os.getenv(env_var)
        if value is not None:
            _set_nested(config, key, value)
    return config


def _deep_merge(base: Dict, override: Dict) -> Dict:
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _set_nested(d: Dict, path: str, value: Any) -> None:
    keys = path.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = type(d[keys[-1]])(value) if keys[-1] in d else value


# ── Data fetchers ─────────────────────────────────────────────────────────────

class ENTSOEFetcher:
    """
    Fetch Greek bidding-zone load and generation data from ENTSO-E.

    Requires a valid API key from https://transparency.entsoe.eu/
    Set ``ENTSOE_API_KEY`` env var or ``entsoe.api_key`` in config.

    Falls back to cached parquet files in ``raw_dir`` if the API call fails.
    """

    AREA_IDS = {
        "GR": "10YGR-HTSO-----Y",
    }

    def __init__(self, config: Dict[str, Any]) -> None:
        self.api_key = config["entsoe"]["api_key"]
        self.area_id = self.AREA_IDS.get(
            config["bidding_zone"],
            config["entsoe"].get("area_id", "10YGR-HTSO-----Y"),
        )
        self.raw_dir = Path(config["raw_dir"])
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def fetch_load(
        self,
        start: date,
        end: date,
        use_cache: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch Day-Ahead Total Load Forecast (PT15M resolution).

        Returns DataFrame with DatetimeIndex and column ``load_mw``,
        or None if the fetch fails and ``use_cache=True`` was given.
        """
        cache_file = self.raw_dir / f"load_{self.area_id}_{start}_{end}.parquet"
        if use_cache and cache_file.exists():
            logger.info("[ENTSOE] Load: using cache %s", cache_file)
            return pd.read_parquet(cache_file)

        if not self.api_key:
            logger.warning("[ENTSOE] No API key — using cache or synthetic data.")
            return None

        try:
            from entsoe import EntsoeRawClient
        except ImportError:
            logger.error("[ENTSOE] entsoe-py not installed.")
            return None

        try:
            client = EntsoeRawClient(self.api_key)
            start_naive = datetime(start.year, start.month, start.day)
            end_naive = datetime(end.year, end.month, end.day, 23, 59)
            df = client.query_total_load(
                self.area_id, start=start_naive, end=end_naive,
            )
            df.index = df.index.tz_convert("Europe/Athens")
            df.columns = ["load_mw"]
            df.to_parquet(cache_file)
            logger.info("[ENTSOE] Load fetched and cached: %s", cache_file)
            return df
        except Exception as exc:
            logger.error("[ENTSOE] Fetch failed: %s", exc)
            return None

    def fetch_generation(
        self,
        start: date,
        end: date,
        use_cache: bool = True,
    ) -> Optional[pd.DataFrame]:
        """Fetch actual generation per production type (PT15M)."""
        cache_file = self.raw_dir / f"generation_{self.area_id}_{start}_{end}.parquet"
        if use_cache and cache_file.exists():
            logger.info("[ENTSOE] Generation: using cache %s", cache_file)
            return pd.read_parquet(cache_file)

        if not self.api_key:
            logger.warning("[ENTSOE] No API key — using cache or synthetic data.")
            return None

        try:
            from entsoe import EntsoeRawClient
            client = EntsoeRawClient(self.api_key)
            start_naive = datetime(start.year, start.month, start.day)
            end_naive = datetime(end.year, end.month, end.day, 23, 59)
            df = client.query_generation(
                self.area_id, start=start_naive, end=end_naive,
            )
            df.index = df.index.tz_convert("Europe/Athens")
            df.to_parquet(cache_file)
            logger.info("[ENTSOE] Generation fetched and cached: %s", cache_file)
            return df
        except Exception as exc:
            logger.error("[ENTSOE] Generation fetch failed: %s", exc)
            return None


class OpenMeteoFetcher:
    """Fetch weather forecast from Open-Meteo (free, no API key required)."""

    BASE_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(self, config: Dict[str, Any]) -> None:
        self.lat = float(config["open_meteo"]["latitude"])
        self.lon = float(config["open_meteo"]["longitude"])
        self.forecast_days = int(config["open_meteo"].get("forecast_days", 7))
        self.raw_dir = Path(config["raw_dir"])
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def fetch(
        self,
        start: Optional[date] = None,
        end: Optional[date] = None,
        use_cache: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical + forecast weather.

        Parameters
        ----------
        start, end : date
            Range to fetch.  ``end`` defaults to today + forecast_days.
        use_cache : bool

        Returns DataFrame with DatetimeIndex and columns:
        temperature_2m_c, wind_speed_10m_ms, cloud_cover, solar_irradiance_wm2
        """
        if end is None:
            end = date.today() + timedelta(days=self.forecast_days)
        if start is None:
            start = end - timedelta(days=7)

        cache_file = self.raw_dir / f"weather_{self.lat}_{self.lon}_{start}_{end}.parquet"
        if use_cache and cache_file.exists():
            logger.info("[OpenMeteo] Using cache %s", cache_file)
            return pd.read_parquet(cache_file)

        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "hourly": (
                "temperature_2m,temperature_80m,"
                "wind_speed_10m,wind_speed_80m,"
                "cloud_cover,shortwave_radiation_instant,"
                "direct_radiation_instant"
            ),
            "timezone": "Europe/Athens",
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
        }
        try:
            import requests
            resp = requests.get(self.BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            hourly = data["hourly"]
            df = pd.DataFrame({
                "timestamp": pd.to_datetime(hourly["time"]),
                "temperature_2m_c": hourly["temperature_2m"],
                "temperature_80m_c": hourly["temperature_80m"],
                "wind_speed_10m_ms": hourly["wind_speed_10m"],
                "wind_speed_80m_ms": hourly["wind_speed_80m"],
                "cloud_cover": hourly["cloud_cover"],
                "solar_irradiance_wm2": hourly["shortwave_radiation_instant"],
                "direct_radiation_wm2": hourly["direct_radiation_instant"],
            })
            df = df.set_index("timestamp")
            # Ensure timezone-aware
            if df.index.tz is None:
                df.index = df.index.tz_localize("Europe/Athens")
            df.to_parquet(cache_file)
            logger.info("[OpenMeteo] Fetched and cached: %s", cache_file)
            return df
        except Exception as exc:
            logger.error("[OpenMeteo] Fetch failed: %s", exc)
            return None


# ── Pipeline ─────────────────────────────────────────────────────────────────

class DailyPipeline:
    """
    End-to-end daily forecasting pipeline.

    Run daily via cron (see ``config.yaml`` / ``Makefile`` targets).

    Usage
    -----
    >>> from gridprice.pipeline import DailyPipeline
    >>> p = DailyPipeline(config_path="config.yaml")
    >>> p.run()           # full: fetch → train → predict
    >>> p.fetch()         # data fetch only
    >>> p.train()         # retrain model
    >>> p.predict()       # produce 24 h forecast
    >>> p.backtest()      # full backtest on historical data
    """

    PRICE_COL = "price_gr"
    LOAD_COL = "load_gr"

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[Path] = None,
    ) -> None:
        self.config = config or load_config(config_path)
        self._setup_dirs()
        self._model = None
        self._fe = None
        self._logged: Dict[str, Any] = {}

    def _setup_dirs(self) -> None:
        for key in ("raw_dir", "interim_dir", "processed_dir", "models_dir", "reports_dir"):
            path = Path(self.config.get(key, f"data/{key}"))
            path.mkdir(parents=True, exist_ok=True)

    # ── Fetch ────────────────────────────────────────────────────────

    def fetch(self) -> "DailyPipeline":
        """
        Fetch latest load, generation, and weather data.
        Caches results to ``raw_dir``.
        """
        today = date.today()
        end = today + timedelta(days=self.config["open_meteo"].get("forecast_days", 7))
        start = end - timedelta(days=self.config["pipeline"].get("backtest_days", 365))

        logger.info("[Pipeline] Fetching data from %s to %s", start, end)

        # ENTSO-E
        entsoe = ENTSOEFetcher(self.config)
        self._logged["load"] = entsoe.fetch_load(start, end) is not None
        self._logged["generation"] = entsoe.fetch_generation(start, end) is not None

        # Open-Meteo
        meteo = OpenMeteoFetcher(self.config)
        self._logged["weather"] = meteo.fetch(start, end) is not None

        return self

    # ── Ingest / merge ──────────────────────────────────────────────

    def load_latest(self) -> pd.DataFrame:
        """
        Load and merge the most recent available data from raw_dir.

        Returns a merged DataFrame with DatetimeIndex and columns:
        load_gr, gen_gr_solar, gen_gr_wind_onshore, gen_gr_gas, gen_gr_hydro,
        temperature_2m_c, wind_speed_10m_ms, cloud_cover, solar_irradiance_wm2
        """
        raw_dir = Path(self.config["raw_dir"])
        tz = "Europe/Athens"

        # Try to load from duckdb cache if available
        interim = Path(self.config["interim_dir"])

        # Fall back: find most recent raw parquet files
        load_files = sorted(raw_dir.glob("load_*.parquet"))
        gen_files = sorted(raw_dir.glob("generation_*.parquet"))
        weather_files = sorted(raw_dir.glob("weather_*.parquet"))

        frames: List[pd.DataFrame] = []

        if load_files:
            df = pd.read_parquet(load_files[-1])
            df = df.rename(columns={"load_mw": self.LOAD_COL})
            frames.append(df)

        if gen_files:
            df = pd.read_parquet(gen_files[-1])
            # Resample to hourly mean
            df = df.resample("h").mean()
            frames.append(df)

        if weather_files:
            df = pd.read_parquet(weather_files[-1])
            frames.append(df)

        if not frames:
            logger.warning("[Pipeline] No raw data found — using synthetic data.")
            from gridprice.synthetic_data import SyntheticDataGenerator
            gen = SyntheticDataGenerator(
                bidding_zone=self.config["bidding_zone"],
                seed=42,
                end_date=date.today(),
            )
            return gen.generate()

        merged = frames[0]
        for df in frames[1:]:
            merged = merged.join(df, how="outer")

        merged = merged.sort_index()
        merged.index = merged.index.tz_convert(tz) if merged.index.tz is None else merged.index

        # Drop UTC columns that might have leaked in
        for col in list(merged.columns):
            if "UTC" in col or col.startswith("Unnamed"):
                merged = merged.drop(columns=[col])

        # Resample everything to hourly
        merged = merged.resample("h").mean()

        return merged

    # ── Train ──────────────────────────────────────────────────────

    def train(
        self,
        df: Optional[pd.DataFrame] = None,
        retrain: bool = True,
    ) -> "DailyPipeline":
        """
        Train or retrain the XGBoost model.

        Parameters
        ----------
        df : DataFrame, optional
            Training data.  If None, loads from load_latest().
        retrain : bool
            If True (default), retrain even if a model exists.
            If False, skip if model file is newer than ``retrain_threshold_days``.
        """
        from gridprice.models import XGBPriceModel, XGBConfig

        model_path = Path(self.config["model"]["path"])
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # Check age of existing model
        if model_path.exists() and not retrain:
            age_days = (datetime.now() - datetime.fromtimestamp(model_path.stat().st_mtime)).days
            threshold = self.config["pipeline"].get("retrain_threshold_days", 30)
            if age_days < threshold:
                logger.info("[Pipeline] Model is %d days old (< threshold %d). "
                            "Skipping retrain.", age_days, threshold)
                return self

        df = df if df is not None else self.load_latest()

        mc = self.config["model"]
        cfg = XGBConfig(
            n_estimators=int(mc.get("n_estimators", 200)),
            max_depth=int(mc.get("max_depth", 6)),
            learning_rate=float(mc.get("learning_rate", 0.05)),
            min_train_size=int(mc.get("min_train_size", 24 * 180)),
            val_size=int(mc.get("val_size", 168)),
        )

        logger.info("[Pipeline] Training XGBPriceModel on %d rows …", len(df))
        model = XGBPriceModel(config=cfg, verbose=True)
        model.fit(df)

        model.save(model_path)
        logger.info("[Pipeline] Model saved to %s", model_path)
        self._model = model
        return self

    # ── Predict ─────────────────────────────────────────────────────

    def predict(
        self,
        reference_date: Optional[date] = None,
        horizon: int = 24,
    ) -> pd.DataFrame:
        """
        Produce 24 h ahead forecasts.

        Parameters
        ----------
        reference_date : date
            Forecast date.  Defaults to today.
        horizon : int
            Number of hours ahead to forecast.

        Returns
        -------
        DataFrame with columns: timestamp, forecast, model_version
        """
        from gridprice.models import XGBPriceModel

        if self._model is None:
            model_path = Path(self.config["model"]["path"])
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model not found at {model_path}. Run train() first."
                )
            self._model = XGBPriceModel.load(model_path)

        df = self.load_latest()
        fc = self._model.forecast_24h(df, horizon=horizon)

        result = pd.DataFrame({
            "timestamp": fc.index,
            "forecast_eur_mwh": fc.values,
            "reference_date": reference_date or date.today(),
        })

        # Save results
        ref_str = (reference_date or date.today()).isoformat()
        out_dir = Path(self.config["processed_dir"])
        out_path = out_dir / f"forecasts_{ref_str}.parquet"
        result.to_parquet(out_path, index=False)
        logger.info("[Pipeline] Forecast saved to %s", out_path)

        return result

    # ── Backtest ───────────────────────────────────────────────────

    def backtest(
        self,
        df: Optional[pd.DataFrame] = None,
        **backtest_kwargs,
    ) -> pd.DataFrame:
        """
        Run the expanding-window backtest on historical data.

        Parameters
        ----------
        df : DataFrame, optional
            Full dataset.  Loaded from raw_dir if not provided.
        **backtest_kwargs
            Passed to ExpandingWindowBacktester.

        Returns
        -------
        pd.DataFrame of per-iteration metrics.
        """
        from gridprice.backtest import ExpandingWindowBacktester, BacktestConfig

        df = df if df is not None else self.load_latest()

        backtest_kwargs.setdefault("train_min",
                                    self.config["model"].get("min_train_size", 24 * 180))
        backtest_kwargs.setdefault("step", 168)
        backtest_kwargs.setdefault("n_estimators",
                                    self.config["model"].get("n_estimators", 200))
        backtest_kwargs.setdefault("max_depth",
                                    self.config["model"].get("max_depth", 6))

        

        cfg = BacktestConfig(**backtest_kwargs)

        def factory():
            from gridprice.models import XGBPriceModel, XGBConfig
            mc = self.config["model"]
            cfg_m = XGBConfig(
                n_estimators=int(mc.get("n_estimators", 200)),
                max_depth=int(mc.get("max_depth", 6)),
                learning_rate=float(mc.get("learning_rate", 0.05)),
                min_train_size=int(mc.get("min_train_size", 24 * 180)),
            )
            return XGBPriceModel(config=cfg_m, verbose=False)

        bt = ExpandingWindowBacktester(config=cfg, model_factory=factory)
        bt.run(df, target_col=self.PRICE_COL, show_progress=True)

        # Export
        reports_dir = Path(self.config["reports_dir"])
        bt.export_metrics_csv(reports_dir / "metrics")

        return bt.summary

    # ── Full run ───────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """
        Execute the full daily pipeline: fetch → train → predict.

        Returns the forecast DataFrame.
        """
        logger.info("[Pipeline] Starting daily run at %s", datetime.now())
        self.fetch()
        self.train()
        forecast = self.predict()
        logger.info("[Pipeline] Daily run complete.")
        return forecast


# ── CLI ─────────────────────────────────────────────────────────────────────

def _cli():
    """CLI entry point: python -m src.gridprice.pipeline [fetch|train|predict|backtest]"""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="GridPrice Daily Pipeline")
    parser.add_argument("action", choices=["fetch", "train", "predict", "backtest", "run"],
                        help="Pipeline action to run")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"),
                        help="Path to config.yaml")
    parser.add_argument("--date", type=date.fromisoformat, default=None,
                        help="Reference date for forecast (YYYY-MM-DD)")
    parser.add_argument("--horizon", type=int, default=24,
                        help="Forecast horizon in hours")
    parser.add_argument("--no-retrain", action="store_true",
                        help="Skip retraining if model exists")
    parser.add_argument("--backtest-days", type=int, default=365,
                        help="Days of data for backtest")
    args = parser.parse_args()

    config = load_config(args.config) if args.config.exists() else DEFAULT_CONFIG
    config["pipeline"]["backtest_days"] = args.backtest_days

    pipeline = DailyPipeline(config=config)

    if args.action == "fetch":
        pipeline.fetch()
        print("[Pipeline] Data fetch complete.")

    elif args.action == "train":
        pipeline.train(retrain=not args.no_retrain)
        print("[Pipeline] Training complete.")

    elif args.action == "predict":
        fc = pipeline.predict(reference_date=args.date, horizon=args.horizon)
        print(f"[Pipeline] Forecast for {args.date or date.today()}:")
        print(fc.to_string(index=False))

    elif args.action == "backtest":
        print("[Pipeline] Running backtest …")
        summary = pipeline.backtest()
        print(summary[["MAE", "RMSE", "MAPE"]].describe().to_string())

    elif args.action == "run":
        fc = pipeline.run()
        print(f"[Pipeline] Forecast:\n{fc.head(24).to_string(index=False)}")


if __name__ == "__main__":
    _cli()
