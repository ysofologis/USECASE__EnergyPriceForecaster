"""
GridPrice — ENTSO-E Data Ingestion

Responsible for downloading day-ahead prices and generation data
from the ENTSO-E Transparency Platform.
"""

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.config import (
    ENTSOE_API_KEY,
    DOMAIN_CODE_MAP,
    BIDDING_ZONE,
    RAW_DIR,
)


class EntsoeIngestor:
    """Ingest day-ahead prices, load, and generation data from ENTSO-E."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or ENTSOE_API_KEY
        if not self.api_key:
            raise ValueError(
                "ENTSOE_API_KEY not set. Provide it via .env or constructor."
            )
        self.client = EntsoePandasClient(api_key=self.api_key)

    # ── Domain helpers ──────────────────────────────────────────────────

    def domain_code(self, bidding_zone: str) -> str:
        return DOMAIN_CODE_MAP.get(bidding_zone.upper(), bidding_zone)

    # ── Data fetchers ───────────────────────────────────────────────────

    def fetch_day_ahead_prices(
        self,
        start: date,
        end: date,
        bidding_zone: Optional[str] = None,
    ) -> pd.Series:
        """
        Download day-ahead hourly prices (EUR/MWh) for a bidding zone.

        Returns a pandas Series indexed by datetime (CET).
        """
        zone = bidding_zone or BIDDING_ZONE
        code = self.domain_code(zone)
        try:
            series: pd.Series = self.client.query_day_ahead_prices(
                code, start=start, end=end
            )
            series.name = f"price_{zone.lower()}"
            # Ensure tz-naive CET
            if hasattr(series.index, "tz"):
                series.index = series.index.tz_convert(None)
            return series
        except NoMatchingDataError:
            print(f"[WARN] No price data for {zone} between {start} — {end}")
            return pd.Series(dtype=float)

    def fetch_load(
        self,
        start: date,
        end: date,
        bidding_zone: Optional[str] = None,
    ) -> pd.Series:
        """
        Download hourly actual total load (MW).
        """
        zone = bidding_zone or BIDDING_ZONE
        code = self.domain_code(zone)
        try:
            series: pd.Series = self.client.query_load(
                code, start=start, end=end
            )
            series.name = f"load_{zone.lower()}"
            if hasattr(series.index, "tz"):
                series.index = series.index.tz_convert(None)
            return series
        except NoMatchingDataError:
            print(f"[WARN] No load data for {zone} between {start} — {end}")
            return pd.Series(dtype=float)

    def fetch_generation(
        self,
        start: date,
        end: date,
        bidding_zone: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Download hourly actual generation per fuel type (MW).
        Returns a DataFrame with one column per fuel type.
        """
        zone = bidding_zone or BIDDING_ZONE
        code = self.domain_code(zone)
        try:
            df: pd.DataFrame = self.client.query_generation(
                code, start=start, end=end, psy_domain="A65"
            )
            # Flatten multi-index columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(1).map(
                    lambda x: str(x).lower().replace(" ", "_")
                )
            # Prefix columns
            df = df.add_prefix(f"gen_{zone.lower()}_")
            if hasattr(df.index, "tz"):
                df.index = df.index.tz_convert(None)
            return df
        except NoMatchingDataError:
            print(f"[WARN] No generation data for {zone} between {start} — {end}")
            return pd.DataFrame()

    def fetch_cross_border_flows(
        self,
        start: date,
        end: date,
        out_domain: str,
        in_domain: str,
    ) -> pd.Series:
        """
        Download hourly cross-border physical flows (MW) between two domains.
        Positive = flow from out_domain to in_domain.
        """
        try:
            series: pd.Series = self.client.query_crossborder_flows(
                out_domain, in_domain, start=start, end=end
            )
            series.name = f"flow_{out_domain}_to_{in_domain}"
            if hasattr(series.index, "tz"):
                series.index = series.index.tz_convert(None)
            return series
        except NoMatchingDataError:
            print(
                f"[WARN] No flow data {out_domain} → {in_domain} "
                f"between {start} — {end}"
            )
            return pd.Series(dtype=float)

    # ── Batch fetcher ───────────────────────────────────────────────────

    def fetch_all_for_zone(
        self,
        start: date,
        end: date,
        bidding_zone: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Download prices, load, and generation for a zone and merge into
        a single DataFrame indexed by datetime.
        """
        zone = bidding_zone or BIDDING_ZONE
        price = self.fetch_day_ahead_prices(start, end, zone)
        load = self.fetch_load(start, end, zone)
        gen = self.fetch_generation(start, end, zone)

        dfs = [df for df in [price, load, gen] if not df.empty]
        if not dfs:
            print(f"[WARN] No data returned for {zone}. Check API key / coverage.")
            return pd.DataFrame()

        merged = pd.concat(dfs, axis=1)
        merged.index.name = "timestamp"
        return merged

    # ── Save / Load helpers ─────────────────────────────────────────────

    def save_raw(self, df: pd.DataFrame, zone: str, dt: date) -> Path:
        """Save raw data to a parquet file."""
        out_path = RAW_DIR / f"{zone.lower()}_{dt:%Y%m%d}.parquet"
        df.to_parquet(out_path)
        print(f"[SAVED] {len(df)} rows → {out_path}")
        return out_path

    def load_raw(self, zone: str, dt: date) -> pd.DataFrame:
        """Load previously saved raw data."""
        path = RAW_DIR / f"{zone.lower()}_{dt:%Y%m%d}.parquet"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)


# ── CLI entry point for manual testing ─────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest ENTSO-E data")
    parser.add_argument("--zone", default=BIDDING_ZONE)
    parser.add_argument("--days", type=int, default=7)
    args = parser.parse_args()

    ing = EntsoeIngestor()
    end = date.today()
    start = end - timedelta(days=args.days)

    df = ing.fetch_all_for_zone(start, end, args.zone)
    if not df.empty:
        print(df.head(10))
        print(f"\nShape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        ing.save_raw(df, args.zone, end)
    else:
        print("No data fetched.")
