"""
GridPrice — Configuration

Central config for bidding zone, data paths, and model hyperparameters.
Reads .env for secrets; sensible defaults for everything else.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Project Root ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Data Paths ────────────────────────────────────────────────────────
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR  = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

for d in [RAW_DIR, INTERIM_DIR, PROCESSED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── ENTSO-E ───────────────────────────────────────────────────────────
ENTSOE_API_KEY = os.getenv("ENTSOE_API_KEY", "")

# Default bidding zone (Greece)
BIDDING_ZONE    = os.getenv("BIDDING_ZONE", "GR")
DOMAIN_CODE_MAP = {
    "GR": "10YGR-HTSO-----Y",   # Greece
    "DE_LU": "10Y1001A1001A63L",  # Germany-Luxembourg
    "FR": "10YFR-RTE------C",
    "IT": "10YIT-GRTN-----B",
    "ES": "10YES-REE------0",
    "UK": "10YGB----------A",
}

# ── Baseline Model ────────────────────────────────────────────────────
TRAIN_START_DATE = "2024-01-01"
LOOKBACK_DAYS    = 365   # training window size

# ── Evaluation ────────────────────────────────────────────────────────
TEST_SPLIT_FRACTION = 0.2   # last 20% of available data

# ── Logging / Scheduling ──────────────────────────────────────────────
TZ = os.getenv("TZ", "Europe/Athens")
INFERENCE_HOUR_CET = 14  # publish predictions before 14:00 CET
