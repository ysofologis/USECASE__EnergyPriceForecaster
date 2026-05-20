# GridPrice — Day-Ahead Electricity Price Forecaster

> **What it does:** Predicts Greece's 24-hour hourly electricity prices (EUR/MWh) for the next day using XGBoost on live grid data, weather forecasts, and engineered features. Runs daily via a fully automated pipeline.

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  ENTSO-E         │    │  Feature         │    │  XGBoost         │
│  Open-Meteo      │───▶│  Engineering     │───▶│  Model           │
│  (live, daily)  │    │  (51 features)   │    │  (recursive 24h) │
└──────────────────┘    └──────────────────┘    └────────┬─────────┘
                                                         │
                                          ┌──────────────▼──────────────┐
                                          │  Forecast: 24 × EUR/MWh    │
                                          │  + feature importance      │
                                          │  + backtest metrics         │
                                          └─────────────────────────────┘
```

---

## Table of Contents

1. [The Problem](#the-problem)
2. [What Is Predicted](#what-is-predicted)
3. [How the Pipeline Works](#how-the-pipeline-works)
4. [Architecture](#architecture)
5. [Feature Engineering](#feature-engineering)
6. [Model](#model)
7. [Getting Started](#getting-started)
8. [CLI & Makefile Targets](#cli--makefile-targets)
9. [Daily Automation](#daily-automation)
10. [Project Structure](#project-structure)
11. [Running Tests](#running-tests)
12. [Configuration](#configuration)
13. [Limitations & Risks](#limitations--risks)

---

## The Problem

Electricity day-ahead prices in Europe are set daily on the EPEX SPOT exchange. They are driven by a complex interplay of factors:

- **Weather** — temperature affects heating/cooling demand; wind and cloud cover directly impact renewable generation
- **Grid conditions** — total load, available generation capacity, cross-border interconnect flows
- **Fuel prices** — gas, coal, and carbonEUA prices set the marginal cost of thermal plants
- **Renewable availability** — high solar/wind pushes prices down during midday (the "duck curve")
- **Calendar** — weekdays vs weekends vs holidays have dramatically different demand profiles
- **DST transitions** — clock shifts disrupt the 24-hour periodicity of the data

For an **Energy Management System (EMS)** operator or energy trader, even a rough 24-hour price forecast enables:
- Scheduling battery storage charge/discharge cycles (buy low, sell high)
- Shifting flexible industrial loads to cheap hours
- Informing day-ahead and intraday trading decisions
- Building P&L scenarios for hedging strategies

---

## What Is Predicted

**Target:** Hourly day-ahead electricity prices for the **Greek bidding zone** (`10YGR-HTSO-----Y`), in EUR/MWh, for each of the next 24 hours.

The output of a daily run is a parquet file `data/processed/forecasts_YYYY-MM-DD.parquet`:

```
timestamp                  forecast_eur_mwh  reference_date
2024-06-02 00:00:00+03:00           87.3     2024-06-01
2024-06-02 01:00:00+03:00           81.4     2024-06-01
...
2024-06-02 23:00:00+03:00           92.1     2024-06-01
```

**Not just a point estimate.** The evaluation module (`notebooks/evaluation.py`) also produces:
- Per-hour-of-day error breakdown (when does the model struggle? peak hours? night?)
- Per-horizon error (does error grow at h=24 vs h=1?)
- Feature importance ranking (what drives the forecast?)
- Daily rolling metrics (is the model degrading over time?)

---

## How the Pipeline Works

The pipeline runs **once per day**, ideally before 12:00 CET (when ENTSO-E publishes the next-day prices).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Daily Pipeline                                    │
│                                                                             │
│  1. FETCH          2. FEATURE ENG.        3. TRAIN           4. PREDICT    │
│  ─────────         ───────────────        ─────────          ─────────    │
│  ENTSO-E           Raw data →             XGBoost on         24 h ahead    │
│  Open-Meteo        51 features            all history        recursive fc   │
│  ↓                 (lag, weather,         weekly retrain     ↓              │
│  Raw parquet       calendar, grid)        ↓                  forecasts.parquet
│  cached            ↓                       model/*.pkl        + metrics     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Step 1 — Fetch

| Source | What | How | Schedule |
|--------|------|-----|----------|
| **ENTSO-E** | Day-ahead load forecast + actual generation by type (solar, wind, gas, hydro) | `entsoe-py` client with free API key from [transparency.entsoe.eu](https://transparency.entsoe.eu) | Daily, D-1 by 12:00 CET |
| **Open-Meteo** | Temperature, wind speed, cloud cover, solar irradiance (hourly, up to 7 days ahead) | Direct HTTP API, no key required | Daily |
| **Cache fallback** | If APIs are down, use the most recent cached parquet files | Automatic in `ENTSOEFetcher` / `OpenMeteoFetcher` | Always |

### Step 2 — Feature Engineering

51 engineered features are computed from the raw inputs (see full list in `src/gridprice/features.py`):

```
Calendar     ── hour, day_of_week, month, weekend, holiday, week_of_year
Cyclical     ── sin/cos encodings of hour, day_of_week, day_of_year
Lags         ── price T-24h, T-48h, T-168h, weekly delta, pct_change_24h
Rolling      ── 24h mean, 168h mean, z-score of price
Weather      ── raw + 6h/24h rolling mean + anomaly for temp/wind/cloud/solar
Grid         ── residual_load = load - solar - wind, renewable_share
DST          ── spring/autumn transition flags (Athens timezone)
```

### Step 3 — Train

XGBoost is retrained weekly using an **expanding window**:
- Training: all data from D-365 to D-8
- Validation: D-7 to D-1 (1 week hold-out for early stopping)
- Retrain trigger: model older than 30 days, or manual `make train`

### Step 4 — Predict

A **recursive 24-hour forecast** is produced:
1. Use the last N hours of data as the input window
2. Predict hour T+1
3. Append T+1 prediction to the input window
4. Repeat for T+2 … T+24

Each step uses the model's own predictions as inputs, simulating the real-world scenario where you only have forecasts, not actuals.

---

## Architecture

```
data/
├── raw/          ← ENTSO-E + Open-Meteo parquet downloads (1 file per fetch)
├── interim/      ← Resampled, merged, cleaned hourly data
└── processed/    ← Forecast output parquet files

models/
└── xgb_model.pkl  ← Trained XGBoost model (dated backups on each retrain)

reports/
├── figures/       ← PNG plots: horizon_error, per_hour, error_dist, etc.
└── metrics/       ← CSV exports: daily, per_hour, per_horizon, per_dow

src/gridprice/
├── synthetic_data.py   ← Synthetic GR data generator (for dev/testing)
├── ingestion.py        ← ENTSO-E + DuckDB raw data loader
├── features.py         ← FeatureEngineer: 51 engineered features
├── models.py           ← XGBPriceModel + baseline models + metrics
├── baseline.py         ← Persistence + seasonal-average baselines
├── backtest.py         ← ExpandingWindowBacktester
└── pipeline.py         ← DailyPipeline orchestrator + CLI

.github/workflows/
├── test.yml    ← pytest + lint on every push/PR
└── train.yml   ← Daily scheduled run: fetch → train → backtest
```

---

## Feature Engineering

The `FeatureEngineer` class transforms raw hourly data into the 51 features used by XGBoost. Key groups:

### Calendar Features
- `hour` (0–23), `day_of_week` (0–6), `month` (1–12), `week_of_year`
- `is_weekend`, `is_holiday` (Greek public holidays hardcoded 2024–2030)
- `part_of_day` (night/morning/afternoon/evening), `is_dst_spring`, `is_dst_autumn`

### Cyclical Encodings
- `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `doy_sin`, `doy_cos`, `month_sin`, `month_cos`
- `hour_of_week` (0–167) — captures intra-week patterns without one-hot explosion

### Lag Features
- `price_lag_24h`, `price_lag_48h`, `price_lag_168h` (same hour last week)
- `weekly_delta` (price change over past week), `pct_change_24h`

### Rolling Statistics
- `price_roll_mean_24h`, `price_roll_std_24h`, `price_roll_zscore_168h`
- `price_vs_7d_mean` (how does current price compare to last week's average?)

### Weather Features
- Raw: `temperature_2m_c`, `wind_speed_10m_ms`, `cloud_cover`, `solar_irradiance_wm2`
- Rolling: 6h and 24h rolling means for each weather variable
- Anomaly: deviation from 24h rolling mean (captures sudden weather shifts)

### Grid Features
- `residual_load = load - solar - wind` — the "net demand" thermal plants must cover
- `renewable_share`, `solar_to_residual`, `load_roll24h`, `load_pct_of_7d`

---

## Model

### XGBoost Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_estimators` | 200 (early stopping) | ~100–150 typically used with early stopping |
| `max_depth` | 6 | Moderate complexity; deeper trees overfit |
| `learning_rate` | 0.05 | Slow learning for stability |
| `subsample` | 0.8 | Row subsampling for regularization |
| `colsample_bytree` | 0.8 | Feature subsampling per tree |
| `reg_lambda` | 1.0 | L2 regularization |
| `reg_alpha` | 0.1 | L1 regularization (sparsity) |
| `min_train_size` | 4,320h (~6 months) | Enough history for seasonal patterns |
| `val_size` | 168h (1 week) | Early stopping hold-out |

### Baselines

The model is always compared against:

| Model | Description | Expected MAE (synthetic) |
|-------|-------------|--------------------------|
| **Persistence** | Tomorrow's price = Today's price (same hour) | ~33 EUR/MWh |
| **Seasonal Average** | Average of same hour, same DOW, last 4 weeks | ~8 EUR/MWh |
| **Linear Regression** | Hour + DOW + weather features | ~7 EUR/MWh |
| **XGBoost** | Full feature set, recursive forecast | ~1.5 EUR/MWh (synthetic) |

### Metrics

| Metric | Formula | Notes |
|--------|---------|-------|
| MAE | mean\|error\| | Primary metric; interpretable in EUR/MWh |
| RMSE | sqrt(mean(error²)) | Penalizes large errors |
| MAPE | mean(\|error\|/\|actual\|) × 100 | Scale-independent, but problematic near 0 |
| sMAPE | Symmetric MAPE | Better behaved than MAPE for energy data |

---

## Getting Started

### Prerequisites

- Python 3.11 or 3.12
- ~2 GB disk space for data and models

### 1. Clone & install

```bash
git clone https://github.com/your-org/energy-price-foracaster.git
cd energy-price-foracaster
make install          # creates .venv and installs dependencies
```

### 2. Get an ENTSO-E API key (optional but recommended)

1. Go to [https://transparency.entsoe.eu](https://transparency.entsoe.eu) → Register
2. Copy your API key
3. Set it: `export ENTSOE_API_KEY="your-key-here"`

Without a key, the pipeline falls back to **synthetic data** (realistic but not live).

### 3. Quick demo (no API key needed)

```bash
# Generate synthetic data and run evaluation
make evaluate END_DATE=2024-06-01

# Or step by step:
make fetch-sample       # synthetic GR data → data/interim/sample_gr.parquet
make train             # train XGBoost on synthetic data
make predict           # produce 24h forecast
make backtest DAYS=90  # quick backtest on 3 months
```

### 4. With live data

```bash
export ENTSOE_API_KEY="your-key-here"
make fetch             # fetch from ENTSO-E
make train             # retrain on real data
make predict           # forecast
make backtest DAYS=365 # full backtest on all available history
make evaluate          # full evaluation + plots + CSVs
```

---

## CLI & Makefile Targets

| Command | What it does |
|---------|-------------|
| `make install` | Create venv and install `requirements.txt` |
| `make fetch-sample` | Download/generate synthetic GR data (no API key) |
| `make fetch` | Fetch live ENTSO-E + Open-Meteo data |
| `make train` | Train or retrain XGBoost model |
| `make predict` | Produce 24 h ahead forecast |
| `make backtest` | Run expanding-window backtest |
| `make evaluate` | Full backtest + all plots + CSV export |
| `make daily` | Run fetch + train + predict (cron target) |
| `make test` | Run unit tests (features + models) |
| `make lint` | Run pylint on `src/gridprice/` |
| `make clean` | Remove generated data and reports |
| `make help` | Show all targets |

### Python CLI

```bash
# Pipeline
python -m src.gridprice.pipeline train --config config.yaml
python -m src.gridprice.pipeline predict --date 2024-06-01 --horizon 24
python -m src.gridprice.pipeline backtest --backtest-days 365

# Evaluation
python notebooks/evaluation.py --data-end 2024-06-01 --n-estimators 50 --step 168
```

---

## Daily Automation

### Option 1 — Cron (recommended for this project)

Add to your crontab (`crontab -e`):

```cron
# Run daily at 12:30 CET (14:30 Athens / EET = UTC+3)
30 12 * * * cd /path/to/energy-price-foracaster && .venv/bin/python -m src.gridprice.pipeline run --config config.yaml >> logs/daily.log 2>&1
```

### Option 2 — GitHub Actions

The `.github/workflows/train.yml` runs automatically every day at **12:00 UTC** (14:00 Athens):

1. Fetches ENTSO-E data (requires `ENTSOE_API_KEY` as a GitHub Secret)
2. Trains / retrains the XGBoost model
3. Runs the expanding-window backtest
4. Uploads backtest metrics and the trained model as GitHub artifacts

To enable: go to **Settings → Secrets** in your GitHub repo and add `ENTSOE_API_KEY`.

---

## Project Structure

```
energy-price-foracaster/
├── .github/workflows/
│   ├── test.yml          ← Unit tests + lint on push/PR
│   └── train.yml         ← Daily scheduled training pipeline
├── .venv/                ← Virtual environment (created by make install)
├── config.yaml           ← Pipeline configuration (API key, hyperparams)
├── Makefile              ← All make targets
├── requirements.txt       ← Python dependencies
├── notebooks/
│   └── evaluation.py     ← Evaluation CLI: backtest + plots + CSV export
├── reports/
│   ├── figures/           ← PNG plots from evaluation
│   └── metrics/           ← CSV metric exports from backtest
├── data/
│   ├── raw/               ← Raw ENTSO-E + Open-Meteo parquet files
│   ├── interim/           ← Resampled, merged hourly data
│   └── processed/         ← Forecast output parquet files
├── models/               ← Trained XGBoost model (.pkl)
├── src/gridprice/
│   ├── __init__.py
│   ├── synthetic_data.py  ← Deterministic synthetic GR data generator
│   ├── ingestion.py       ← ENTSO-E + DuckDB raw data ingestion
│   ├── features.py        ← FeatureEngineer: 51 engineered features
│   ├── models.py          ← XGBPriceModel, baselines, metrics
│   ├── baseline.py        ← Persistence + seasonal-average baselines
│   ├── backtest.py        ← ExpandingWindowBacktester
│   └── pipeline.py        ← DailyPipeline orchestrator + CLI
└── tests/
    ├── test_features.py   ← 50 tests for feature engineering
    └── test_models.py     ← 39 tests for models + backtest
```

---

## Running Tests

```bash
make test                    # All tests
make test-features           # Feature engineering tests only (~50 tests)
make test-models             # Model tests only (~39 tests, ~5 min)
make test-cov                # With coverage report
```

All **89 tests** currently pass on the synthetic dataset.

---

## Configuration

All configuration lives in `config.yaml`. **Environment variables always override** the YAML file.

```yaml
bidding_zone: "GR"

entsoe:
  api_key: ""                  # set via ENTOSE_API_KEY env var
  area_id: "10YGR-HTSO-----Y" # Greek bidding zone

open_meteo:
  latitude: 37.9838           # Athens
  longitude: 23.7275
  forecast_days: 7

model:
  path: "models/xgb_model.pkl"
  n_estimators: 200
  max_depth: 6
  learning_rate: 0.05
  min_train_size: 4320        # ~6 months
  val_size: 168              # 1 week early-stopping holdout

pipeline:
  backtest_days: 365         # historical window for backtest
  retrain_threshold_days: 30  # auto-retrain if model is older
```

### Environment Variables

| Variable | Overrides | Notes |
|----------|-----------|-------|
| `ENTSOE_API_KEY` | `entsoe.api_key` | Required for live data |
| `BIDDING_ZONE` | `bidding_zone` | Change target zone |
| `OPEN_METEO_LAT` | `open_meteo.latitude` | |
| `OPEN_METEO_LON` | `open_meteo.longitude` | |
| `MODEL_DIR` | `models_dir` | |
| `REPORTS_DIR` | `reports_dir` | |
| `DATA_DIR` | `data_dir` | |

---

## Limitations & Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| **ENTSO-E API downtime** | No live data fetch | Cache fallback: use last cached parquet file |
| **Negative prices** | Log-transform would break | Using raw (signed) prices; XGBoost handles this naturally |
| **Cold start** | New zone has insufficient history | Transfer learning from similar Mediterranean zones (IT, ES) |
| **Concept drift** | Market rule changes degrade model | Daily retraining; backtest monitors rolling MAE |
| **Weather forecast error** | Poor weather → poor price forecast | Weather anomaly features flag unusual inputs |
| **Renewable curtailment** | Not in data — prices spike unexpectedly | Lag features + residual load help, but tail risk remains |

---

## Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| TS-1 to TS-7 | ✅ Done | Scaffold, ingestion, features, XGBoost model |
| **TS-8** | ✅ Done | Evaluation, backtester, plots, CSV export |
| **TS-9** | ✅ Done | Daily pipeline, Makefile, GitHub Actions CI |
| TS-10 Prediction Intervals | Backlog | Quantile regression → 80%/95% confidence intervals |
| TS-11 Anomaly Detection | Backlog | Flag when actual deviates > 3σ from forecast |
| TS-12 Dashboard | Backlog | Streamlit: forecast vs actual, live metrics, alerts |
| TS-13 Testing | Backlog | Integration tests, pipeline smoke tests |
| TS-14 Multi-Zone | Stretch | DE-LU, FR, IT, ES bidding zones |

---

*Built with Python · XGBoost · ENTSO-E · Open-Meteo · GitHub Actions*
