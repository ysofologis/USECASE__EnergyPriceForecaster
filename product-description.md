# Product Description: Day-Ahead Electricity Price Forecaster

## 1. Product Overview

**Name:** GridPrice — Day-Ahead Electricity Price Forecasting Engine

**Tagline:** Predict tomorrow's electricity prices with machine learning, using live grid and weather data.

**Type:** ML-powered forecasting service with automated daily retraining pipeline.

**Status:** Concept / Discovery

**Repository:** `/storage/drive-S/Work/SideProjects/energy-price-foracaster`

---

## 2. Problem Statement

Electricity day-ahead prices are notoriously volatile — driven by weather, fuel prices, grid congestion, renewable generation, and cross-border flows. Market participants (traders, utilities, industrial consumers, EMS operators) need accurate forecasts to:

- Optimize battery storage dispatch (charge when cheap, discharge when expensive)
- Schedule industrial loads
- Hedge price risk
- Inform trading strategies

Existing solutions are either black-box proprietary models or simplistic benchmarks (persistence, historical averages). There is room for a transparent, ML-driven model that combines multiple live data streams and provides both point forecasts and uncertainty intervals.

---

## 3. Target Users & Use Cases

| User | Use Case |
|------|----------|
| Energy traders | Inform intraday and day-ahead positions |
| EMS operators (our primary) | Optimize battery/storage schedules against predicted prices |
| Industrial consumers | Shift load to low-price hours |
| Portfolio managers | Estimate P&L scenarios for trading desks |
| Research teams | Benchmark against alternative forecast methods |

---

## 4. Data Sources

All sources are free, publicly available, and updated live every day.

| Source | Data | Update Frequency | Coverage |
|--------|------|------------------|----------|
| **ENTSO-E Transparency Platform** | Day-ahead prices, actual generation (solar, wind, gas, coal, nuclear), cross-border physical flows, load forecasts | Hourly / daily (D-1 by 12:00 CET) | Europe (all bidding zones) |
| **Open-Meteo** | Forecasted weather — temperature, wind speed/direction, cloud cover, solar irradiance, precipitation | Hourly, up to 16 days ahead | Global (gridded, any lat/lon) |
| **EMHIR** (if needed) | Day-ahead allocation, flow-based parameters | Daily | Central Europe |
| **Calendar / Holiday data** | Public holidays, school holidays, workdays | Static (annual) | Per country |

---

## 5. ML Approach

**Problem type:** Multivariate time series regression — multi-step ahead (tomorrow's 24 hourly prices).

**Baseline models:**
- Persistence (today = tomorrow)
- Historical average (same day-of-week over last 4 weeks)
- Linear regression with weather features

**Primary models to test:**
- XGBoost with lagged features, rolling statistics, weather, calendar — strong tabular baseline
- LSTM / GRU — captures sequential dependencies across the 24h horizon
- Temporal Fusion Transformer (TFT) — state-of-the-art for multi-horizon forecasting, with built-in feature importance

**Architecture decision:** Start with XGBoost (fast, interpretable, hard to beat on tabular data). Graduate to TFT if performance gap is material.

**Output per forecast run:**
- 24 hourly point forecasts (EUR/MWh)
- 80% and 95% prediction intervals
- Feature importance ranking
- Anomaly flags when actual price deviates > 3x historical sigma from forecast

**Training regime:**
- Expanding window: retrain daily on all available history (D-365 to D-1)
- Daily inference: fetch latest data, run model, publish predictions before 14:00 CET

---

## 6. Features & Engineering

**Weather features (per bidding zone centroid):**
- Temperature (current + rolling 24h mean)
- Wind speed at 100m (relevant for wind generation)
- Cloud cover (solar irradiance proxy)
- Precipitation

**Calendar features:**
- Hour of day, day of week, month
- Public holiday flags (country-specific)
- Daylight saving time transitions

**Grid features (from ENTSO-E):**
- Total load (actual + forecasted)
- Solar generation (actual + forecasted)
- Wind generation (actual + forecasted)
- Cross-border net flow (import/export per interconnection)
- Residual load = load - (solar + wind)
- Price of adjacent bidding zones

**Lagged & rolling features:**
- Price_T-24, Price_T-48, Price_T-168 (same hour yesterday, day before, week before)
- Rolling mean / std over 7-day and 30-day windows
- Price spread vs neighboring zones

---

## 7. Success Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| MAE (Mean Absolute Error) | < 8 EUR/MWh | TBD after baseline |
| sMAPE | < 15% | Symmetric, handles near-zero prices well |
| Pinball Loss (quantile) | < 3 EUR/MWh at 90th percentile | Penalizes bad interval estimates |
| Ramp accuracy | Correctly predict price spikes > 50 EUR/MWh move | Binary: precision + recall |
| Daily run success rate | > 95% | Pipeline reliability metric |

---

## 8. Technical Architecture (High-Level)

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  Data        │────▶│  Feature     │────▶│  Model       │
│  Ingestion   │     │  Engineering │     │  Training    │
│  (ENTSO-E,   │     │  (lag/roll/  │     │  (XGBoost /  │
│   O-Meteo)   │     │   encode)    │     │   TFT)       │
└─────────────┘     └──────────────┘     └──────┬───────┘
                                                │
┌─────────────┐     ┌──────────────┐            │
│  Dashboard   │◀────│  Inference   │◀───────────┘
│  (Streamlit/ │     │  Engine      │
│   Grafana)   │     │  (daily job) │
└─────────────┘     └──────────────┘

┌─────────────┐     ┌──────────────┐
│  Forecast   │     │  Monitoring  │
│  Storage    │     │  (errors,    │
│  (SQLite /  │     │   drift,     │
│   Parquet)  │     │   alerts)    │
└─────────────┘     └──────────────┘
```

**Key stack choices (to be validated):**
- **Language:** Python (pandas, polars, scikit-learn, xgboost, pytorch-forecasting)
- **Scheduling:** cron or simple scheduler (daily at 12:30 CET)
- **Storage:** Parquet for time series data, SQLite for metadata/logs
- **Visualization:** Streamlit dashboard or Grafana for live monitoring

---

## 9. Phased Delivery Plan

| Phase | Scope | Deliverable |
|-------|-------|-------------|
| **P1 — Baseline** | Fetch ENTSO-E data for a single bidding zone (Greece/GR), build persistence + linear models | Working pipeline, baseline metrics |
| **P2 — Feature Engineering** | Add weather, calendar, lagged features. Implement XGBoost | Trained XGBoost model, feature importance report |
| **P3 — Deep Learning** | Implement LSTM and TFT. Compare all models | Model comparison report, pick final architecture |
| **P4 — Production Pipeline** | Daily automated retrain + inference. Handle data gaps gracefully | Fully automated daily forecast |
| **P5 — Dashboard** | Streamlit dashboard: forecast vs actual, metrics, alerts | Live dashboard URL |
| **P6 — Extension** | Add multiple bidding zones, uncertainty intervals, trading signal layer | Multi-zone, production-ready |

---

## 10. Constraints & Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| ENTSO-E API downtime | No data injection | Cache last available data, retry with backoff |
| Negative electricity prices (common with high solar) | Breaks log-transform | Use asinh transform or treat signed data directly |
| Concept drift (changing market rules, new interconnections) | Model degrades | Daily retraining + drift detection monitor |
| Cold start — not enough history | Poor initial predictions | Use transfer learning from similar bidding zones |

---

## 11. Competitors & Alternatives

| Solution | Type | Cost |
|----------|------|------|
| Aurora (EPEX) | Commercial | High |
| EpexSpot API | Simple benchmark | Free |
| SMARD (Germany) | Free data + dashboard | Free, no ML |
| In-house teams | Custom | Variable |

Differentiator: **Transparent, ML-native, open architecture** — not a black box. Built by someone who actually operates EMS systems.

---

## 12. Next Steps

1. [x] Product description written
2. [ ] ENTSO-E API key obtained
3. [ ] Data ingestion script for GR bidding zone
4. [ ] Exploratory data analysis notebook
5. [ ] Baseline model (persistence + linear)
6. [ ] Feature engineering loop
7. [ ] Model training pipeline
8. [ ] Dashboard MVP
