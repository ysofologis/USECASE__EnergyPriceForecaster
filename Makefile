.PHONY: install lint test test-features test-models \
	fetch fetch-data train predict backtest evaluate \
	clean clean-all format

PYTHON   := .venv/bin/python
PYTEST   := .venv/bin/pytest
PYLINT   := .venv/bin/pylint
PYTEST_COV:= .venv/bin/pytest --cov=src.gridprice --cov-report=term-missing

ZONE     ?= GR
API_KEY  ?= $(ENTSOE_API_KEY)
END_DATE ?= $(shell date +%Y-%m-%d)
CONFIGS  := config.yaml

# ── Setup ────────────────────────────────────────────────────────────────────

install:
	@echo "Installing dependencies…"
	python3 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt
	@echo "Done. Activate with: . .venv/bin/activate"

# ── Lint / Format ────────────────────────────────────────────────────────────

lint:
	@echo "Running pylint on src/gridprice/ …"
	@$(PYLINT) src/gridprice/ --disable=C,R,W0611 --max-line-length=100 || true

format:
	@echo "Formatting with ruff …"
	. .venv/bin/activate && pip install -q ruff && ruff format src/ tests/ || true

# ── Tests ────────────────────────────────────────────────────────────────────

test: test-features test-models
	@echo "All tests passed ✓"

test-features:
	@echo "Running feature tests …"
	$(PYTEST) tests/test_features.py -v --tb=short

test-models:
	@echo "Running model tests (may take ~5 min) …"
	$(PYTEST) tests/test_models.py -v --tb=short

# Run the full test suite with coverage
test-cov:
	$(PYTEST_COV) tests/ --tb=short

# ── Data ─────────────────────────────────────────────────────────────────────

# Fetch data from ENTSO-E (requires API key).
# Usage: make fetch ZONE=GR API_KEY=your-key-here
fetch fetch-data:
	@echo "Fetching ENTSO-E data for $(ZONE) …"
	@if [ -z "$(API_KEY)" ]; then \
		echo "ERROR: Set API_KEY=... or ENTOSE_API_KEY=..."; exit 1; fi
	$(PYTHON) -m src.gridprice.pipeline fetch \
		--config $(CONFIGS) \
		--zone $(ZONE)

# Download and cache ENTSO-E data without a key (public datasets only)
fetch-sample:
	$(PYTHON) -c "from gridprice.synthetic_data import SyntheticDataGenerator; \
		from datetime import date; \
		df = SyntheticDataGenerator(bidding_zone='$(ZONE)', seed=42, end_date=date.today()).generate(); \
		df.to_parquet('data/interim/sample_gr.parquet'); \
		print(f'Sample data: {df.shape}')"

# ── Train ───────────────────────────────────────────────────────────────────

# Train (or retrain) the XGBoost model.
# Usage: make train [API_KEY=...]
train:
	$(PYTHON) -m src.gridprice.pipeline train --config $(CONFIGS)

# Train without retraining if model is recent
train-check:
	$(PYTHON) -m src.gridprice.pipeline train --config $(CONFIGS) --no-retrain

# ── Predict ─────────────────────────────────────────────────────────────────

# Produce 24 h forecast.
# Usage: make predict [DATE=2024-06-01]
predict:
	$(PYTHON) -m src.gridprice.pipeline predict \
		--config $(CONFIGS) \
		--date $(END_DATE) \
		--horizon 24

# ── Backtest ────────────────────────────────────────────────────────────────

# Run expanding-window backtest.
# Usage: make backtest [DAYS=365]
backtest:
	$(PYTHON) -m src.gridprice.pipeline backtest \
		--config $(CONFIGS) \
		--backtest-days $(DAYS)

# ── Evaluate ─────────────────────────────────────────────────────────────────

# Run full evaluation: backtest + plots + CSV export.
# Usage: make evaluate END_DATE=2024-06-01
evaluate:
	$(PYTHON) notebooks/evaluation.py \
		--data-end $(END_DATE) \
		--n-estimators 50 \
		--step 168 \
		--out reports

# ── Daily pipeline (cron) ────────────────────────────────────────────────────
# Run fetch + train + predict in one shot.
# Add to crontab: 0 12 * * * cd /path/to/project && make daily
daily:
	$(PYTHON) -m src.gridprice.pipeline run --config $(CONFIGS)

# ── Notebook ─────────────────────────────────────────────────────────────────

# Launch JupyterLab with the evaluation notebook.
lab:
	$(PYTHON) -m jupyter lab notebooks/

# Convert evaluation.py to a notebook
to-nb:
	$(PYTHON) -c "import nbformat; \
		from nbconvert import PythonScriptToNotebook; \
		with open('notebooks/evaluation.py') as f: \
			nb = nbformat.v4.new_notebook(); \
			c = nbformat.v4.new_code_cell(f.read()); \
			nb.cells.append(c); \
		with open('notebooks/evaluation.ipynb','w') as f: \
			nbformat.write(nb, f)"

# ── Reports ──────────────────────────────────────────────────────────────────

reports/figures:
	mkdir -p reports/figures reports/metrics

# ── Clean ────────────────────────────────────────────────────────────────────

clean:
	rm -rf data/raw/*.parquet data/interim/*.parquet data/processed/*.parquet
	rm -rf reports/figures/* reports/metrics/*
	rm -rf models/*.pkl
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

clean-all: clean
	rm -rf .venv

# ── Help ─────────────────────────────────────────────────────────────────────

help:
	@echo "GridPrice Makefile"
	@echo ""
	@echo "Setup:"
	@echo "  make install              Create venv and install dependencies"
	@echo ""
	@echo "Data:"
	@echo "  make fetch ZONE=GR API_KEY=...   Fetch ENTSO-E data"
	@echo "  make fetch-sample           Download sample (no API key)"
	@echo ""
	@echo "ML pipeline:"
	@echo "  make train                 Train / retrain XGBoost model"
	@echo "  make predict               Produce 24 h forecast"
	@echo "  make backtest DAYS=365     Expanding-window backtest"
	@echo "  make evaluate              Full evaluation + plots + CSVs"
	@echo "  make daily                 fetch + train + predict (for cron)"
	@echo ""
	@echo "Tests:"
	@echo "  make test                  Run all tests"
	@echo "  make test-cov              Tests with coverage report"
	@echo ""
	@echo "Utilities:"
	@echo "  make lint                  Run pylint"
	@echo "  make format                Auto-format code with ruff"
	@echo "  make clean                 Remove generated data/reports"
	@echo "  make clean-all             Remove everything including .venv"
