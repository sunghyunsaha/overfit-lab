# Overfit Lab - one command per task. Everything is driven by YAML configs.
PY ?= python3
export PYTHONPATH := src

.PHONY: install data data-check data-hourly hourly data-fallback demo all figures docs test lint clean

install:            ## install pinned dependencies and the package
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e . --no-deps

data:               ## download Binance OHLCV into data/ (primary real-data source)
	$(PY) -m overfitlab.cli download --config configs/full.yaml --source binance

data-check:         ## 10-second pre-flight check of the Binance source (run before `make data`)
	$(PY) -m overfitlab.cli check-data --config configs/full.yaml

data-hourly:        ## download Binance 1-hour OHLCV into data/
	$(PY) -m overfitlab.cli download --config configs/hourly.yaml --source binance

hourly:             ## E3 on 1-hour bars (needs `make data-hourly`) -> reports/hourly/
	$(PY) -m overfitlab.cli e3 --config configs/hourly.yaml

data-fallback:      ## download CoinMetrics daily closes into data/ (used if Binance is unreachable)
	$(PY) -m overfitlab.cli download --config configs/full.yaml --source coinmetrics

demo:               ## fast end-to-end run on a small synthetic sample -> reports/demo/
	$(PY) -m overfitlab.cli all --config configs/demo.yaml

all:                ## full pipeline: E1-E5, figures, README/REPORT rendered from results
	$(PY) -m overfitlab.cli all --config configs/full.yaml

figures:            ## redraw all figures from results + saved arrays (no recomputation)
	$(PY) -m overfitlab.cli figures --config configs/full.yaml

docs:               ## re-render README.md / REPORT.md / INTERVIEW_PREP.md from results JSON
	$(PY) -m overfitlab.cli docs --config configs/full.yaml

test:               ## run the test suite
	$(PY) -m pytest

lint:               ## ruff lint + format check
	ruff check .
	ruff format --check .

clean:
	rm -rf artifacts reports/demo .pytest_cache .ruff_cache
