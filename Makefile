.PHONY: install format lint typecheck test download-data validate-data backtest backtest-real build-dataset train evaluate walk-forward dry-run report
DATA_START ?= 2024-01-01
DATA_END ?= 2024-02-01
DATA_FILE ?= data/raw/btcusdt_1h.csv
ML_DATASET_DIR ?= data/datasets/btcusdt_1h_v1
ML_MODEL_DIR ?= models/btcusdt_1h_logistic_v1
REPORT_FILE ?= reports/backtests/btcusdt_1h_baseline.json
install:
	uv sync --extra dev
format:
	uv run ruff format .
lint:
	uv run ruff check .
typecheck:
	uv run mypy src
test:
	uv run pytest
download-data:
	uv run trading-bot download-data --start $(DATA_START) --end $(DATA_END) --output $(DATA_FILE)
validate-data:
	uv run trading-bot validate-data --input $(DATA_FILE)
backtest:
	uv run trading-bot backtest --fixture --output reports/backtests/fixture.json
backtest-real:
	uv run trading-bot backtest --input $(DATA_FILE) --output $(REPORT_FILE)
build-dataset:
	uv run trading-bot build-dataset --input $(DATA_FILE) --output-dir $(ML_DATASET_DIR)
train:
	uv run trading-bot train --dataset-dir $(ML_DATASET_DIR) --output-dir $(ML_MODEL_DIR)
evaluate:
	uv run trading-bot evaluate
walk-forward:
	uv run trading-bot walk-forward
dry-run:
	uv run trading-bot dry-run
report:
	uv run trading-bot report
