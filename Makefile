.PHONY: install format lint typecheck test download-data validate-data backtest build-dataset train evaluate walk-forward dry-run report
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
	uv run trading-bot download-data
validate-data:
	uv run trading-bot validate-data
backtest:
	uv run trading-bot backtest
build-dataset:
	uv run trading-bot build-dataset
train:
	uv run trading-bot train
evaluate:
	uv run trading-bot evaluate
walk-forward:
	uv run trading-bot walk-forward
dry-run:
	uv run trading-bot dry-run
report:
	uv run trading-bot report
