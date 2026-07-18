# BTC/USDT Spot Trading Bot

Safe, reproducible research, backtesting and paper-trading infrastructure for one market: BTC/USDT on 1-hour closed candles. It is not financial advice. Crypto is volatile; this repository does not enable live trading and no API key is needed.

## Architecture

`data → validation → features → EMA/volume/ATR strategy → optional ML filter → risk engine → simulated/dry-run broker → storage/reporting`. Strategy candidates are filled at the next candle open in the backtester. Stops win ambiguous intrabar OHLC ties, which is intentionally conservative.

## Install and run

Use Python 3.12 and [uv](https://docs.astral.sh/uv/): `uv sync --extra dev`, then `make format lint typecheck test`. Run the deterministic fixture backtest with `make backtest`. Use `make download-data` to invoke the public-data entry point; automated tests never make network calls.

## Real historical data

The public-only Binance Spot pipeline downloads closed `BTCUSDT` 1h candles, stores normalized UTC CSV without API keys, validates the result, and backtests it directly. Date-only download ranges mean UTC midnight; datetimes with a time require an explicit timezone. See [docs/data.md](docs/data.md). Typical commands are `make download-data DATA_START=2024-01-01 DATA_END=2024-02-01`, `make validate-data`, and `make backtest-real`. Downloaded CSV and JSON reports are ignored by Git.

## ML workflow

`make build-dataset` builds labels only at rule-based entry candidates. The triple barrier is entry at next open, 2×ATR stop, 4×ATR target and 48-candle limit. Timeouts are excluded from binary baseline training but must be reported. `make train`, `make evaluate`, and `make walk-forward` are chronological only: scaler/selection train-only and threshold validation-only. Model metadata records ordered features/schema/checksum.

## Dry-run and operations

`make dry-run` uses a paper broker and cannot submit an exchange order. Configure public data and SQLite through `.env.example`; do not create a secret-bearing `.env` in source control. Optional FastAPI health endpoints are `/health/live`, `/health/ready`, `/metrics` after installing the `api` extra. See `docs/` for architecture, strategy, risk, ML, backtesting, deployment and operations.

## Live mode is locked

`BotSettings` rejects `BOT_MODE=live`; `BinanceBroker` always raises before any operation. This is a deliberate project safety boundary, even if environment variables or API secrets are present. Before any future long-running dry run: download/validate recent history, train a compatible model if ML is enabled, run walk-forward on unseen periods, and monitor data gaps/circuit-breaker state.
