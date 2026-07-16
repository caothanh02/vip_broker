# Contributor guide

- Python 3.12, typed code, `Decimal` for prices, quantities, balances, fees and PnL.
- Keep domain rules in `domain`, market IO in `data`, and do not couple strategy/risk to SQLAlchemy.
- Never use a candle that is not closed. Feature code must only use values available at the decision candle.
- Every order flows through `RiskEngine`; strategy and ML may never submit directly to a broker.
- `BinanceBroker` is a safety-locked design artifact: do not enable or test order submission.
- Do not commit secrets, databases, downloaded market data, binary models, or generated reports.
- Any strategy change needs a deterministic backtest regression test.

Checks: `uv run ruff format .`, `uv run ruff check .`, `uv run mypy src`, `uv run pytest`.
