# Recommendation research protocol V6

**Status: `source_selected_access_verification_authorized`.** CoinAPI is preregistered only as a
potential historical OHLCV source for `BINANCE_SPOT_BTC_USDT`, UTC 1-hour candles over
`[2019-01-01T00:00:00Z, 2022-01-01T00:00:00Z)`.

The sole authorized action is `trading-bot verify-protocol-v6-coinapi-access`. It reads
`COINAPI_API_KEY` only from local `.env`, then makes one authenticated request to the fixed
CoinAPI filtered symbol-metadata collection endpoint. Its output is terminal-only, contains no
credential, and proves only authenticated access plus the fixed Spot BTC/USDT symbol identity. It
neither requests historical OHLCV nor writes a CSV, report, manifest or any other artifact.

This check does **not** verify historical-OHLCV entitlement, license/terms, provenance, candle
continuity, or any strategy result. Downloading data, freezing an input, candidate/parameter
work, recommendation, backtest, selection and OOS authorization all remain fail-closed. Source
selection must never use strategy signals, returns, accuracy, backtests, PnL or performance
metrics. OOS 2025 remains sealed and the project default is `NEUTRAL`.
