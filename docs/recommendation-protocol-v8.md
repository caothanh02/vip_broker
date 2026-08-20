# Recommendation research protocol V8

**Status: `source_selected_historical_availability_audit_authorized`.** Protocol V8 is a new,
independent CoinAPI input-availability protocol. It does not alter V1--V7, their inputs, candidate
decisions, or the sealed 2025 OOS boundary.

The only authorized command is
`trading-bot audit-protocol-v8-coinapi-historical-availability`. It reads `COINAPI_API_KEY` from
local `.env`, verifies exactly one `BINANCE_SPOT_BTC_USDT` Spot identity, then requests the fixed
UTC 1-hour range `[2019-01-01T00:00:00Z, 2022-01-01T00:00:00Z)` in memory. It makes at most 28
requests: one metadata identity request and at most 27 pages of 1,000 candles.

Every received row must be a closed UTC 1-hour BTC/USDT candle with finite OHLCV, coherent trade
timestamps, strict order, no duplicates and absolute continuity. An HTTP error, malformed payload,
open candle, gap, duplicate, count mismatch or request-bound breach fails closed. There is no
fallback source, range change, automatic retry, CSV/report/manifest write or data cache.

A successful audit proves only authenticated API read access and mechanical availability at audit
time. It does not verify CoinAPI licence/terms, authorize input persistence, candidate or parameter
work, feature/signal calculation, recommendation/backtest execution, selection or strict OOS.
Those actions remain fail-closed; the default remains `NEUTRAL`, and no broker, order, live-trading
or ML path is loaded.
