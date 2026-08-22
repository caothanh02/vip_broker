# Recommendation research protocol V9

**Status: `source_selected_availability_audit_authorized`.** V9 is a new, independent
governance record following V8's closed authenticated request. It records a separately confirmed
paid CoinAPI entitlement context, but does **not** claim that historical OHLCV access works.

The sole future action authorized by V9 is a separately implemented, bounded availability audit
for `BINANCE_SPOT_BTC_USDT`, 1-hour UTC candles over
`[2019-01-01T00:00:00Z, 2022-01-01T00:00:00Z)`. It may make at most 28 requests (one identity
request plus 27 pages of at most 1,000 candles) and must validate symbol identity, closed candles,
UTC timestamps, duplicates, gaps and exact continuity in memory.

V9 does not currently contain that audit implementation and does not read a credential or make a
network request. It cannot persist/download an input, freeze a manifest, execute a candidate or
backtest, select a policy, or authorize strict OOS. The 2025 holdout remains sealed and the product
default remains `NEUTRAL`; there is no broker, order, live-trading or ML path.

V8 remains closed and is never retried. A V9 audit result is an input-availability finding, never a
strategy result, accuracy claim, or investment recommendation.
