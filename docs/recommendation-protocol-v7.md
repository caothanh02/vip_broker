# Recommendation research protocol V7

**Status: closed_input_unavailable.** V7 audited Binance Spot's public, unauthenticated REST
kline endpoint for BTCUSDT 1-hour UTC candles over
[2019-01-01T00:00:00Z, 2022-01-01T00:00:00Z). It required no API key, subscription, payment
method or authenticated request.

## Sole permitted action

The audit made public paginated requests in memory only. The full range did not pass the required
closed, continuous BTC/USDT Spot 1-hour UTC OHLCV validation, so no data was published. This is a
source-input availability finding, not a strategy or performance result.

## Immutable closure

V7 must not retry the audit, change its source/range/continuity condition, persist data, select or
evaluate a candidate, issue recommendations, or authorize strict OOS 2025. The safe default
remains NEUTRAL; no broker, order, live trading, ML or OOS path is authorized.
