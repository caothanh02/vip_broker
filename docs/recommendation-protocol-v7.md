# Recommendation research protocol V7

**Status: source_selected_availability_audit_authorized.** V7 selects only Binance Spot's
public, unauthenticated REST kline endpoint for a mechanical audit of BTCUSDT 1-hour UTC candles
over [2019-01-01T00:00:00Z, 2022-01-01T00:00:00Z). It requires no API key, subscription,
payment method or authenticated request.

## Sole permitted action

The audit-protocol-v7-binance-rest-availability command may make at most 27 public requests
(1,000 candles per page) and keeps all returned candles in memory. It accepts the source only if
the exact 26,304 closed candles are present, continuous, valid BTC/USDT Spot 1-hour UTC OHLCV,
and span the exact fixed boundaries. It writes no CSV, sidecar, cache, manifest or report.

The output is a terminal-only availability result. A successful audit is not a strategy result,
does not select a candidate, and does not authorize data persistence, backtesting,
recommendations, policy selection or strict OOS 2025.

Any missing/duplicate/open/invalid candle or network failure stops the audit. V7 then cannot alter
its source, range or continuity condition; it must be closed in a separately reviewed change. The
safe default remains NEUTRAL; no broker, order, live trading, ML or OOS path is authorized.
