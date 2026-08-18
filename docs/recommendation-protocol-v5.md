# Recommendation research protocol V5

**Status: `source_selected_availability_audit_required`.** V5 selected Gate.io's public Spot
`BTC_USDT` candle endpoint only for a mechanical availability audit over
`[2019-01-01T00:00:00Z, 2022-01-01T00:00:00Z)`. The request contract is exactly BTC/USDT Spot,
UTC 1-hour candles, at most 1,000 candles per page, paced at one request per second, with no
authentication. This is not a strategy, candidate, development input, input lock or OOS approval.

## Source-governance boundary

Gate was selected using source facts only:

- license or terms verification;
- source provenance;
- BTC/USDT Spot 1-hour UTC identity;
- closed candles; and
- absolute continuity.

No signal, return, accuracy, backtest, PnL or performance metric may influence source selection
or the audited range.

`download-protocol-v5-gate-availability` accepts no symbol, timeframe, range, API-key or retry
override. It writes no output until all 26,304 requested candles pass strict closed-candle, UTC,
identity, OHLCV, duplicate and continuity checks. A gap, malformed page, rate-limit exhaustion or
incomplete coverage fails closed without replacing an existing output.

A subsequent reviewed revision must record the audit outcome and separately preregister one
candidate, parameters, costs, folds and gates before it can freeze a validated input.

Strict OOS from 2025 remains sealed. Until every future governance gate is passed, the default is
`NEUTRAL`; no broker, order, live-trading, ML, credential, candidate execution, selection or OOS
activity is authorized.
