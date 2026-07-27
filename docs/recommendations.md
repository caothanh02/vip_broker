# BTC/USDT recommendations

This repository can produce non-executable BTC/USDT 1-hour market recommendations. It is a
research output, not investment advice and not an automated trading system. It never sends an
order, reads an API key, manages money, or enables `BOT_MODE=live`.

`trading-bot recommend` accepts only validated, contiguous, closed UTC candles. It uses the
causal feature pipeline and the EMA/volume/ATR rule to identify a candidate. The entry reference
is the already-known signal close, solely for measuring future outcomes; it is not an order price.
The invalidation and target references are respectively 2x and 4x causal ATR from that close.

The default CLI is rule-only. It may emit `BUY_BIAS` for a rule candidate, otherwise `NEUTRAL`.
No ML probability is emitted unless an explicit, schema-compatible, production-eligible,
live-disabled inference model is supplied by a future offline integration. Missing, incompatible,
or ineligible models produce `NEUTRAL`, not an invented probability.

Outcomes are evaluated only after 1h, 4h, or 24h of future closed candles exist. Realized returns
deduct the configured entry/exit fee and slippage model. `BUY_BIAS` is directionally correct only
when that after-cost return is positive; `AVOID` is correct when it is non-positive. A stop/invalidation
touch wins an ambiguous target touch. Incomplete horizons are stored as
`insufficient_future_data` and never enter accuracy calculations.

Accuracy reports include coverage, directional accuracy, BUY_BIAS/AVOID precision, NEUTRAL rate,
and Brier score only when real ML probabilities exist. Reports with fewer than 30 applicable
resolved recommendations are labelled `inconclusive`; they must not be presented as a reliable
out-of-sample accuracy claim.

```powershell
uv run trading-bot recommend --input data/raw/btcusdt_1h.csv --output reports/recommendations/latest.json
uv run trading-bot evaluate-recommendations --input reports/recommendations/history.json --output reports/recommendations/accuracy.json
```

The latest recommendation and history are atomic JSON files under ignored `reports/recommendations/`.
They contain neither credentials nor broker/order identifiers and can be restored on restart.
