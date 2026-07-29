# BTC/USDT recommendations

This repository can produce non-executable BTC/USDT 1-hour market recommendations. It is a
research output, not investment advice and not an automated trading system. It never sends an
order, reads an API key, manages money, or enables `BOT_MODE=live`.

`trading-bot recommend` accepts only validated, closed UTC candles. An otherwise-contiguous file
may contain a gap only when its checksum-verified anomaly sidecar identifies an audited market
interruption. Each continuous segment is treated as an independent series: feature warm-up resets
after the interruption, and neither a recommendation nor an outcome can use candles across it.
An unknown, unverified, or malformed gap fails closed. It uses the causal feature pipeline and the
EMA/volume/ATR rule to identify a candidate. For `BUY_BIAS`, the
entry reference is the already-known signal close, solely for measuring future outcomes; it is not
an order price. The invalidation and target references are respectively 2x and 4x causal ATR from
that close. `AVOID` means **avoid opening a long/buy**, not a bearish forecast or short
recommendation: its entry, target, and invalidation fields are null. Its accuracy is avoid-buy
directionality—after-cost return at the observation horizon is non-positive—not short PnL.

The default CLI is rule-only. It may emit `BUY_BIAS` for a rule candidate, otherwise `NEUTRAL`.
No ML probability is emitted unless an explicit, schema-compatible, production-eligible,
live-disabled inference model is supplied by a future offline integration. Missing, incompatible,
or ineligible models produce `NEUTRAL`, not an invented probability.

Use `backfill-recommendations` to build an out-of-sample history. It invokes the engine once per
closed candle with only the prefix available at that candle, then resolves outcomes only after the
history is created. It computes trailing features once and exposes only each decision candle plus
its predecessor to the rule. For strict OOS reports, provide `--evaluation-start` as an explicit
UTC close-candle timestamp. The history stores and locks its boundary plus input checksum; reruns
without the same boundary and input fail before modifying the history. Legacy v1 histories cannot
be adopted as strict OOS evidence: create a new output history instead. Strict accuracy reports
repeat the boundary, input SHA-256, and first/last input close timestamps for audit.

Outcomes are evaluated only after 1h, 4h, or 24h of future closed candles exist in the same
continuous segment. Realized returns
deduct the configured entry/exit fee and slippage model. `BUY_BIAS` is directionally correct only
when that after-cost return is positive; `AVOID` is correct when it is non-positive. A stop/invalidation
touch wins an ambiguous target touch. Incomplete horizons are stored as
`insufficient_future_data` and never enter accuracy calculations.

Accuracy reports include coverage, directional accuracy, BUY_BIAS/AVOID precision, NEUTRAL rate,
and Brier score only when real ML probabilities exist. Reports with fewer than 30 applicable
resolved recommendations are labelled `inconclusive`; they must not be presented as a reliable
out-of-sample accuracy claim. `research_claim_eligible` is a separate, stricter protocol gate:
every horizon needs at least 100 applicable resolved recommendations and a 95% confidence lower
bound above 50%. A technical `inconclusive: false` alone is never an OOS performance claim.

```powershell
uv run trading-bot recommend --input data/raw/btcusdt_1h.csv --output reports/recommendations/latest.json
uv run trading-bot backfill-recommendations --input data/raw/btcusdt_1h.csv --output reports/recommendations/history.json
uv run trading-bot backfill-recommendations --input data/raw/btcusdt_1h.csv --output reports/recommendations/oos_history.json --evaluation-start 2025-01-01T00:00:00Z
uv run trading-bot evaluate-recommendations --input reports/recommendations/history.json --output reports/recommendations/accuracy.json
```

The latest recommendation and history are atomic JSON files under ignored `reports/recommendations/`.
They contain neither credentials nor broker/order identifiers and can be restored on restart.
