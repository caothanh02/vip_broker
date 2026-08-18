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
every horizon needs at least 100 applicable resolved recommendations and a two-sided 95% exact
Clopper--Pearson confidence lower bound above 50%. The history must also be a checksum-locked,
validated strict OOS history. Development, legacy, and non-strict histories always report
`research_claim_eligible: false`, even when their statistical result passes. A technical
`inconclusive: false` alone is never an OOS performance claim.

Evaluation report schema `1.2` records `statistical_gate_passed`, the two-sided 95% exact
Clopper--Pearson lower bound, strict-OOS validation, and final claim eligibility. This changes the
evaluation report only; persisted recommendation history remains schema `1.1`.

```powershell
uv run trading-bot recommend --input data/raw/btcusdt_1h.csv --output reports/recommendations/latest.json
uv run trading-bot backfill-recommendations --input data/raw/btcusdt_1h.csv --output reports/recommendations/history.json
uv run trading-bot backfill-recommendations --input data/raw/btcusdt_1h.csv --output reports/recommendations/oos_history.json --evaluation-start 2025-01-01T00:00:00Z
uv run trading-bot evaluate-recommendations --input reports/recommendations/history.json --output reports/recommendations/accuracy.json
```

The latest recommendation and history are atomic JSON files under ignored `reports/recommendations/`.
They contain neither credentials nor broker/order identifiers and can be restored on restart.

Operational status is intentionally separate from recommendation generation. Use
`trading-bot operational-status` and `trading-bot audit-safety` only for read-only local dataset
and safety-contract observability; they do not create recommendations, accuracy evidence, or
strict-OOS output. See [operations](operations.md).

## Freeze development research input

Before a walk-forward experiment, freeze the verified development dataset—not a strict OOS history—with:

```powershell
uv run trading-bot freeze-recommendation-research --input data/raw/btcusdt_1h_development_2022_2024.csv --output reports/research/manifests/development.json
```

The command rejects absent or checksum-invalid metadata/anomaly sidecars, unverified gaps, open
candles, non-BTC/USDT 1h UTC data, incorrect development range, and any interruption other than
the audited 2023-03-24 13:00 UTC non-tradable event. Its ignored atomic manifest records the CSV,
metadata, and anomaly-report SHA-256 values, generation ID, official-online verification mode, and
interruption URLs. This manifest is a prerequisite for walk-forward research; it is not an accuracy
report, recommendation history, strict OOS evidence, or an instruction to trade. Run with
`--overwrite` only after validation completes if replacing an existing manifest is intentional.
The output is mandatory: it must be a `.json` file directly or recursively under
`reports/research/manifests/`; traversal, source/docs paths, dataset files, and sidecars are
rejected before the dataset is read.

## Development experiments

The frozen v1 baseline is `baseline_ema_volume_atr_v1`. Protocol v2 separately preregistered and
executed `trend_pullback_ema_atr_v2`; its development-only result was `no_policy_selected`, so it
has no selected policy or OOS claim. Its hypothesis and fixed parameters are listed in the
[experiment registry](recommendation-experiment-registry.md). The historical baseline command is:

```powershell
uv run trading-bot run-recommendation-experiment --manifest reports/research/manifests/development.json --candidate baseline_ema_volume_atr_v1 --output reports/research/experiments/baseline_ema_volume_atr_v1.json
```

The command revalidates manifest, CSV, metadata, anomaly sidecar, generation, market interruption,
and closed-candle continuity before causal backfill. Its atomic output must be under ignored
`reports/research/experiments/`, is always marked `research_role: development`, and forces
`research_claim_eligible: false`. Development metrics are not strict OOS evidence, public accuracy
claims, investment advice, or an instruction to trade. The 2025 OOS period remains sealed.
The candidate also locks its four `Decimal` fee/slippage rates; a local settings override is rejected
rather than silently producing different metrics under the same candidate ID.

## Current research decision

V1 baseline and V2 trend-pullback development reports both recorded `no_policy_selected`. Their
shared 2022–2024 development dataset is exhausted. [Protocol V3](recommendation-protocol-v3.md)
is `closed_input_unavailable` because its independent input did not meet predeclared absolute
continuity; this is not a strategy result and cannot be repaired by changing its range or rules.
[Protocol V4](recommendation-protocol-v4.md) is also `closed_input_unavailable` after its
availability audit found no adequate continuous range. [Protocol V5](recommendation-protocol-v5.md)
is also `closed_input_unavailable`: Gate's public endpoint could not serve the preregistered
2019–2022 input because it retains only the most recent 10,000 candlesticks. No data was
published and no V5 candidate or authorization exists. The safe project default is `NEUTRAL`; it
is not investment advice or a trading instruction.

No selected candidate or sealed selection artifact exists. Therefore strict OOS 2025 must remain
sealed: do not copy or run strict OOS freeze/evaluation commands, and do not create a selection
artifact. The project has no OOS accuracy claim and no live-trading, broker, or order-submission
path. Any future candidate research needs an explicit governance decision and an independent,
previously unused development range.
