# Recommendation research protocol V3

**Status: `candidate_preregistered_input_unfrozen`.** V3 records one candidate hypothesis before
an independent development input exists. It is not an executable protocol, selection policy,
performance report, or investment instruction. It does not authorize data download, validation,
walk-forward execution, selection, strict-OOS work, broker/order/risk activity, ML inference,
network access, credentials, or live trading.

The machine-validated contract is
[`config/recommendation_protocol_v3.yaml`](../config/recommendation_protocol_v3.yaml). Its
unbound `required_input_lock` section deliberately contains no fabricated digest. Any changed
field is a different protocol, not a V3 retry.

## Why V3 exists

V1 baseline and V2 trend-pullback both reached `no_policy_selected`. The 2022--2024 development
range is therefore exhausted evidence: it must not select, tune, retry, or otherwise rescue V3.
V2 emitted only `BUY_BIAS`/`NEUTRAL`, so it did not separately test an avoid-buy regime. V3 locks
exactly one technical hypothesis, without weakening a V1/V2 gate:

> A causal, symmetric trend-regime reclaim/rejection rule can identify both `BUY_BIAS` and
> `AVOID` observations whose after-cost directionality and mean return pass every preregistered
> fold and horizon gate on a previously unused development input.

The sole future candidate is `dual_regime_reclaim_avoid_v3`. There is no parameter sweep,
threshold search, ML candidate, or replacement variant under V3.

## Independent input requirement

The only V3 development target is
`[2019-01-01T00:00:00Z, 2022-01-01T00:00:00Z)`, BTC/USDT 1-hour UTC. It is a target range, not a
present local dataset, manifest, report, or evidence set. No V3 execution may begin until a new
dataset for that exact range has passed the existing validation pipeline and produced a
checksum-verified input-lock artifact.

The currently audited maintenance interruption is
`binance-spot-2023-03-24-trailing-stop-maintenance`. It lies outside the V3 target range, so V3's
future input lock must record an empty interruption list unless a separately audited interruption
is actually present in the new target dataset. It must never invent, fill, or treat an
interruption as tradable.

Strict OOS 2025 remains sealed. It must not be read, downloaded, frozen, evaluated, or reported;
an unbound V3 candidate cannot authorize it.

## Causal signal contract

All future inputs must be BTC/USDT 1-hour UTC candles closed at decision time. At candle T,
features and rules may use only OHLCV and derived causal values at or before T in the same
continuous segment. Unknown gaps and unclosed candles fail closed. A validated interruption is
non-tradable, resets warm-up, and cannot be crossed by a signal or outcome.

The fixed candidate may use only existing causal EMA20, EMA50, EMA200, volume-SMA20, and ATR14:

- `BUY_BIAS`: close > EMA200, EMA20 > EMA50, previous close <= EMA20, current close > EMA20,
  current volume >= 1.2 times volume-SMA20, and ATR14 > 0.
- `AVOID`: close < EMA200, EMA20 < EMA50, previous close >= EMA20, current close < EMA20,
  current volume >= 1.2 times volume-SMA20, and ATR14 > 0.
- Otherwise: `NEUTRAL`.

`AVOID` means **avoid opening a long/buy**. It is never a short signal, target, stop, or short-PnL
instruction.

## Locked folds, costs, and comparators

After a valid future input lock, V3 uses exactly these chronological folds:

| Fold | Calibration context | Future validation |
| --- | --- | --- |
| `fold_1` | `[2019-01-01, 2020-01-01)` | `[2020-01-01, 2020-09-01)` |
| `fold_2` | `[2019-01-01, 2020-09-01)` | `[2020-09-01, 2021-05-01)` |
| `fold_3` | `[2019-01-01, 2021-05-01)` | `[2021-05-01, 2022-01-01)` |

The immutable `Decimal` cost contract is entry/exit fee `0.001` and entry/exit slippage `0.0005`.
The fixed horizons are 1h, 4h, and 24h.

V3 does not maximize raw accuracy. Accuracy alone can reward sparse output, conceal costs, or say
nothing about the magnitude of an after-cost result. Every fold, direction, and horizon must have
at least 30 applicable resolved observations, including at least 10 for each of `BUY_BIAS` and
`AVOID`; non-neutral coverage must be inclusively within `[0.01, 0.50]`; after-cost directional
accuracy must be strictly greater than `0.50`; and mean after-cost return must be strictly greater
than `0`.

For pooled observations at every horizon, V3 additionally requires at least 100 applicable
resolved observations and a two-sided 95% exact Clopper--Pearson lower bound strictly greater than
`0.50`; mean after-cost return must also remain strictly greater than `0`. Incomplete future
horizons are reported separately and never imputed. These are development selection gates only,
not public accuracy evidence or an OOS claim.

There is one candidate and no tie-break. Any failed preregistered gate closes V3; a new idea,
parameter, cost, fold, or gate requires Protocol V4.

## Future input lock and execution boundary

The tracked config is intentionally unbound. A future input-lock artifact must bind all of the
following before any V3 execution boundary can be enabled:

- SHA-256 of this exact protocol config and of the frozen dataset manifest;
- dataset generation ID plus CSV, metadata-sidecar, and anomaly-sidecar SHA-256 values;
- BTC/USDT, 1-hour UTC range `[2019-01-01T00:00:00Z, 2022-01-01T00:00:00Z)`; and
- the exact audited interruption IDs in the new dataset.

The validator is pure and read-only: it validates a supplied lock object and never reads a dataset.
The current `input_unfrozen` status fails before that future activation boundary, so no lock,
selection artifact, or strict-OOS action is authorized today. A later status transition and lock
would require separate review; the lock's config SHA-256 prevents editing the candidate contract
after its data identity is fixed.

## Safety

Default output remains `NEUTRAL`. This repository remains research-only: it does not provide
investment advice, manage funds, use a broker or order endpoint, load credentials, or enable live
trading.
