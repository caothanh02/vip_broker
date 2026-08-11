# Recommendation research protocol V3

**Status: `closed_input_unavailable`.** V3 is closed because its predeclared independent input
cannot satisfy its absolute-continuity contract. This is neither a strategy success nor failure:
no candidate execution, performance evaluation, selection, or OOS activity occurred. It is not an
executable protocol, selection policy, performance report, or investment instruction.

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

The sole historical preregistered candidate was `dual_regime_reclaim_avoid_v3`. There is no
parameter sweep, threshold search, ML candidate, or replacement variant under V3.

## Closure record

V3 predeclared BTC/USDT 1-hour UTC `[2019-01-01T00:00:00Z, 2022-01-01T00:00:00Z)` with no
interruption or gap. Binance Vision verification found `BTCUSDT-1h-2019-03.zip` has 738 rather
than 744 rows, missing candles from `2019-03-12T02:00:00Z` through
`2019-03-12T07:00:00Z`. The corresponding daily 1-hour archive has 18/24 rows and daily 1-minute
archive has 1080/1440 rows; official checksums verified. Therefore V3 must not narrow or change
its range, whitelist this interruption, or relax continuity after observing availability.

The currently audited maintenance interruption is
`binance-spot-2023-03-24-trailing-stop-maintenance`. It lies outside the V3 target range, so V3's
historical input-lock design would have recorded an empty interruption list. It must never invent,
fill, or treat an interruption as tradable.

V3 cannot freeze an input, execute, select a policy, or authorize strict OOS. Strict OOS 2025
remains sealed and must not be read, downloaded, frozen, evaluated, or reported.

## Historical causal signal contract

The closed V3 candidate would have required BTC/USDT 1-hour UTC candles closed at decision time.
At candle T, features and rules could use only OHLCV and derived causal values at or before T in
the same continuous segment. Unknown gaps and unclosed candles fail closed. A validated
interruption is non-tradable, resets warm-up, and cannot be crossed by a signal or outcome.

The fixed candidate may use only existing causal EMA20, EMA50, EMA200, volume-SMA20, and ATR14:

- `BUY_BIAS`: close > EMA200, EMA20 > EMA50, previous close <= EMA20, current close > EMA20,
  current volume >= 1.2 times volume-SMA20, and ATR14 > 0.
- `AVOID`: close < EMA200, EMA20 < EMA50, previous close >= EMA20, current close < EMA20,
  current volume >= 1.2 times volume-SMA20, and ATR14 > 0.
- Otherwise: `NEUTRAL`.

`AVOID` means **avoid opening a long/buy**. It is never a short signal, target, stop, or short-PnL
instruction.

## Locked folds, costs, and comparators

The following are historical V3 folds and must never be run:

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

## Historical input-lock design only

The tracked input-lock design remains historical documentation only. It cannot be used to reopen
V3. A future protocol must establish its own independently reviewed contract.

- SHA-256 of this exact protocol config and of the frozen dataset manifest;
- dataset generation ID plus CSV, metadata-sidecar, and anomaly-sidecar SHA-256 values;
- BTC/USDT, 1-hour UTC range `[2019-01-01T00:00:00Z, 2022-01-01T00:00:00Z)`; and
- the exact audited interruption IDs in the new dataset.

The validator is pure and read-only. Because V3 is closed, every freeze/execution boundary fails
before opening a dataset. No lock, selection artifact, or strict-OOS action is authorized.

## Safety

Default output remains `NEUTRAL`. This repository remains research-only: it does not provide
investment advice, manage funds, use a broker or order endpoint, load credentials, or enable live
trading.
