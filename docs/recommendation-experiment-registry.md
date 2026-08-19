# Recommendation experiment registry

This registry records predeclared, development-only research candidates. It is not a record of
investment recommendations, live-trading policies, or public accuracy claims.

## `baseline_ema_volume_atr_v1`

| Field | Value |
| --- | --- |
| Candidate ID | `baseline_ema_volume_atr_v1` |
| Status | baseline |
| Created | 2026-07-31 |
| Dataset role | development |
| Hypothesis | Baseline rule-only EMA/volume/ATR creates a measurable `BUY_BIAS` / `AVOID` / `NEUTRAL` recommendation baseline. |
| Rationale | Establish a deterministic rule-only reference before considering any new rule-based family. |
| Predeclared parameters | EMA 20/50/200; volume SMA 20 with multiplier 1.2; ATR 14. |
| Predeclared cost model | Entry fee 0.001; exit fee 0.001; entry slippage 0.0005; exit slippage 0.0005. These must match `BotSettings` exactly as `Decimal` values. |
| Intended folds | All three folds in Development walk-forward protocol v1. |
| Deterministic regression | `tests/unit/test_recommendations.py::test_optimized_backfill_matches_unmocked_causal_reference` and `tests/unit/test_recommendation_experiments.py::test_causal_backfill_recommendation_is_unchanged_by_future_mutation`. |
| Selection rule | No OOS evaluation or public accuracy claim from development results. |

The frozen development manifest is the only valid input for an experiment. Development output can
guide research design only; it cannot select a threshold against the sealed 2025 holdout, establish
an edge, or become investment advice. `AVOID` means avoid opening a long/buy and never means a
short instruction.

## Development walk-forward protocol v1 registration

The authoritative protocol is [Development walk-forward protocol v1](recommendation-research.md#development-walk-forward-protocol-v1).
It fixes the development range, three chronological folds, after-cost selection gates, tie-breaks,
and candidate budget before any new candidate is run.

- Maximum candidate families: two rule-based families, including the frozen baseline.
- Maximum variants: three pre-registered variants per family; six candidates total.
- ML candidates are prohibited in protocol v1.
- The baseline is immutable; a changed parameter or cost contract is a distinct candidate and
  consumes the v1 budget only when registered before execution.
- No candidate may be added, changed, or rerun as a new variant after looking at OOS 2025. Such a
  deviation requires a new protocol version rather than an edit to v1.

## Adding a candidate

Before implementation, a new candidate must record its hypothesis, rationale, creation date,
complete predeclared parameter and cost contract, intended v1 folds, development selection rule,
and deterministic regression test. Do not tune a candidate, cost assumption, confidence value, or
threshold after viewing a fold or any 2025 OOS result. The 2025 strict OOS period remains sealed
until exactly one development-selected policy is frozen.

## Development walk-forward protocol v2 registration

Protocol v1 completed with `baseline_ema_volume_atr_v1` as `no_policy_selected`. Protocol v2 is a
new immutable governance record; it does not alter the v1 candidate, folds, report, or decision.
It retains the development-only `[2022-01-01T00:00:00Z, 2025-01-01T00:00:00Z)` dataset and keeps
all 2025 OOS data sealed.

- Candidate budget: exactly one new rule-based candidate, `trend_pullback_ema_atr_v2`. No other
  v2 candidate or parameter variant is permitted.
- ML, probability filters, and post-result tuning are prohibited.
- Intended folds, selection gate, and deterministic tie-break are the three chronological folds
  and predeclared gates in [protocol v2](recommendation-research.md#development-walk-forward-protocol-v2).
- A v2 report can authorize strict OOS only if its sole candidate is `selected`; otherwise retain
  the safe `NEUTRAL` default and do not open OOS data.

## `trend_pullback_ema_atr_v2`

| Field | Value |
| --- | --- |
| Candidate ID | `trend_pullback_ema_atr_v2` |
| Status | pre-registered; not executed |
| Created | 2026-08-03 |
| Dataset role | development only |
| Hypothesis | A closed-candle trend pullback/reclaim with volume confirmation creates a measurable `BUY_BIAS` / `NEUTRAL` baseline. |
| Rationale | Test one rule family distinct from the EMA20/EMA50 crossover baseline without adding features or ML. |
| Immutable predicate | Current close > EMA200 and EMA20 > EMA50; previous close <= EMA20; current close > EMA20; current volume >= 1.2×volume SMA20; ATR14 > 0. |
| Output | `BUY_BIAS` only when every predicate is true; otherwise `NEUTRAL`. It never creates `AVOID` or a short instruction. |
| Predeclared parameters | EMA 20/50/200; volume SMA 20 with multiplier 1.2; ATR 14 for levels/risk reference only. |
| Predeclared cost model | Entry/exit fee `0.001`; entry/exit slippage `0.0005`, matched as `Decimal` values. |
| Intended folds | The three chronological development protocol v2 folds; no 2025 candle. |
| Deterministic regression | `tests/unit/test_trend_pullback_candidate.py` verifies the exact predicate, closed/gap handling, future-candle invariance, baseline dispatch, costs, and safety isolation. |
| Selection rule | Exactly one v2 candidate; it must pass every predeclared fold and pooled gate. No selection retains `NEUTRAL`; it is never an OOS or public accuracy claim. |

## Protocol V3/V4/V5 input-availability closure

V3 is `closed_input_unavailable`: its predeclared BTC/USDT 1-hour UTC input
`[2019-01-01T00:00:00Z, 2022-01-01T00:00:00Z)` failed absolute continuity in verified public
archives. It cannot freeze, execute, select, or authorize OOS, and its range/continuity rule must
not be changed after the finding. The authoritative closure record is
[Protocol V3](recommendation-protocol-v3.md).

- Historical candidate record: `dual_regime_reclaim_avoid_v3`; no parameter variant, threshold
  sweep, or ML candidate was permitted. This record is not an active V3 authorization.
- Hypothesis: a causal symmetric trend-regime reclaim/rejection signal can separately produce
  after-cost-quality `BUY_BIAS` and avoid-buy `AVOID` observations.
- Parameters: EMA 20/50/200, volume-SMA 20, volume multiplier 1.2, ATR 14.
- Cost contract: entry/exit fee `0.001`; entry/exit slippage `0.0005`, all exact `Decimal` values.
- Intended folds: three historical V3 folds over its unavailable independent target; they must
  never run and 2025 remains excluded and sealed.
- Historical regression record: the unexecuted V3 candidate was specified to test
  closed-candle causality, no future-candle dependence, immutable costs, `AVOID` avoid-buy
  semantics, and no broker/order/ML dependency. It is not a requirement or authorization for V4.
- Selection rule: closed; no V3 run or input lock is permitted.

Protocol V4 is `closed_input_unavailable`. Its mechanical Binance Vision audit covered 52 monthly
archives from 2017-09 through 2022-01: checksums verified, but 24 archives failed continuity or
timestamp policy and the longest continuous block was four months. It cannot audit, freeze,
execute, select, or authorize OOS. See [Protocol V4](recommendation-protocol-v4.md).

Protocol V5 is `closed_input_unavailable`. Its selected Gate public endpoint rejected the fixed
2019–2022 request with HTTP 400 because only the most recent 10,000 candlesticks are available.
No data was published and V5 cannot retry, alter its range, use a fallback source, freeze, execute,
select or authorize OOS. This is an input-availability finding, not a strategy result. See
[Protocol V5](recommendation-protocol-v5.md).

Protocol V6 permits only a bounded CoinAPI access verification. Its provider rejected the
authenticated metadata request with HTTP 403, so it cannot ingest, freeze, execute, select or
authorize OOS. See [Protocol V6](recommendation-protocol-v6.md).

Protocol V7 is closed_input_unavailable: its public-Binance-REST full-range audit did not meet
the fixed closed-continuous OHLCV validation. No data was published, and V7 cannot retry, alter
the source/range, execute research, select a policy or authorize OOS. See
[Protocol V7](recommendation-protocol-v7.md).
