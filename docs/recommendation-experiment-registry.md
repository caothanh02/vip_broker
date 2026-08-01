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
