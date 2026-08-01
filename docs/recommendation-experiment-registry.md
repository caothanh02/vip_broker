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
| Predeclared parameters | EMA 20/50/200; volume SMA 20 with multiplier 1.2; ATR 14. |
| Predeclared cost model | Entry fee 0.001; exit fee 0.001; entry slippage 0.0005; exit slippage 0.0005. These must match `BotSettings` exactly as `Decimal` values. |
| Selection rule | No OOS evaluation or public accuracy claim from development results. |

The frozen development manifest is the only valid input for an experiment. Development output can
guide research design only; it cannot select a threshold against the sealed 2025 holdout, establish
an edge, or become investment advice. `AVOID` means avoid opening a long/buy and never means a
short instruction.

## Adding a candidate

Before implementation, a new candidate must record a hypothesis, creation date, complete
predeclared parameters, intended development selection rule, and a deterministic regression test.
Do not tune a candidate, cost assumption, confidence value, or threshold after viewing 2025 OOS
results. The 2025 strict OOS period remains sealed until exactly one development-selected policy is
frozen.
