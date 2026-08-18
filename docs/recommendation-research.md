# Recommendation research protocol

This protocol governs research-only BTC/USDT 1-hour recommendations. It does not enable live
trading, broker access, order submission, or API-key use.

## Research-safe decision: candidate research closed

Development protocol v1 recorded `baseline_ema_volume_atr_v1` as `no_policy_selected`, and
development protocol v2 recorded `trend_pullback_ema_atr_v2` as `no_policy_selected`. The
`[2022-01-01T00:00:00Z, 2025-01-01T00:00:00Z)` development dataset is closed to ad-hoc candidate
research, parameter variants, and retries. This prevents data-snooping on the same development
period.

Strict OOS 2025 remains sealed and has not been opened. No candidate is selected, so no strict OOS
command, selection artifact, or further development walk-forward command is authorized. The
research and product-safe default is `NEUTRAL`; it is not investment advice or a trading
instruction. New candidate research requires a separate governance decision and a new,
independent development range that has not already been used for candidate selection.

## Protocol V3/V4 closure and Protocol V5 draft

Protocol V3 is `closed_input_unavailable`. Its independent target
`[2019-01-01T00:00:00Z, 2022-01-01T00:00:00Z)` failed the predeclared continuity requirement;
this is an availability finding, not a strategy result. V3 cannot freeze, execute, select, or
authorize OOS, and must not change its range or continuity policy after observing the archive.
See [the Protocol V3 closure record](recommendation-protocol-v3.md).

Protocol V4 is `closed_input_unavailable`: its 52-month public archive audit verified official
checksums but found 24 failed archives; the longest continuous block was only four months. This is
an input-availability finding, not a strategy result. V4 cannot audit, freeze, execute, select, or
authorize OOS. See [the Protocol V4 closure record](recommendation-protocol-v4.md).

Protocol V5 is `draft_source_selection_required`. It has no source, candidate, parameters,
dataset range, input lock, selection artifact, or OOS authorization. Its only future governance
decision may consider license, provenance and mechanical availability facts—never signals,
returns, accuracy, backtests, PnL or performance metrics. See
[the Protocol V5 draft](recommendation-protocol-v5.md).

## Development walk-forward protocol v1

This is the pre-registered governance protocol for development research. It applies before a new
recommendation candidate is implemented or evaluated. It does not authorize a strategy change,
ML candidate, OOS evaluation, public accuracy claim, broker access, or live trading.

### Immutable dataset boundary

- Development is exactly `[2022-01-01T00:00:00Z, 2025-01-01T00:00:00Z)`.
- Strict OOS is sealed from `2025-01-01T00:00:00Z` onward.
- All currently generated development data, manifests, histories, and reports are development-only
  research artifacts. None is strict OOS evidence or an accuracy claim.

The frozen development manifest is the required input identity. It locks the verified CSV,
metadata-sidecar, and anomaly-sidecar checksums, generation, UTC range, and audited interruption
provenance. It does not contain or lock fee/slippage assumptions. Those assumptions are locked by
the registered immutable candidate contract and recorded in the generated development experiment
report. An auditable selection decision therefore requires all three artifacts: the frozen dataset
manifest, the immutable candidate contract, and the generated experiment report with its cost
model; the manifest alone is not evidence of a cost assumption. The 2023-03-24 market interruption
is permanently non-tradable: it is a segment boundary, indicators warm up again after it, and no
recommendation, outcome, fill, or statistic may cross it. An unknown or unverified gap rejects the
input.

### Chronological folds

Each candidate uses these three non-overlapping future-validation folds. A fold's decision and
calibration window ends before its validation window starts; an expanding earlier window may be
used only for causal feature warm-up and predeclared calibration.

| Fold | Decision / calibration window | Future validation window |
| --- | --- | --- |
| 1 | `[2022-01-01T00:00:00Z, 2023-01-01T00:00:00Z)` | `[2023-01-01T00:00:00Z, 2023-09-01T00:00:00Z)` |
| 2 | `[2022-01-01T00:00:00Z, 2023-09-01T00:00:00Z)` | `[2023-09-01T00:00:00Z, 2024-05-01T00:00:00Z)` |
| 3 | `[2022-01-01T00:00:00Z, 2024-05-01T00:00:00Z)` | `[2024-05-01T00:00:00Z, 2025-01-01T00:00:00Z)` |

At decision candle T, features, signals, probabilities, confidence, and levels may use only
closed candles at or before T in the same continuous segment. Outcomes may use later closed
candles only after the decision has been persisted. Protocol v1 prohibits random shuffle,
random k-fold, forward filling from the future, and choosing a threshold after viewing a fold's
future validation outcome.

### Candidate registration and budget

A candidate must be registered in the experiment registry before implementation or execution. Its
entry must contain a hypothesis, rationale, complete immutable parameter and cost contract,
creation date, intended folds, and deterministic regression test. The frozen
`baseline_ema_volume_atr_v1` contract must not be changed.

Protocol v1 permits at most two rule-based candidate families in total, including the baseline,
and at most three pre-registered parameter variants per family (six candidates maximum). No ML
candidate, model filter, model calibration, or threshold-learning candidate is allowed in v1. ML
may be considered only by a later protocol version after a rule-based candidate has completed this
protocol with sufficient development evidence.

### Predeclared development selection gate

All outcomes use the candidate's immutable, after-cost fee/slippage contract. For every fold and
each 1h, 4h, and 24h horizon, report applicable resolved sample count, non-NEUTRAL coverage,
NEUTRAL rate, `BUY_BIAS` precision, `AVOID` avoid-buy precision, and after-cost directional
accuracy. An incomplete horizon is excluded from applicable metrics and reported separately; it
is never imputed or forward-filled.

For a candidate to pass development selection, every validation fold and every horizon must have
at least 30 applicable resolved observations, non-NEUTRAL coverage between 1% and 50%, and
after-cost directional accuracy above 50%. Across the pooled validation observations for each
horizon, it must also have at least 100 applicable resolved observations and a two-sided 95%
exact Clopper--Pearson lower confidence bound above 50%. These are development selection gates,
not an OOS gate or public accuracy claim.

Candidates must pass every stated gate across all three validation folds; one attractive fold
cannot select a policy. If more than one passes, choose in this order: higher minimum
fold-and-horizon directional accuracy; then higher minimum applicable sample count; then lower
maximum NEUTRAL rate; then the earlier registry entry. An unresolved exact tie selects no policy.
If no candidate passes, select no policy and retain the safe research default rather than adding
or tuning variants.

After exactly one policy is selected, seal an immutable development selection artifact before
opening any strict OOS 2025 history. The artifact binds the selected candidate and complete cost
contract to the checksum-locked development walk-forward report, frozen development manifest,
protocol version, and deterministic source identity. The sealer recomputes every fold and pooled
gate from the report evidence and derives the deterministic registry decision; it does not trust a
standalone `selection_decision` field. It is accepted only when the recomputed result is
`selected`; `no_policy_selected` means retain the safe `NEUTRAL` default and do not open OOS data.
No report from OOS 2025 may be read before that artifact is valid.

The source identity is the checked-out revision plus Git object IDs for `src/trading_bot`,
`pyproject.toml`, and `uv.lock`. The development runner records it in the report and fails closed
before opening its manifest when any executable input has staged, unstaged, or untracked changes.
Sealing and strict freeze/evaluation require the same clean identity. Documentation and ignored
reports/cache are deliberately outside this check, so they cannot create a false failure.

Selection authorization also replays the complete deterministic walk-forward computation from the
report's checksum-locked frozen development manifest, with the recorded candidate/cost contract.
The replay stays in memory and never overwrites a report, history, or artifact. Its folds, pooled
metrics, selection gate, decision, provenance, and safety locks must match the submitted report
exactly. A mutable JSON metric is therefore never sufficient to authorize strict OOS access.
`run_at` is runtime metadata only and may differ between runs. Historical recommendation
`created_at` values are instead the causal decision-candle close times, so they remain part of the
fully replayed fold evidence rather than introducing wall-clock variability.

### Leakage and overfitting guard

Do not tune features, thresholds, fee/slippage, horizons, date subsets, confidence rules, or
filters from a validation or OOS result outside this protocol. Do not add a replacement variant
solely because a fold or OOS result is unattractive. Any deviation, including a new candidate
family, altered budget, folds, gates, or tie-breaks, requires a new immutable protocol version;
protocol v1 must not be rewritten retroactively.

### Historical protocol status

V1 execution is complete and selected no policy. Do not rerun it or add a variant on its already
used development dataset.

## Development walk-forward protocol v2

Protocol v1 has already recorded `baseline_ema_volume_atr_v1` as `no_policy_selected`. Protocol
v2 is a separate immutable protocol: it does not rewrite v1 or select a policy retroactively.
Development remains exactly `[2022-01-01T00:00:00Z, 2025-01-01T00:00:00Z)`; strict OOS remains
sealed from `2025-01-01T00:00:00Z`, and no 2025 candle, manifest, report, or metric may be opened
for a v2 decision.

V2 pre-registers exactly one new rule-based family: `trend_pullback_ema_atr_v2`. Its closed UTC
1h predicate requires a long trend (close > EMA200 and EMA20 > EMA50), a previous-candle
pullback at or below EMA20 followed by a current-candle close above EMA20, current volume at least
1.2× volume SMA20, and positive ATR14. ATR is used only as the existing level/risk reference. The
candidate emits `BUY_BIAS` only when every condition holds and otherwise emits `NEUTRAL`; it adds
no `AVOID`, short, ML, threshold, or filter behavior. Its immutable cost contract is entry/exit
fees `0.001` and entry/exit slippage `0.0005` as `Decimal` values.

V2 uses the same three fixed chronological folds and full after-cost fold/pooled selection gate as
v1. Because its candidate budget is exactly one, the deterministic selection outcome is either
that candidate or `no_policy_selected`; no post-result variant may be added. A selected policy
must still be sealed and replay-verified before any strict OOS input can be read. If it is not
selected, retain the safe `NEUTRAL` default.

V2 execution is complete and selected no policy. Do not rerun it or add a variant on its already
used development dataset.

## Strict OOS selected-candidate evaluation

Strict OOS would require exactly one candidate bound in a valid sealed development selection
artifact. No such artifact exists after V1 and V2, so strict OOS commands must not be copied or
run. They fail closed before opening any OOS input.

The selection artifact, strict manifest, and strict report are ignored artifacts. They lock the
verified deterministic development replay plus dataset and sidecar provenance, and record the
exact source identity. They cannot alter candidate parameters, costs, features, or the registry. The
evaluation report binds the selection-artifact checksum together with strict provenance and the
exact statistical gate; it never creates a replacement candidate, submits an order, or
constitutes investment advice.

## Acceptance gates

Report 1h, 4h, and 24h independently. The primary statistic is after-cost directional accuracy
among non-NEUTRAL recommendations; it is not PnL, expected return, or investment advice.

A candidate is not eligible for an OOS claim unless each reported horizon has:

- at least 100 applicable, resolved recommendations;
- a two-sided 95% exact Clopper--Pearson confidence interval whose lower bound is above 50%;
- checksum-locked strict OOS provenance that passes history validation; and
- fixed fees/slippage and unchanged decision logic from the selected development experiment.

If these gates are not met, retain `NEUTRAL` as the safe default and label the output
`research-only` or `inconclusive`. `AVOID` remains an avoid-buy signal, never a short instruction.

## Audit requirements

Persist all research outputs outside Git. A report must identify the code commit, verified dataset
checksum/range, strict-OOS provenance, recommendation counts, coverage, applicable sample count,
confidence interval, and configured cost model. Evaluation report schema `1.2` uses the exact
Clopper--Pearson lower bound and records the independent statistical and strict-OOS claim gates.
No raw market data, models, reports, caches, or credentials may be committed.
