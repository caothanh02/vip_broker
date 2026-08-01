# Recommendation research protocol

This protocol governs research-only BTC/USDT 1-hour recommendations. It does not enable live
trading, broker access, order submission, or API-key use.

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

After exactly one policy is selected, freeze its complete candidate contract and selection record
before opening any strict OOS 2025 history. No report from OOS 2025 may be read before that
decision.

### Leakage and overfitting guard

Do not tune features, thresholds, fee/slippage, horizons, date subsets, confidence rules, or
filters from a validation or OOS result outside this protocol. Do not add a replacement variant
solely because a fold or OOS result is unattractive. Any deviation, including a new candidate
family, altered budget, folds, gates, or tie-breaks, requires a new immutable protocol version;
protocol v1 must not be rewritten retroactively.

### Next action

Execute the registered candidate with the protocol runner after review of its registration:

```powershell
uv run trading-bot run-recommendation-walk-forward --manifest reports/research/manifests/development.json --candidate baseline_ema_volume_atr_v1 --output reports/research/walk-forward/baseline_ema_volume_atr_v1.json
```

The runner executes these exact v1 folds against the frozen development manifest, writes an
ignored atomic development-only report, and can select only `selected` or `no_policy_selected`
within development. It does not download, freeze, evaluate, or otherwise open OOS 2025.

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
