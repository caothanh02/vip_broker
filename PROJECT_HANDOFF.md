# Project handoff — BTC/USDT research and recommendation system

**Purpose.** This is the durable starting point for maintainers and future AI agents. Read
[`AGENTS.md`](AGENTS.md) first, then this file, before changing code, running a research command,
or acting on a pull request.

## 1. Product status — do not overstate it

The repository is a **research, backtest, and paper/dry-run** system for one market:
BTC/USDT on closed 1-hour candles. It can produce research-only `BUY_BIAS`, `AVOID`, and
`NEUTRAL` recommendations, but it is **not** a live-trading product and is not investment advice.

The current safe product outcome is:

- Default recommendation: `NEUTRAL`.
- Development protocols V1 and V2 both ended `no_policy_selected`.
- Strict OOS from `2025-01-01T00:00:00Z` is sealed and has **not** been evaluated.
- No public accuracy, edge, PnL, or investment-performance claim is authorized.
- `BOT_MODE=live` is rejected and `BinanceBroker` is a deliberately disabled design artifact.

Do not interpret a historical backtest, development report, or a `BUY_BIAS` produced by the engine
as permission to trade or as a claim that the strategy is profitable.

## 2. Non-negotiable safety and engineering rules

These rules are also enforced by `AGENTS.md` and tests:

1. Use only **closed** candles. At decision candle T, features and signals may use data at or
   before T, never future rows.
2. Preserve interruption and gap boundaries. Never forward-fill or synthesize a missing candle.
3. Prices, quantities, balances, fees, and PnL use `Decimal`; do not introduce binary floats for
   financial arithmetic.
4. Strategies propose signals only. Any execution path would have to pass through `RiskEngine`;
   strategy and ML must never call a broker directly.
5. Do not enable, test, or implement real Binance order submission.
6. Never commit API keys, `.env`, downloaded market data, sidecars, reports, model binaries,
   databases, caches, or pytest temporary directories.
7. A strategy change needs deterministic regression coverage, including causal/no-look-ahead
   behavior.
8. Any new candidate, source, range, cost model, threshold, fold, or selection rule needs a new
   governance protocol. Do not edit a completed protocol to make an unsuccessful result pass.

## 3. Repository map

| Location | Responsibility |
| --- | --- |
| `src/trading_bot/domain/` | Typed, dependency-light domain models. |
| `src/trading_bot/data/` | Market-data adapters, CSV persistence, validation, ranges and interruption handling. |
| `src/trading_bot/features/` | Causal shared feature pipeline. |
| `src/trading_bot/strategy/` | Rule-only strategy predicates; no broker dependency. |
| `src/trading_bot/backtest/` | Conservative simulation: proposed entries fill at the next candle open; a stop wins an ambiguous intrabar stop/target tie. |
| `src/trading_bot/recommendations/` | Recommendation/backfill/outcome logic plus research governance, candidate registry, walk-forward and OOS seals. |
| `src/trading_bot/risk/` | Risk authorization; the only permitted execution gateway. |
| `src/trading_bot/execution/` | Safety-locked broker abstraction. |
| `src/trading_bot/runtime/` | Paper-only dry-run processing and persisted state. |
| `src/trading_bot/operations.py` | Read-only operational-status and safety-audit contracts. |
| `config/` | Immutable protocol/config records. |
| `docs/` | Architecture, data, strategy, risk, operations, research protocol, and protocol closure records. |
| `tests/unit/` | Safety, data, research, and regression coverage. |

## 4. Data and provenance model

The supported working instrument is `BTC/USDT`, 1-hour, UTC, closed candles.

- The public Binance REST/Vision pipeline normalizes data to a CSV with adjacent canonical
  `<csv>.metadata.json` and `<csv>.anomalies.json` sidecars.
- Validation checks symbol/timeframe/UTC, candle closure/order/continuity, OHLCV validity,
  duplicate/conflict handling, checksums, generation identity, and documented market
  interruptions.
- Generated data lives below `data/` and reports below `reports/`; both are intentionally ignored
  by Git. A future agent must validate that a local artifact exists and matches its sidecars rather
  than assuming it is present from this handoff.
- The prior development interval was `[2022-01-01T00:00:00Z, 2025-01-01T00:00:00Z)`. The audited
  2023-03-24 interruption is permanently non-tradable; features warm up again afterwards and no
  statistic/outcome/fill may cross it.

Useful read-only commands, when a valid local CSV and sidecars exist:

```powershell
uv run trading-bot operational-status --input data/raw/btcusdt_1h.csv --output reports/operations/status.json
uv run trading-bot audit-safety --output reports/operations/safety-audit.json
```

These commands must not be represented as recommendations, research evidence, or trade approval.

## 5. Recommendation and research controls

The engine intentionally separates a causal decision from later outcome evaluation:

- `recommend` produces a latest research observation.
- `backfill-recommendations` creates a causal, candle-by-candle history from prefixes only.
- Outcomes resolve only after 1h, 4h, or 24h future closed candles exist. Resolved outcomes are
  immutable when later partial input is evaluated.
- `AVOID` means “avoid opening a long/buy”; it is never a short order or bearish trade.
- Accuracy reports track coverage, direction, `BUY_BIAS`/`AVOID` precision, neutral rate, Brier
  score/calibration where applicable, and explicitly report insufficient samples.

The historical development candidates are closed:

| Protocol | Candidate / outcome | Status |
| --- | --- | --- |
| V1 | `baseline_ema_volume_atr_v1` | Executed; `no_policy_selected`. |
| V2 | `trend_pullback_ema_atr_v2` | Executed; `no_policy_selected`. |
| Strict OOS | 2025 onward | Sealed; cannot run without a valid selected-candidate artifact. |

Development selection requires every fixed fold/horizon to meet its preregistered coverage,
sample-count, after-cost accuracy and exact confidence-bound gates. A tempting isolated result does
not authorize tuning, a replacement candidate, or OOS access.

## 6. Independent-source protocol history

These are **availability/provenance** findings, not strategy findings. Closed protocols must not be
retried with a new range, fallback, or relaxed rule.

| Protocol | Source / target | Result and allowed state |
| --- | --- | --- |
| V3 | Binance Vision, 2019–2022 | Closed: predeclared absolute continuity failed. |
| V4 | Binance Vision archive audit | Closed: numerous failed archives; longest continuous block was insufficient. |
| V5 | Gate.io public OHLCV, 2019–2022 | Closed: provider history window too short. |
| V6 | CoinAPI metadata identity | Symbol identity check only; historical OHLCV entitlement remains unverified. |
| V7 | Public Binance REST, 2019–2022 | Closed: fixed range did not meet closed-continuous OHLCV validation. |
| V8 | CoinAPI authenticated availability audit | Closed: provider rejected the one preregistered audit. |
| V9 | Paid CoinAPI authenticated availability audit | Closed: provider rejected the one preregistered audit. |
| V10 | Tardis Binance spot trades, 2019-04-01 to 2022-04-01 | Pending code review; only a future one-request metadata identity probe is preregistered. |

### Protocol V10 — exact handoff state

The implementation is on branch `codex/protocol-v10-tardis-preregistration` at commit
`24caed81c9b60c688ae4b280ace62da0d1387fa9`. It was prepared as pull request **#34** against
`master`; always verify its current remote/CI/review state before acting.

V10 is intentionally narrow:

- Its proposed independent interval is `[2019-04-01T00:00:00Z, 2022-04-01T00:00:00Z)`.
- It pins a single future Tardis instrument metadata endpoint for `binance/BTCUSDT`, including the
  expected spot/base/quote identity and `availableSince` response fields.
- It does **not** read `TARDIS_API_KEY`, make a network call, download trade CSVs, aggregate OHLCV,
  persist an input, execute research, select a policy, or authorize OOS.
- A metadata response cannot prove paid historical-download entitlement, licence suitability,
  data completeness, or candle quality.
- Tardis daily files are partitioned by provider local-arrival time whereas future candles would
  require exchange-event UTC timestamps. No finite boundary partition set, event filter, duplicate
  policy, or continuity evidence has been preregistered. Ingest remains fail-closed until a later,
  independent protocol binds all of them.

Do **not** add a Tardis downloader simply because an API key is available. First merge/review V10;
then, only after a user-approved paid plan and the permitted metadata probe, prepare a separate
protocol for any subsequent entitlement or raw-trade availability audit.

## 7. Current Git snapshot and collaboration workflow

At the time this handoff was written, local `master`/`origin/master` was
`7d749788ac0dc8eb3d835a5d0b41d0aea0c22790` (V9 closure). The V10 branch above is intentionally
not assumed to be merged.

Before any work:

```powershell
git status --short --branch
git fetch --no-write-fetch-head origin
git log --oneline --decorate -5
```

For any PR, verify its base/head SHA, Draft/Ready state, CI outcome, review decision, changed file
list, and clean worktree. Use a `codex/`-prefixed branch, make focused commits, push the branch,
and create/review a PR. Never force-push, reset unrelated work, delete a worktree, or write
directly to `master` without an explicit request.

On this Windows host, GitHub CLI authentication may be absent or stale across environments. Treat
an HTTP 401 or `gh auth status` failure as a hard stop for GitHub API actions; ask the user to run
`gh auth login -h github.com` rather than storing a token in the repository, `.env`, or a command
log. Git transport may need the user’s configured credential helper. Re-authenticate and rerun all
PR preflight checks; never bypass them.

## 8. Quality gate and safe local workflow

Run these before committing a source or documentation change:

```powershell
uv run ruff format .
uv run ruff check .
uv run mypy src
uv run pytest -q --basetemp .pytest-handoff
git diff --check
```

The `--basetemp` directory is generated and ignored. Do not stage it. If Windows locks it, leave it
alone; do not use destructive cleanup outside the repository or change ownership/ACLs merely to
remove test output.

When adding a strategy or research change, also run its targeted deterministic causal regression
and document the fixed cost/fee/slippage assumptions. Tests must not make network requests.

## 9. Next-action decision tree

1. **Immediate:** review PR #34 again against its exact SHA. If CI and review are clean, merge it
   through a regular merge commit, then fast-forward local `master`. This only records V10
   governance; it does not authorize a probe or data activity.
2. **Only with user approval and a valid Tardis plan/key:** implement and review the single V10
   metadata identity probe. It must issue at most the preregistered request, redact credentials,
   persist no provider data, and fail closed for every non-identity result.
3. **After a successful metadata probe:** stop. It is not a data-availability result. Propose a new
   source-governance protocol that explicitly covers licence/entitlement, raw-trade partition
   boundaries, exchange-event filtering, duplicate identity/order, empty files, interruptions,
   aggregation, checksums, and continuity. Do not download first and write policy afterwards.
4. **Only after an approved independent dataset passes all validation:** freeze it, perform the
   predeclared development workflow, and let the selection gate decide. Do not open 2025 OOS unless
   exactly one candidate is selected and the selection artifact validates.
5. **Never make live trading the next step.** Even a future successful OOS report would require a
   separate product/risk/governance decision; the present repository is deliberately locked to
   research and paper/dry-run operation.

## 10. What to read for a specific task

- System and module boundaries: `docs/architecture.md`, `docs/data.md`, `docs/backtesting.md`.
- Recommendation behavior: `docs/recommendations.md` and `src/trading_bot/recommendations/engine.py`.
- Research controls and anti-overfitting policy: `docs/recommendation-research.md`.
- Candidate history: `docs/recommendation-experiment-registry.md`.
- Per-source closure evidence: `docs/recommendation-protocol-v3.md` through
  `docs/recommendation-protocol-v10.md`.
- Read-only health/operational observability: `docs/operations.md` and
  `src/trading_bot/operations.py`.
- Existing regression intent: `tests/unit/test_recommendations.py`,
  `tests/unit/test_recommendation_*.py`, and protocol-specific tests.

If information in this handoff conflicts with a checked-in immutable protocol or the actual remote
Git history, treat the protocol and Git history as authoritative, report the discrepancy, and do
not widen the workflow until it is resolved.
