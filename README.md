# BTC/USDT Spot Trading Bot

Safe, reproducible research, backtesting and paper-trading infrastructure for one market: BTC/USDT on 1-hour closed candles. It is not financial advice. Crypto is volatile; this repository does not enable live trading and no API key is needed.

## Architecture

`data → validation → features → EMA/volume/ATR strategy → optional ML filter → risk engine → simulated/dry-run broker → storage/reporting`. Strategy candidates are filled at the next candle open in the backtester. Stops win ambiguous intrabar OHLC ties, which is intentionally conservative.

## Install and run

Use Python 3.12 and [uv](https://docs.astral.sh/uv/): `uv sync --extra dev`, then `make format lint typecheck test`. Run the deterministic fixture backtest with `make backtest`. Use `make download-data` to invoke the public-data entry point; automated tests never make network calls.

## Real historical data

The public-only Binance Spot pipeline downloads closed `BTCUSDT` 1h candles, stores normalized UTC CSV without API keys, validates the result, and backtests it directly. Date-only download ranges mean UTC midnight; datetimes with a time require an explicit timezone. For audited long history, use `download-data --source binance-vision --start 2021-12-01 --end now --output data/raw/btcusdt_1h.csv --overwrite`; official ZIP checksums, archive timestamp exceptions, and CSV metadata checksums are all verified. See [docs/data.md](docs/data.md). Typical commands are `make download-data DATA_START=2024-01-01 DATA_END=2024-02-01`, `make validate-data`, and `make backtest-real`. Downloaded CSV and JSON reports are ignored by Git.

## ML workflow

`make build-dataset` builds labels only at rule-based entry candidates. The triple barrier is entry at next open, 2×ATR stop, 4×ATR target and 48-candle limit. Timeouts are excluded from binary baseline training but must be reported. `make train`, `make evaluate`, and `make walk-forward` are chronological only: scaler/selection train-only and threshold validation-only. Model metadata records ordered features/schema/checksum.

For the verified, interruption-segmented, sealed-holdout dataset contract and the exact immutable
split/label policy, see [docs/ml.md](docs/ml.md). The build command creates data only: it does not
train, tune, evaluate, backtest, or enable live trading.

## Dry-run and operations

`make dry-run` first replays validated closed candles through `DryRunBroker`; it writes a resumable paper state under ignored `data/dry_run/`. To use public market data only after replay/tests pass, run `uv run trading-bot dry-run --public --state data/dry_run/btcusdt_1h.state.json`. It bootstraps and recovers gaps with public REST, then reconnects the public closed-kline WebSocket with bounded exponential backoff. No API key is read and no Binance order endpoint exists in this path. State includes paper cash, position, pending signals, risk/circuit-breaker state and recent feature warm-up candles; it is saved atomically after each candle. Stop safely with `Ctrl+C`: the last completed candle has already been persisted. With the optional `api` extra, add `--health-port 8080` to expose local-only `/health/live`, `/health/ready`, and `/metrics`; the health payload includes the latest closed candle and stream status. See `docs/` for architecture, strategy, risk, ML, backtesting, deployment and operations.

## Recommendations only

`trading-bot recommend --input <validated.csv> --output reports/recommendations/latest.json` produces a BTC/USDT 1h `BUY_BIAS`, `NEUTRAL`, or `AVOID` research recommendation from closed candles. `AVOID` means avoid opening a long/buy; it is never a short instruction. Use `backfill-recommendations --input <validated.csv> --output reports/recommendations/history.json` to create an out-of-sample, candle-by-candle history before evaluating it. Strict OOS histories lock their UTC boundary and input checksum to prevent evidence from being mixed. It has no broker, order, balance, or live-trading path. Recommendation outcomes are resolved only once 1h, 4h, or 24h of future closed candles are available and are evaluated after the configured cost model. Run `evaluate-recommendations` on the persisted JSON history for coverage and accuracy; reports with fewer than 30 applicable outcomes are explicitly `inconclusive`. `research_claim_eligible` is never a development claim: it requires checksum-locked strict OOS provenance, at least 100 applicable resolved samples at every horizon, and a two-sided 95% exact Clopper--Pearson lower bound above 50% at every horizon. Evaluation report schema `1.2` records the separate statistical and strict-OOS eligibility gates. This is not financial or investment advice. See [recommendation documentation](docs/recommendations.md).

The V1 baseline and V2 trend-pullback development protocols both reached `no_policy_selected`.
The 2022–2024 development dataset is exhausted and closed to all V3 candidate selection or tuning.
Protocol V3 is `closed_input_unavailable`: its independent 2019–2021 input could not meet the
predeclared absolute-continuity requirement. This is not a strategy result; V3 cannot freeze,
execute, select, or authorize OOS. Protocol V4 is `draft_availability_audit_required`, with no
candidate or dataset yet; only a mechanical public-archive availability audit may propose a future
range. OOS 2025 remains sealed and has not been opened. The project default is `NEUTRAL`, with no
OOS accuracy claim, live trading, broker use, or order submission. See the
[research protocol](docs/recommendation-research.md),
[Protocol V3](docs/recommendation-protocol-v3.md), and
[Protocol V4](docs/recommendation-protocol-v4.md), and
[experiment registry](docs/recommendation-experiment-registry.md).

Strict OOS is a dormant safety boundary, not a currently authorized workflow. It would require a
clean deterministic source identity and a sealed artifact from exactly one selected development
candidate before opening any OOS input. V1 and V2 did not produce that artifact, so strict OOS
commands must not be copied or run. `no_policy_selected` means stop at the safe `NEUTRAL` default;
it never authorizes a replacement candidate, order submission, or live trading.

## Neutral-only operational status

`trading-bot operational-status --input <validated.csv> --output reports/operations/status.json`
verifies a local BTC/USDT 1h CSV plus its canonical metadata and anomaly sidecars, then publishes
an atomic, ignored status JSON. It reports identity/checksum state, candle range/count, freshness,
and any audited non-tradable interruption. It never downloads data, runs a recommendation,
backfill, walk-forward, experiment, or OOS evaluation. `trading-bot audit-safety --output
reports/operations/safety-audit.json` records the bounded runtime safety contract. Both commands
remain read-only with respect to data and do not load settings, credentials, brokers, orders, ML,
or network clients. They are observability only—not trading, prediction, investment advice, or
research/OOS evidence. See [operations documentation](docs/operations.md).

Before changing a recommendation rule or filter, follow the [research protocol](docs/recommendation-research.md): 2025 is a sealed holdout, and no recommendation claim is made without sufficient chronological development and OOS evidence.

## Live mode is locked

`BotSettings` rejects `BOT_MODE=live`; `BinanceBroker` always raises before any operation. This is a deliberate project safety boundary, even if environment variables or API secrets are present. Before any future long-running dry run: download/validate recent history, train a compatible model if ML is enabled, run walk-forward on unseen periods, and monitor data gaps/circuit-breaker state.
