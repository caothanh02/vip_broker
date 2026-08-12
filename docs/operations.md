# Operations

## Neutral-only operational observability

The recommendation programme is closed: V1 and V2 both ended in
`no_policy_selected`, the default recommendation is `NEUTRAL`, and strict OOS 2025 remains sealed.
The following commands observe locally published data without changing that state:

```powershell
uv run trading-bot operational-status --input data/raw/btcusdt_1h.csv --output reports/operations/status.json
uv run trading-bot audit-safety --output reports/operations/safety-audit.json
```

`operational-status` accepts only a regular, non-symlink CSV under `data/raw/`, with its adjacent
canonical `<csv>.metadata.json` and `<csv>.anomalies.json` sidecars. Before publication it verifies
the metadata-to-CSV checksum, sidecar identity/checksum/generation, closed UTC BTC/USDT 1h candle
validation, OHLCV continuity, and any audited interruption. Its JSON includes dataset identity,
source, requested/stored range and count, freshness from the last closed candle, checksum
verification mode, and a clear
distinction between continuous tradability and audited non-tradable intervals.

`audit-safety` reports stable machine-readable findings about this bounded operational path. It is
not a certification and intentionally does not call shell or Git. It does not load settings or
credentials; the live-mode lock is a build contract, not permission to run a live process.

Outputs are atomically written only below `reports/operations/*.json`, are ignored by Git, and
refuse overwrite unless `--overwrite` is supplied after validation. Absolute paths, traversal,
symlinks, unsupported extensions, output outside that namespace, invalid sidecars, and tampered
checksums fail closed. A failed validation or publication leaves an existing output untouched.

Neither command calls a broker, order, `RiskEngine`, `DryRunBroker`, recommendation engine,
backfill, evaluator, ML inference, REST/WebSocket client, or authenticated endpoint. No API key or
secret is required, read into the output, or logged. The status is observability only: it is not a
recommendation, accuracy report, research evidence, strict-OOS evaluation, or investment advice.

### Daily Windows read-only check

`scripts/run-operational-check.ps1` is an orchestration-only runbook for the frozen development
dataset. It refuses to run unless the local worktree is clean, the branch is `master`, and local
`HEAD` matches the existing `origin/master` ref. It checks the three canonical development artifacts
are regular files rather than symlinks, then uses `uv --offline` to run only `audit-safety` and
`operational-status`. It creates timestamped reports under ignored `reports/operations/` and never
uses `--overwrite`, downloads data, reads credentials, or starts a research, recommendation,
dry-run, broker, order, REST, WebSocket, ML, or OOS workflow.

During first installation only, `-BootstrapSmoke` permits one manual smoke run while exactly this
script and this runbook are the only uncommitted files. It rejects every other changed or untracked
path. The scheduled task never uses this switch and therefore always requires a clean worktree.

The optional, local-only Windows Scheduled Task `VipBrokerNeutralOperationalCheck` runs this script
daily at 09:00 local time using a limited interactive user token. It must not be configured to run
with highest privileges, with a password, or with an execution-policy bypass. A failed check only
returns a non-zero exit code; it does not delete, repair, replace, or download artifacts.

Inspect its local configuration without changing Task Scheduler state:

```powershell
powershell -NoLogo -NoProfile -File scripts/install-operational-check-task.ps1 -Inspect
```

Installation is an explicit opt-in and is refused unless the repository is clean `master` matching
its existing `origin/master` ref. It does not replace an existing task, elevate privileges, use a
service account, or run the task after registration:

```powershell
powershell -NoLogo -NoProfile -File scripts/install-operational-check-task.ps1 -Install
```

Both scripts only write timestamped, Git-ignored reports below `reports/operations/`. The runner
requires parseable audit/status JSON with the neutral, sealed-OOS safety contract before printing a
success summary; an exit code alone is never treated as a passing audit.

## Dry-run operations

The separate dry-run service remains a paper-only workflow. Use structured events with
signal/order/model identifiers. Reconnect public websockets with exponential backoff and recover
gaps through REST before dry-run processing resumes. Alert on start/stop, disconnects, gaps,
candidate and risk decisions, position events, breakers, and model incompatibility. Telegram is
optional: lack of configuration must not stop the bot.
