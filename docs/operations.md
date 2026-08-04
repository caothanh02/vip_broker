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
range/count, freshness from the last closed candle, checksum verification mode, and a clear
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

## Dry-run operations

The separate dry-run service remains a paper-only workflow. Use structured events with
signal/order/model identifiers. Reconnect public websockets with exponential backoff and recover
gaps through REST before dry-run processing resumes. Alert on start/stop, disconnects, gaps,
candidate and risk decisions, position events, breakers, and model incompatibility. Telegram is
optional: lack of configuration must not stop the bot.
