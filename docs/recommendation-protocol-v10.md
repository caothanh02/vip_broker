# Recommendation research protocol V10

**Status: `source_selected_metadata_identity_probe_authorized`.** V10 is a new independent
source-governance record after V9 closed. It uses Tardis Binance Spot **trade** data, not a retry or
fallback inside V9.

The fixed independent source interval is `[2019-04-01T00:00:00Z, 2022-04-01T00:00:00Z)`: 26,304
hourly UTC intervals. Future candle construction, if ever authorized separately, must aggregate
only Binance exchange-event timestamps to one-hour OHLCV. It may not use provider local-arrival
timestamps, synthesize missing candles, or cross an unverified gap.

The only currently authorized future action is one authenticated Tardis **instrument metadata
identity probe** at `GET /v1/instruments/binance/BTCUSDT`. It requires a locally stored
`TARDIS_API_KEY`; no probe is implemented or run by this preregistration. Its response contract is
the provider's `id`, `datasetId`, `exchange`, normalized base/quote currencies, `type`, and
`availableSince` fields. It may establish metadata identity only; it cannot establish paid
historical-download entitlement, licence suitability, data continuity, or candle validity.

Tardis CSV files are partitioned by provider local-arrival timestamp, while any future candles
would use Binance exchange-event UTC timestamps. V10 deliberately preregisters no finite boundary
partition set, event filter, or deduplication rule. It therefore cannot download trade files,
aggregate candles, persist an input, freeze a manifest, run research/backtests, select a policy, or
authorize strict OOS. A later independently preregistered protocol must bind those boundary and
continuity rules before ingest can be considered.

The 2025 holdout remains sealed and the default remains `NEUTRAL`. There is no broker, order,
live-trading, RiskEngine, DryRunBroker, ML or recommendation path in V10.
