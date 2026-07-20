# Historical data

The historical pipeline downloads Binance **Spot** public `BTCUSDT` 1-hour klines. It never sends an authenticated request, does not need an API key, and does not enable live trading. `BTCUSDT` is the exchange symbol; the domain model uses `BTC/USDT`.

All ranges are UTC and half-open: `[start, end)`. Date-only inputs such as `2024-01-01` mean UTC midnight (`2024-01-01T00:00:00Z`). Datetime inputs with a time must include a timezone offset; naive datetimes are rejected. Binance reports the raw close timestamp at the final millisecond of the interval. The downloader canonicalizes it to `open_time + 1 hour`, so every stored candle represents `[open_time, close_time)` and remains compatible with validation.

Only closed candles are stored. The CSV uses Decimal strings for OHLCV, is written atomically with a metadata sidecar, and is ignored by Git.

For long history, `--source binance-vision` reads the official Binance Vision Spot archives first, verifies each ZIP against its official `.CHECKSUM`, then uses public Binance REST only for the newest archive suffix. It needs no API key. Archive rows before 2025 use milliseconds and rows from 2025 use microseconds. REST close timestamps remain exact; verified archive rows may close early by at most 60 seconds, never late. Every accepted early-close exception is recorded in `<csv-stem>.anomalies.json` with its archive checksum, raw timestamps, row number, and canonical UTC candle interval.

The Vision writer stages the CSV, anomaly report, and metadata together. Metadata is published last and contains a generation ID plus SHA-256 checksums for both sidecars. `validate-data` rejects a checksum mismatch, a missing referenced anomaly report, or summary records that disagree with metadata.

```powershell
trading-bot download-data --start 2024-01-01 --end 2024-02-01 --output data/raw/btcusdt_1h.csv
trading-bot download-data --source binance-vision --start 2021-12-01 --end now --output data/raw/btcusdt_1h.csv --overwrite
trading-bot validate-data --input data/raw/btcusdt_1h.csv --max-age-hours 48
trading-bot backtest --input data/raw/btcusdt_1h.csv --output reports/backtests/btcusdt_1h_baseline.json
```

Running `download-data` again with the same output continues at the candle after the existing final candle. Use `--overwrite` only to replace the requested range after a successful validated download. The downloader retries bounded HTTP 429 and 5xx failures; malformed data, pagination stalls, duplicates, gaps, and CSV conflicts fail clearly without replacing an existing file.

Backtests retain the project OHLC assumptions: signals are known only after a candle closes, entries and signal exits fill at a later candle open, and unknown intrabar ordering is treated conservatively. Binance history is one-exchange data and may be revised by the exchange.
