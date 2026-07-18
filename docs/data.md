# Historical data

The historical pipeline downloads Binance **Spot** public `BTCUSDT` 1-hour klines. It never sends an authenticated request, does not need an API key, and does not enable live trading. `BTCUSDT` is the exchange symbol; the domain model uses `BTC/USDT`.

All ranges are UTC and half-open: `[start, end)`. Date-only inputs such as `2024-01-01` mean UTC midnight (`2024-01-01T00:00:00Z`). Datetime inputs with a time must include a timezone offset; naive datetimes are rejected. Binance reports the raw close timestamp at the final millisecond of the interval. The downloader canonicalizes it to `open_time + 1 hour`, so every stored candle represents `[open_time, close_time)` and remains compatible with validation.

Only closed candles are stored. The CSV uses Decimal strings for OHLCV, is written atomically with a metadata sidecar, and is ignored by Git.

```powershell
trading-bot download-data --start 2024-01-01 --end 2024-02-01 --output data/raw/btcusdt_1h.csv
trading-bot validate-data --input data/raw/btcusdt_1h.csv --max-age-hours 48
trading-bot backtest --input data/raw/btcusdt_1h.csv --output reports/backtests/btcusdt_1h_baseline.json
```

Running `download-data` again with the same output continues at the candle after the existing final candle. Use `--overwrite` only to replace the requested range after a successful validated download. The downloader retries bounded HTTP 429 and 5xx failures; malformed data, pagination stalls, duplicates, gaps, and CSV conflicts fail clearly without replacing an existing file.

Backtests retain the project OHLC assumptions: signals are known only after a candle closes, entries and signal exits fill at a later candle open, and unknown intrabar ordering is treated conservatively. Binance history is one-exchange data and may be revised by the exchange.
