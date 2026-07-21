# Leakage-safe ML dataset

`trading-bot build-dataset` prepares data for a future entry filter only. It does not train a
model, alter the EMA/volume/ATR strategy, run a backtest, or enable live trading.

The builder accepts only a verified BTC/USDT 1h Vision generation: CSV, metadata, and anomaly
report checksums and generation identity must validate before anything is published. Rows are
created only for deterministic long candidates, never for every candle.

The immutable half-open ranges are train `[2022-01-01, 2025-01-01)`, validation `[2025-01-01,
2026-01-01)`, test `[2026-01-01, 2026-05-01)`, and final holdout from `2026-05-01` to the source
end. Indicators warm up independently for each split and continuous tradable segment. There is no
random split, shuffle, interpolation, or history across a verified interruption.

Development labels use next-open entry with modeled entry/exit slippage and fees, a 2x causal ATR
stop, a 4x ATR profit barrier, and a 48-candle maximum hold. Stop wins an ambiguous OHLC touch.
Timeouts are reported but excluded from binary rows. The final holdout is sealed: it contains only
candidate features and audit timestamps, never targets, outcomes, label-end times, or returns.

Outputs are staged, checksum-verified, and atomically published with `dataset.manifest.json` as
the commit marker. Generated data is ignored by Git and must not be committed.

```text
make build-dataset DATA_FILE=data/raw/btcusdt_1h.csv ML_DATASET_DIR=data/datasets/btcusdt_1h_v1
```
