# Leakage-safe ML dataset

`trading-bot build-dataset` prepares data for a future entry filter only. It does not train a
model, alter the EMA/volume/ATR strategy, run a backtest, or enable live trading.

The builder accepts only a verified BTC/USDT 1h Vision generation: CSV, metadata, and anomaly
report checksums and generation identity must validate before anything is published. Rows are
created only for deterministic long candidates, never for every candle.

The candidate policy is an immutable, versioned baseline embedded in the builder: BTC/USDT 1h,
EMA 20/50/200, volume window 20 with multiplier 1.2, ATR 14, and the versioned
`EmaVolumeAtrStrategy` entry rule. It is included with the label policy, feature schema, split,
segmentation, and holdout policy versions in the content-derived generation ID and manifest. The
builder does not read `.env` candidate settings.

The immutable half-open ranges are train `[2022-01-01, 2025-01-01)`, validation `[2025-01-01,
2026-01-01)`, test `[2026-01-01, 2026-05-01)`, and final holdout from `2026-05-01` to the source
end. Indicators warm up independently for each split and continuous tradable segment. There is no
random split, shuffle, interpolation, or history across a verified interruption.

Development labels use next-open entry with modeled entry/exit slippage and fees, a 2x causal ATR
stop, a 4x ATR profit barrier, and a 48-candle maximum hold. An opening gap is resolved at the
known open (stop first, then capped-at-target profit); otherwise stop wins an ambiguous OHLC touch
inside a candle. Label availability is the open for gap outcomes and candle close for intrabar
outcomes.
Timeouts are reported but excluded from binary rows. The final holdout is sealed: it contains only
candidate features and audit timestamps, never targets, outcomes, label-end times, or returns.

Outputs are staged, checksum-verified, and atomically published with `dataset.manifest.json` as
the commit marker. Generated data is ignored by Git and must not be committed.

```text
make build-dataset DATA_FILE=data/raw/btcusdt_1h.csv ML_DATASET_DIR=data/datasets/btcusdt_1h_v1
```

## Experimental Logistic Regression baseline

`trading-bot train --dataset-dir <dir> --output-dir <dir>` fits `StandardScaler` and a fixed
`LogisticRegression(random_state=42)` on `train.csv` only. It selects one versioned probability
threshold only from validation cumulative net return (at least five selected validation trades),
then evaluates test once. `final_holdout.csv` is intentionally never opened by this command.

The local-only artifact includes the fitted model, ordered feature schema, input checksums,
dataset/source generation IDs, fixed model configuration, threshold selection, validation/test
metrics, code and dependency provenance. Every artifact is marked experimental, not production
eligible, and live trading remains disabled. It is not connected to strategy, backtest, broker, or
live trading.
