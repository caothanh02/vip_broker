from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from trading_bot.data.csv_store import write_candles_atomic
from trading_bot.domain.models import Candle
from trading_bot.ml.dataset import (
    FINAL_HOLDOUT,
    DatasetBuildError,
    LabelPolicy,
    _atomic_publish,
    _label,
    _output_columns,
    _Segment,
    _segment_frame,
    _segments,
    build_ml_dataset,
)

BASE = datetime(2022, 1, 1, tzinfo=UTC)


def candle(hour: int, close: Decimal | None = None) -> Candle:
    value = close if close is not None else Decimal(100 + hour)
    return Candle(
        BASE + timedelta(hours=hour),
        BASE + timedelta(hours=hour + 1),
        "BTC/USDT",
        "1h",
        value,
        value + 1,
        value - 1,
        value,
        Decimal("1000"),
        True,
    )


def label_candles(count: int = 50) -> list[Candle]:
    return [candle(index, Decimal("100")) for index in range(count)]


def test_segment_features_are_causal() -> None:
    original = _segment_frame(_Segment("train", "train-0", [candle(i) for i in range(240)]))
    changed = [candle(i) for i in range(240)]
    last = changed[-1]
    changed[-1] = Candle(
        last.open_time,
        last.close_time,
        last.symbol,
        last.timeframe,
        Decimal("999"),
        Decimal("1000"),
        Decimal("998"),
        Decimal("999"),
        Decimal("9999"),
        True,
    )
    revised = _segment_frame(_Segment("train", "train-0", changed))
    assert original.iloc[-2].equals(revised.iloc[-2])


def test_split_and_interruption_boundaries_reset_indicator_warmup() -> None:
    data = [candle(index) for index in range(450)]
    splits = {"train": (BASE, BASE + timedelta(hours=450))}
    parts = _segments(data, splits, {BASE + timedelta(hours=225)}, {BASE + timedelta(hours=224)})
    assert [(part.candles[0].open_time, len(part.candles)) for part in parts] == [
        (BASE, 224),
        (BASE + timedelta(hours=226), 224),
    ]
    resumed = _segment_frame(parts[1])
    assert pd.isna(resumed.iloc[198].ema200)
    assert not pd.isna(resumed.iloc[199].ema200)


def test_label_uses_next_open_atr_barriers_and_costs() -> None:
    data = label_candles()
    entry = data[1]
    data[1] = Candle(
        entry.open_time,
        entry.close_time,
        entry.symbol,
        entry.timeframe,
        Decimal("110"),
        Decimal("111"),
        Decimal("109"),
        Decimal("110"),
        entry.volume,
        True,
    )
    touched = data[2]
    data[2] = Candle(
        touched.open_time,
        touched.close_time,
        touched.symbol,
        touched.timeframe,
        Decimal("110"),
        Decimal("119"),
        Decimal("109"),
        Decimal("118"),
        touched.volume,
        True,
    )
    outcome, target, end, net = _label(data, 0, Decimal("2"), LabelPolicy())
    assert outcome == "profit" and target == 1 and end == data[2].open_time
    expected_entry = Decimal("110") * Decimal("1.0005") * Decimal("1.001")
    expected_exit = Decimal("118") * Decimal("0.9995") * Decimal("0.999")
    assert net == expected_exit / expected_entry - 1


def test_same_candle_barrier_tie_is_stop_first_and_gap_stop_uses_open() -> None:
    data = label_candles()
    both = data[1]
    data[1] = Candle(
        both.open_time,
        both.close_time,
        both.symbol,
        both.timeframe,
        Decimal("100"),
        Decimal("109"),
        Decimal("95"),
        Decimal("100"),
        both.volume,
        True,
    )
    outcome, target, _, _ = _label(data, 0, Decimal("2"), LabelPolicy())
    assert outcome == "stop" and target == 0
    data[1] = candle(1, Decimal("100"))
    gap = data[2]
    data[2] = Candle(
        gap.open_time,
        gap.close_time,
        gap.symbol,
        gap.timeframe,
        Decimal("90"),
        Decimal("101"),
        Decimal("89"),
        Decimal("100"),
        gap.volume,
        True,
    )
    _, _, _, net = _label(data, 0, Decimal("2"), LabelPolicy())
    expected_entry = Decimal("100") * Decimal("1.0005") * Decimal("1.001")
    expected_exit = Decimal("90") * Decimal("0.9995") * Decimal("0.999")
    assert net == expected_exit / expected_entry - 1


def test_timeout_and_incomplete_horizon_are_excluded_from_labels() -> None:
    outcome, target, _, net = _label(label_candles(), 0, Decimal("2"), LabelPolicy())
    assert outcome == "timeout" and target is None and net is None
    outcome, _, _, _ = _label(label_candles(10), 0, Decimal("2"), LabelPolicy())
    assert outcome == "horizon_incomplete"


def test_final_holdout_schema_has_no_future_derived_columns() -> None:
    columns = _output_columns(FINAL_HOLDOUT)
    assert "target" not in columns
    assert "outcome" not in columns
    assert "label_end_time" not in columns
    assert "net_return_after_costs" not in columns


def test_unverified_source_fails_before_creating_output(tmp_path: Path) -> None:
    destination = tmp_path / "output"
    with pytest.raises(DatasetBuildError, match="verified CSV"):
        build_ml_dataset(tmp_path / "missing.csv", destination)
    assert not destination.exists()


def test_tampered_metadata_is_rejected_before_output_publication(tmp_path: Path) -> None:
    source = tmp_path / "btc.csv"
    write_candles_atomic(source, [candle(0)])
    source.with_name("btc.csv.metadata.json").write_text('{"csv_sha256":"0"}', encoding="utf-8")
    source.with_name("btc.anomalies.json").write_text("{}", encoding="utf-8")
    destination = tmp_path / "output"
    with pytest.raises(DatasetBuildError, match="source verification failed"):
        build_ml_dataset(source, destination)
    assert not destination.exists()


def test_atomic_publish_failure_preserves_previous_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "generation"
    destination.mkdir()
    (destination / "dataset.manifest.json").write_text("old", encoding="utf-8")
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "dataset.manifest.json").write_text("new", encoding="utf-8")
    original_replace = __import__("os").replace

    def fail_second_replace(source: Path | str, target: Path | str) -> None:
        if Path(source) == staged:
            raise OSError("simulated publish failure")
        original_replace(source, target)

    monkeypatch.setattr("trading_bot.ml.dataset.os.replace", fail_second_replace)
    with pytest.raises(OSError, match="simulated"):
        _atomic_publish(staged, destination)
    assert (destination / "dataset.manifest.json").read_text(encoding="utf-8") == "old"
