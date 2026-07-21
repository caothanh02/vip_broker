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
    CandidatePolicy,
    DatasetBuildError,
    LabelPolicy,
    _atomic_publish,
    _effective_splits,
    _generation_id,
    _label,
    _output_columns,
    _recover_generation,
    _Segment,
    _segment_frame,
    _segments,
    _validate_fixed_coverage,
    build_ml_dataset,
)
from trading_bot.strategy.ema_volume_atr import (
    is_long_entry_candidate,
    long_entry_candidate_mask,
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
    assert outcome == "profit" and target == 1 and end == data[2].close_time
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


def test_label_availability_orders_open_gaps_before_intrabar_barriers() -> None:
    data = label_candles()
    gap_stop = data[2]
    data[2] = Candle(
        gap_stop.open_time,
        gap_stop.close_time,
        gap_stop.symbol,
        gap_stop.timeframe,
        Decimal("90"),
        Decimal("101"),
        Decimal("89"),
        Decimal("100"),
        gap_stop.volume,
        True,
    )
    outcome, target, end, _ = _label(data, 0, Decimal("2"), LabelPolicy())
    assert (outcome, target, end) == ("stop", 0, data[2].open_time)

    target_gap = data[2]
    data[2] = Candle(
        target_gap.open_time,
        target_gap.close_time,
        target_gap.symbol,
        target_gap.timeframe,
        Decimal("109"),
        Decimal("110"),
        Decimal("90"),
        Decimal("100"),
        target_gap.volume,
        True,
    )
    outcome, target, end, net = _label(data, 0, Decimal("2"), LabelPolicy())
    assert (outcome, target, end) == ("profit", 1, data[2].open_time)
    expected_entry = Decimal("100") * Decimal("1.0005") * Decimal("1.001")
    expected_target_exit = Decimal("108") * Decimal("0.9995") * Decimal("0.999")
    assert net == expected_target_exit / expected_entry - 1


def test_intrabar_label_availability_is_candle_close() -> None:
    data = label_candles()
    touched = data[2]
    data[2] = Candle(
        touched.open_time,
        touched.close_time,
        touched.symbol,
        touched.timeframe,
        Decimal("100"),
        Decimal("109"),
        Decimal("99"),
        Decimal("108"),
        touched.volume,
        True,
    )
    outcome, target, end, _ = _label(data, 0, Decimal("2"), LabelPolicy())
    assert (outcome, target, end) == ("profit", 1, data[2].close_time)


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


def test_fixed_range_metadata_and_actual_coverage_fail_closed() -> None:
    with pytest.raises(DatasetBuildError, match="starts after"):
        _effective_splits(
            {
                "requested_start": "2024-01-01T00:00:00Z",
                "effective_end": "2026-05-02T00:00:00Z",
            }
        )
    with pytest.raises(DatasetBuildError, match="hour-aligned"):
        _effective_splits(
            {
                "requested_start": "2022-01-01T00:00:00Z",
                "effective_end": "2026-05-02T00:30:00Z",
            }
        )
    with pytest.raises(DatasetBuildError, match="must be UTC"):
        _effective_splits(
            {
                "requested_start": "2022-01-01T00:00:00+07:00",
                "effective_end": "2026-05-02T00:00:00Z",
            }
        )
    splits = {"train": (BASE, BASE + timedelta(hours=3))}
    with pytest.raises(DatasetBuildError, match="incomplete"):
        _validate_fixed_coverage([candle(0), candle(2)], splits, set(), set())
    _validate_fixed_coverage([candle(0), candle(2)], splits, {BASE + timedelta(hours=1)}, set())


def test_public_builder_rejects_verified_but_partial_fixed_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "btc.csv"
    source.write_text("source", encoding="utf-8")
    report = source.with_name("btc.anomalies.json")
    report.write_text("report", encoding="utf-8")
    metadata = {
        "generation_id": "verified-source",
        "requested_start": "2022-01-01T00:00:00Z",
        "effective_end": "2026-05-02T00:00:00Z",
    }
    monkeypatch.setattr(
        "trading_bot.ml.dataset._read_verified_source",
        lambda _: ([candle(index) for index in range(240)], metadata, set(), report),
    )
    monkeypatch.setattr("trading_bot.ml.dataset._non_tradable_open_times", lambda _: set())
    with pytest.raises(DatasetBuildError, match="fixed split coverage is incomplete"):
        build_ml_dataset(source, tmp_path / "output")


def test_candidate_policy_is_reproducible_and_matches_execution_predicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = CandidatePolicy()
    frame = pd.DataFrame(
        [
            {
                "is_closed": True,
                "ema20": 1.0,
                "ema50": 1.0,
                "ema200": 0.5,
                "close": 1.0,
                "volume": 10.0,
                "volume_sma20": 5.0,
                "atr14": 1.0,
            },
            {
                "is_closed": True,
                "ema20": 2.0,
                "ema50": 1.0,
                "ema200": 0.5,
                "close": 1.0,
                "volume": 10.0,
                "volume_sma20": 5.0,
                "atr14": 1.0,
            },
        ]
    )
    assert bool(long_entry_candidate_mask(frame, policy).iloc[-1])
    assert not bool(
        long_entry_candidate_mask(frame, CandidatePolicy(volume_multiplier=2.1)).iloc[-1]
    )
    monkeypatch.setenv("VOLUME_MULTIPLIER", "999")
    assert bool(long_entry_candidate_mask(frame, CandidatePolicy()).iloc[-1])
    assert is_long_entry_candidate(frame.iloc[-1], frame.iloc[-2], policy)
    checksums = {"csv": "a", "metadata": "b", "report": "c"}
    splits = {"train": (BASE, BASE + timedelta(hours=1))}
    identity = _generation_id(checksums, "source", splits, policy, LabelPolicy())
    changed = _generation_id(
        checksums,
        "source",
        splits,
        CandidatePolicy(volume_multiplier=1.5),
        LabelPolicy(),
    )
    assert identity != changed


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
    monkeypatch.setattr("trading_bot.ml.dataset._recover_generation", lambda _: None)
    monkeypatch.setattr("trading_bot.ml.dataset._validate_generation", lambda _: None)
    with pytest.raises(OSError, match="simulated"):
        _atomic_publish(staged, destination)
    assert (destination / "dataset.manifest.json").read_text(encoding="utf-8") == "old"


def test_recovery_prefers_valid_destination_and_restores_valid_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "generation"
    backup = tmp_path / ".generation.previous"
    destination.mkdir()
    backup.mkdir()
    valid: set[Path] = {destination, backup}
    monkeypatch.setattr("trading_bot.ml.dataset._generation_is_valid", lambda path: path in valid)
    monkeypatch.setattr("trading_bot.ml.dataset._validate_generation", lambda path: None)
    _recover_generation(destination)
    assert destination.exists() and not backup.exists()

    backup.mkdir()
    valid = {backup}
    _recover_generation(destination)
    assert destination.exists() and not backup.exists()

    destination.mkdir(exist_ok=True)
    backup.mkdir()
    valid = {destination}
    _recover_generation(destination)
    assert destination.exists() and not backup.exists()


def test_repeated_publisher_builds_are_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "btc.csv"
    source.write_text("source", encoding="utf-8")
    source.with_name("btc.csv.metadata.json").write_text("metadata", encoding="utf-8")
    source.with_name("btc.anomalies.json").write_text("report", encoding="utf-8")
    metadata: dict[str, object] = {
        "generation_id": "verified-source",
        "requested_start": "2022-01-01T00:00:00Z",
        "effective_end": "2026-05-02T00:00:00Z",
    }
    end = datetime(2026, 5, 2, tzinfo=UTC)
    count = int((end - BASE) / timedelta(hours=1))
    monkeypatch.setattr(
        "trading_bot.ml.dataset._read_verified_source",
        lambda _: (
            [candle(index) for index in range(count)],
            metadata,
            set(),
            source.with_name("btc.anomalies.json"),
        ),
    )
    monkeypatch.setattr("trading_bot.ml.dataset._non_tradable_open_times", lambda _: set())
    first, second = tmp_path / "first", tmp_path / "second"
    build_ml_dataset(source, first)
    build_ml_dataset(source, second)
    assert sorted(path.name for path in first.iterdir()) == sorted(
        path.name for path in second.iterdir()
    )
    for path in first.iterdir():
        assert path.read_bytes() == (second / path.name).read_bytes()
