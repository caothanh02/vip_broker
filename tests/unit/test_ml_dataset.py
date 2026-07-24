from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from trading_bot.data.csv_store import write_candles_atomic
from trading_bot.domain.models import Candle
from trading_bot.features.pipeline import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION
from trading_bot.ml.dataset import (
    CANDIDATE_POLICY_VERSION,
    DATASET_SCHEMA_VERSION,
    DEFAULT_CANDIDATE_POLICY,
    DEFAULT_LABEL_POLICY,
    FINAL_HOLDOUT,
    HOLDOUT_POLICY_VERSION,
    LABEL_POLICY_VERSION,
    SEGMENTATION_POLICY_VERSION,
    SPLITS,
    CandidatePolicy,
    DatasetBuildError,
    LabelPolicy,
    VerifiedSourceSnapshot,
    _atomic_publish,
    _effective_splits,
    _file_sha256,
    _generation_id,
    _generation_is_valid,
    _label,
    _output_columns,
    _output_times,
    _recover_generation,
    _reverify_source_snapshot,
    _Segment,
    _segment_frame,
    _segments,
    _validate_fixed_coverage,
    _validate_generation,
    _write_csv,
    build_ml_dataset,
    validate_development_dataset_generation,
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


def _canonical_splits() -> dict[str, tuple[datetime, datetime]]:
    return {**SPLITS, FINAL_HOLDOUT: (SPLITS["test"][1], datetime(2026, 5, 3, tzinfo=UTC))}


def _row_for_split(split: str, start: datetime) -> dict[str, Any]:
    row: dict[str, Any] = {
        "split": split,
        "segment_id": f"{split}-0",
        "signal_time": (start + timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
        "entry_time": (start + timedelta(hours=2)).isoformat().replace("+00:00", "Z"),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        **{column: 1.0 for column in FEATURE_COLUMNS},
    }
    if split != FINAL_HOLDOUT:
        row.update(
            {
                "label_end_time": (start + timedelta(hours=2)).isoformat().replace("+00:00", "Z"),
                "target": 1,
                "outcome": "profit",
                "net_return_after_costs": "0.01",
            }
        )
    return row


def _recompute_generation_id(manifest: dict[str, Any]) -> None:
    raw_splits = manifest["splits"]
    splits = {
        name: (
            datetime.fromisoformat(record["start"].replace("Z", "+00:00")),
            datetime.fromisoformat(record["end"].replace("Z", "+00:00")),
        )
        for name, record in raw_splits.items()
    }
    label_values = dict(manifest["label_policy"])
    for key in (
        "stop_atr_multiple",
        "profit_atr_multiple",
        "entry_fee_rate",
        "exit_fee_rate",
        "entry_slippage_rate",
        "exit_slippage_rate",
    ):
        label_values[key] = Decimal(label_values[key])
    manifest["dataset_generation_id"] = _generation_id(
        {
            key: manifest[key]
            for key in ("source_csv_sha256", "source_metadata_sha256", "anomaly_report_sha256")
        },
        manifest["source_generation_id"],
        splits,
        CandidatePolicy(**manifest["candidate_policy"]),
        LabelPolicy(**label_values),
    )


def _write_manifest(directory: Path, manifest: dict[str, Any]) -> None:
    (directory / "dataset.manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def _read_manifest(directory: Path) -> dict[str, Any]:
    return json.loads((directory / "dataset.manifest.json").read_text(encoding="utf-8"))


def _refresh_output_checksum(directory: Path, split: str) -> None:
    manifest = _read_manifest(directory)
    manifest["output_file_sha256"][f"{split}.csv"] = _file_sha256(directory / f"{split}.csv")
    _write_manifest(directory, manifest)


def _minimal_valid_generation(directory: Path) -> Path:
    directory.mkdir()
    splits = _canonical_splits()
    rows = {split: [_row_for_split(split, bounds[0])] for split, bounds in splits.items()}
    checksums: dict[str, str] = {}
    for split, split_rows in rows.items():
        path = directory / f"{split}.csv"
        _write_csv(path, split_rows, split)
        checksums[path.name] = _file_sha256(path)
    source_checksums = {
        "source_csv_sha256": "a" * 64,
        "source_metadata_sha256": "b" * 64,
        "anomaly_report_sha256": "c" * 64,
    }
    segments = {
        f"{split}-0": {
            "split": split,
            "start": start.isoformat().replace("+00:00", "Z"),
            "end": end.isoformat().replace("+00:00", "Z"),
        }
        for split, (start, end) in splits.items()
    }
    manifest: dict[str, Any] = {
        "dataset_generation_id": "",
        **source_checksums,
        "anomaly_report": "btc.anomalies.json",
        "source_generation_id": "verified-source",
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "splits": {
            name: {
                "start": start.isoformat().replace("+00:00", "Z"),
                "end": end.isoformat().replace("+00:00", "Z"),
            }
            for name, (start, end) in splits.items()
        },
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "candidate_policy_version": CANDIDATE_POLICY_VERSION,
        "label_policy_version": LABEL_POLICY_VERSION,
        "segmentation_policy_version": SEGMENTATION_POLICY_VERSION,
        "holdout_policy_version": HOLDOUT_POLICY_VERSION,
        "candidate_policy": asdict(DEFAULT_CANDIDATE_POLICY),
        "label_policy": {
            key: str(value) if isinstance(value, Decimal) else value
            for key, value in asdict(DEFAULT_LABEL_POLICY).items()
        },
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_columns": FEATURE_COLUMNS,
        "code_commit": None,
        "row_counts": {split: len(value) for split, value in rows.items()},
        "candidate_counts": {split: 1 for split in rows},
        "development_label_counts": {
            split: {"positive": 1, "negative": 0, "timeout": 0} for split in SPLITS
        },
        "development_trainable": {split: False for split in SPLITS},
        "excluded_row_counts": {},
        "split_signal_coverage": {split: _output_times(value) for split, value in rows.items()},
        "output_file_sha256": checksums,
        "interruption_summary": {"policy_version": SEGMENTATION_POLICY_VERSION, "segment_count": 4},
        "segments": {key: segments[key] for key in sorted(segments)},
    }
    _recompute_generation_id(manifest)
    _write_manifest(directory, manifest)
    _validate_generation(directory)
    return directory


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
        "trading_bot.ml.dataset._capture_verified_source",
        lambda _: VerifiedSourceSnapshot(
            source,
            source.with_name("btc.csv.metadata.json"),
            report,
            tuple(candle(index) for index in range(240)),
            metadata,
            frozenset(),
            "verified-source",
            "csv",
            "metadata",
            "report",
            BASE,
            datetime(2026, 5, 2, tzinfo=UTC),
        ),
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


def test_source_snapshot_reverification_rejects_changed_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.csv"
    metadata_path = source.with_name("source.csv.metadata.json")
    report = tmp_path / "report.json"
    source.write_text("csv-a", encoding="utf-8")
    metadata_path.write_text(
        '{"generation_id":"generation-a","anomaly_report":"report.json"}', encoding="utf-8"
    )
    report.write_text("{}", encoding="utf-8")
    snapshot = VerifiedSourceSnapshot(
        source,
        metadata_path,
        report,
        tuple(),
        {},
        frozenset(),
        "generation-a",
        hashlib.sha256(b"csv-a").hexdigest(),
        hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        hashlib.sha256(b"{}").hexdigest(),
        BASE,
        BASE + timedelta(hours=1),
    )
    monkeypatch.setattr("trading_bot.ml.dataset.verify_metadata_checksum", lambda _: True)
    _reverify_source_snapshot(snapshot)
    source.write_text("csv-b", encoding="utf-8")
    with pytest.raises(DatasetBuildError, match="changed"):
        _reverify_source_snapshot(snapshot)


@pytest.mark.parametrize(
    ("candidate_policy", "label_policy"),
    [
        (CandidatePolicy(ema_fast=21), LabelPolicy()),
        (CandidatePolicy(atr_window=15), LabelPolicy()),
        (CandidatePolicy(volume_multiplier=1.3), LabelPolicy()),
        (CandidatePolicy(), LabelPolicy(entry_timing="same_open")),
        (CandidatePolicy(), LabelPolicy(same_candle_resolution="target_first")),
        (CandidatePolicy(), LabelPolicy(max_holding_candles=49)),
    ],
)
def test_public_builder_rejects_unsupported_policies(
    tmp_path: Path, candidate_policy: CandidatePolicy, label_policy: LabelPolicy
) -> None:
    with pytest.raises(DatasetBuildError, match="unsupported"):
        build_ml_dataset(
            tmp_path / "missing.csv",
            tmp_path / "output",
            candidate_policy=candidate_policy,
            label_policy=label_policy,
        )


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
        "trading_bot.ml.dataset._capture_verified_source",
        lambda _: VerifiedSourceSnapshot(
            source,
            source.with_name("btc.csv.metadata.json"),
            source.with_name("btc.anomalies.json"),
            tuple(candle(index) for index in range(count)),
            metadata,
            frozenset(),
            "verified-source",
            "csv",
            "metadata",
            "report",
            BASE,
            end,
        ),
    )
    monkeypatch.setattr("trading_bot.ml.dataset._non_tradable_open_times", lambda _: set())
    monkeypatch.setattr("trading_bot.ml.dataset._reverify_source_snapshot", lambda _: None)
    first, second = tmp_path / "first", tmp_path / "second"
    build_ml_dataset(source, first)
    build_ml_dataset(source, second)
    assert sorted(path.name for path in first.iterdir()) == sorted(
        path.name for path in second.iterdir()
    )
    for path in first.iterdir():
        assert path.read_bytes() == (second / path.name).read_bytes()


@pytest.mark.parametrize(
    ("split", "field", "value"),
    [
        ("train", "start", "2022-01-01T01:00:00Z"),
        ("train", "end", "2025-01-01T01:00:00Z"),
        ("validation", "start", "2025-01-01T01:00:00Z"),
        ("test", "end", "2026-05-01T01:00:00Z"),
        (FINAL_HOLDOUT, "start", "2026-05-01T01:00:00Z"),
    ],
)
def test_generation_rejects_recomputed_noncanonical_split(
    tmp_path: Path, split: str, field: str, value: str
) -> None:
    directory = _minimal_valid_generation(tmp_path / "generation")
    manifest = _read_manifest(directory)
    manifest["splits"][split][field] = value
    _recompute_generation_id(manifest)
    _write_manifest(directory, manifest)
    with pytest.raises(DatasetBuildError, match="split boundaries|final holdout"):
        _validate_generation(directory)


def test_generation_rejects_unknown_split_even_with_recomputed_identity(tmp_path: Path) -> None:
    directory = _minimal_valid_generation(tmp_path / "generation")
    manifest = _read_manifest(directory)
    manifest["splits"]["research"] = {
        "start": "2026-05-01T00:00:00Z",
        "end": "2026-05-02T00:00:00Z",
    }
    _recompute_generation_id(manifest)
    _write_manifest(directory, manifest)
    with pytest.raises(DatasetBuildError, match="splits are invalid"):
        _validate_generation(directory)


@pytest.mark.parametrize(
    ("policy_name", "key", "value"),
    [
        ("candidate_policy", "ema_fast", 21),
        ("candidate_policy", "atr_window", 15),
        ("candidate_policy", "volume_multiplier", 1.3),
        ("candidate_policy", "strategy_name", "OtherStrategy"),
        ("candidate_policy", "strategy_version", "2.0.0"),
        ("candidate_policy", "entry_rule_version", "2.0.0"),
        ("label_policy", "max_holding_candles", 49),
        ("label_policy", "stop_atr_multiple", "3"),
        ("label_policy", "profit_atr_multiple", "5"),
        ("label_policy", "entry_timing", "same_candle_close"),
        ("label_policy", "same_candle_resolution", "target_first"),
        ("label_policy", "entry_fee_rate", "0.002"),
        ("label_policy", "exit_fee_rate", "0.002"),
        ("label_policy", "entry_slippage_rate", "0.002"),
        ("label_policy", "exit_slippage_rate", "0.002"),
    ],
)
def test_generation_rejects_recomputed_unsupported_policy(
    tmp_path: Path, policy_name: str, key: str, value: object
) -> None:
    directory = _minimal_valid_generation(tmp_path / "generation")
    manifest = _read_manifest(directory)
    manifest[policy_name][key] = value
    _recompute_generation_id(manifest)
    _write_manifest(directory, manifest)
    with pytest.raises(DatasetBuildError, match="unsupported"):
        _validate_generation(directory)


@pytest.mark.parametrize(("key", "value"), [("symbol", "ETH/USDT"), ("timeframe", "4h")])
def test_generation_rejects_market_metadata_mismatch(tmp_path: Path, key: str, value: str) -> None:
    directory = _minimal_valid_generation(tmp_path / "generation")
    manifest = _read_manifest(directory)
    manifest[key] = value
    _write_manifest(directory, manifest)
    with pytest.raises(DatasetBuildError, match="market metadata"):
        _validate_generation(directory)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("development_label_counts", FINAL_HOLDOUT, {"positive": 0, "negative": 0, "timeout": 0}),
        ("development_label_counts", "research", {"positive": 0, "negative": 0, "timeout": 0}),
        ("development_trainable", FINAL_HOLDOUT, False),
        ("development_trainable", "research", False),
    ],
)
def test_generation_rejects_nondevelopment_metadata_keys(
    tmp_path: Path, section: str, key: str, value: object
) -> None:
    directory = _minimal_valid_generation(tmp_path / "generation")
    manifest = _read_manifest(directory)
    manifest[section][key] = value
    _write_manifest(directory, manifest)
    with pytest.raises(DatasetBuildError, match="development metadata"):
        _validate_generation(directory)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("development_label_counts", "train", {"positive": True, "negative": 0, "timeout": 0}),
        ("development_label_counts", "train", {"positive": 1, "negative": 0}),
        ("development_trainable", "train", 0),
    ],
)
def test_generation_rejects_invalid_development_metadata_values(
    tmp_path: Path, section: str, key: str, value: object
) -> None:
    directory = _minimal_valid_generation(tmp_path / "generation")
    manifest = _read_manifest(directory)
    manifest[section][key] = value
    _write_manifest(directory, manifest)
    with pytest.raises(DatasetBuildError, match="label counts|trainable"):
        _validate_generation(directory)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda manifest: manifest["segments"]["train-0"].update({"start": "2021-12-31T23:00:00Z"}),
        lambda manifest: manifest["segments"].update(
            {
                "train-1": {
                    "split": "train",
                    "start": "2022-01-01T01:00:00Z",
                    "end": "2022-01-02T00:00:00Z",
                }
            }
        ),
        lambda manifest: manifest["segments"]["train-0"].update({"split": "research"}),
        lambda manifest: manifest["segments"]["train-0"].update({"start": "2022-01-01T00:30:00Z"}),
    ],
)
def test_generation_rejects_invalid_segment_structure(tmp_path: Path, mutation: object) -> None:
    directory = _minimal_valid_generation(tmp_path / "generation")
    manifest = _read_manifest(directory)
    assert callable(mutation)
    mutation(manifest)
    _write_manifest(directory, manifest)
    with pytest.raises(DatasetBuildError, match="segment"):
        _validate_generation(directory)


def _rewrite_split_rows(directory: Path, split: str, mutate: object) -> None:
    path = directory / f"{split}.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert callable(mutate)
    mutate(rows[0])
    _write_csv(path, rows, split)
    _refresh_output_checksum(directory, split)


def test_generation_requires_signal_and_entry_in_declared_segment(tmp_path: Path) -> None:
    directory = _minimal_valid_generation(tmp_path / "generation")
    _rewrite_split_rows(
        directory, "train", lambda row: row.update({"signal_time": "2021-12-31T23:00:00Z"})
    )
    with pytest.raises(DatasetBuildError, match="signal/entry leaves its split"):
        _validate_generation(directory)


def test_generation_rejects_signal_from_previous_segment(tmp_path: Path) -> None:
    directory = _minimal_valid_generation(tmp_path / "generation")
    manifest = _read_manifest(directory)
    manifest["segments"] = {
        **manifest["segments"],
        "train-1": {
            "split": "train",
            "start": "2022-01-01T02:00:00Z",
            "end": "2025-01-01T00:00:00Z",
        },
    }
    manifest["segments"]["train-0"]["end"] = "2022-01-01T02:00:00Z"
    _write_manifest(directory, manifest)
    _rewrite_split_rows(
        directory,
        "train",
        lambda row: row.update(
            {
                "segment_id": "train-1",
                "signal_time": "2022-01-01T01:00:00Z",
                "entry_time": "2022-01-01T03:00:00Z",
                "label_end_time": "2022-01-01T03:00:00Z",
            }
        ),
    )
    with pytest.raises(DatasetBuildError, match="invalid segment"):
        _validate_generation(directory)


def test_generation_rejects_entry_or_label_outside_segment(tmp_path: Path) -> None:
    directory = _minimal_valid_generation(tmp_path / "generation")
    manifest = _read_manifest(directory)
    manifest["segments"]["train-0"]["end"] = "2022-01-01T03:00:00Z"
    _write_manifest(directory, manifest)
    _rewrite_split_rows(
        directory,
        "train",
        lambda row: row.update(
            {"entry_time": "2022-01-01T03:00:00Z", "label_end_time": "2022-01-01T03:00:00Z"}
        ),
    )
    with pytest.raises(DatasetBuildError, match="invalid segment"):
        _validate_generation(directory)

    directory = _minimal_valid_generation(tmp_path / "label-generation")
    manifest = _read_manifest(directory)
    manifest["segments"]["train-0"]["end"] = "2022-01-01T03:00:00Z"
    _write_manifest(directory, manifest)
    _rewrite_split_rows(
        directory, "train", lambda row: row.update({"label_end_time": "2022-01-01T04:00:00Z"})
    )
    with pytest.raises(DatasetBuildError, match="label leaves"):
        _validate_generation(directory)


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_generation_rejects_label_at_development_split_end(tmp_path: Path, split: str) -> None:
    directory = _minimal_valid_generation(tmp_path / "generation")
    manifest = _read_manifest(directory)
    split_end = manifest["splits"][split]["end"]
    _rewrite_split_rows(directory, split, lambda row: row.update({"label_end_time": split_end}))
    with pytest.raises(DatasetBuildError, match="label leaves"):
        _validate_generation(directory)
    assert not _generation_is_valid(directory)


def test_generation_accepts_label_one_candle_before_development_split_end(tmp_path: Path) -> None:
    directory = _minimal_valid_generation(tmp_path / "generation")
    manifest = _read_manifest(directory)
    split_end = datetime.fromisoformat(manifest["splits"]["train"]["end"].replace("Z", "+00:00"))
    label_end = (split_end - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    _rewrite_split_rows(directory, "train", lambda row: row.update({"label_end_time": label_end}))
    _validate_generation(directory)


def test_minimal_generation_accepts_normalized_signal_and_next_open(tmp_path: Path) -> None:
    _validate_generation(_minimal_valid_generation(tmp_path / "generation"))


def test_generation_recovery_normalizes_corrupt_csv_and_restores_backup(tmp_path: Path) -> None:
    destination = _minimal_valid_generation(tmp_path / "generation")
    backup = tmp_path / ".generation.previous"
    shutil.copytree(destination, backup)
    final_path = destination / f"{FINAL_HOLDOUT}.csv"
    lines = final_path.read_text(encoding="utf-8").splitlines()
    final_path.write_text(
        lines[0] + "\n" + ",".join(lines[1].split(",")[:-1]) + "\n", encoding="utf-8"
    )
    _refresh_output_checksum(destination, FINAL_HOLDOUT)
    assert not _generation_is_valid(destination)
    _recover_generation(destination)
    _validate_generation(destination)
    assert not backup.exists()


@pytest.mark.parametrize("corruption", ["malformed", "missing", "manifest", "checksum"])
def test_generation_is_invalid_for_expected_filesystem_corruption(
    tmp_path: Path, corruption: str
) -> None:
    directory = _minimal_valid_generation(tmp_path / "generation")
    if corruption == "malformed":
        path = directory / "train.csv"
        path.write_text(path.read_text(encoding="utf-8") + '"unterminated\n', encoding="utf-8")
        _refresh_output_checksum(directory, "train")
    elif corruption == "missing":
        (directory / "train.csv").unlink()
    elif corruption == "manifest":
        (directory / "dataset.manifest.json").write_text("{", encoding="utf-8")
    else:
        (directory / "train.csv").write_text("tampered", encoding="utf-8")
    assert not _generation_is_valid(directory)


def test_generation_recovery_keeps_valid_destination_and_fails_closed_when_both_invalid(
    tmp_path: Path,
) -> None:
    destination = _minimal_valid_generation(tmp_path / "generation")
    backup = tmp_path / ".generation.previous"
    shutil.copytree(destination, backup)
    (backup / "dataset.manifest.json").write_text("{", encoding="utf-8")
    _recover_generation(destination)
    _validate_generation(destination)
    assert not backup.exists()

    backup = tmp_path / ".generation.previous"
    shutil.copytree(destination, backup)
    (destination / "dataset.manifest.json").write_text("{", encoding="utf-8")
    (backup / "dataset.manifest.json").write_text("{", encoding="utf-8")
    with pytest.raises(DatasetBuildError, match="no valid"):
        _recover_generation(destination)


def test_development_validator_never_accesses_final_holdout_but_full_validator_does(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = _minimal_valid_generation(tmp_path / "generation")
    final_holdout = directory / "final_holdout.csv"
    final_holdout.write_text("sealed sentinel", encoding="utf-8")
    original_open, original_stat = Path.open, Path.stat

    def forbid_open(path: Path, *args: object, **kwargs: object) -> object:
        if path == final_holdout:
            raise AssertionError("development validator opened final holdout")
        return original_open(path, *args, **kwargs)

    def forbid_stat(path: Path, *args: object, **kwargs: object) -> object:
        if path == final_holdout:
            raise AssertionError("development validator stat final holdout")
        return original_stat(path, *args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "open", forbid_open)
        scoped.setattr(Path, "stat", forbid_stat)
        assert validate_development_dataset_generation(directory)["dataset_generation_id"]
    with pytest.raises(DatasetBuildError):
        _validate_generation(directory)
