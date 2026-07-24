"""Leakage-safe, candidate-only dataset construction for the BTC/USDT strategy.

This module deliberately creates data only.  It neither fits a model nor invokes a
broker, risk engine, or backtester.  Development labels are generated from a
conservative OHLC policy; the final holdout has no future-derived columns.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, cast

import numpy as np
import pandas as pd

from trading_bot.data.csv_store import (
    CsvDataError,
    csv_sha256,
    metadata_path,
    read_candles,
    verified_missing_open_times,
    verify_metadata_checksum,
)
from trading_bot.data.validation import validate_candles
from trading_bot.domain.models import Candle
from trading_bot.features.pipeline import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION, build_features
from trading_bot.strategy.ema_volume_atr import long_entry_candidate_mask

HOUR: Final = timedelta(hours=1)
DATASET_SCHEMA_VERSION: Final = "2.0.0"
CANDIDATE_POLICY_VERSION: Final = "1.0.0"
LABEL_POLICY_VERSION: Final = "1.0.0"
SEGMENTATION_POLICY_VERSION: Final = "1.0.0"
HOLDOUT_POLICY_VERSION: Final = "1.0.0"
SPLITS: Final = {
    "train": (datetime(2022, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC)),
    "validation": (datetime(2025, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC)),
    "test": (datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 5, 1, tzinfo=UTC)),
}
FINAL_HOLDOUT: Final = "final_holdout"
CANONICAL_SPLIT_NAMES: Final = (*SPLITS, FINAL_HOLDOUT)
AUDIT_COLUMNS: Final = [
    "split",
    "segment_id",
    "signal_time",
    "entry_time",
    "feature_schema_version",
]
LABEL_COLUMNS: Final = ["label_end_time", "target", "outcome", "net_return_after_costs"]


class DatasetBuildError(ValueError):
    """The verified source or requested immutable dataset policy is invalid."""


@dataclass(frozen=True, slots=True)
class LabelPolicy:
    """Fixed baseline barrier policy.  It is intentionally not a CLI surface."""

    stop_atr_multiple: Decimal = Decimal("2")
    profit_atr_multiple: Decimal = Decimal("4")
    max_holding_candles: int = 48
    entry_fee_rate: Decimal = Decimal("0.001")
    exit_fee_rate: Decimal = Decimal("0.001")
    entry_slippage_rate: Decimal = Decimal("0.0005")
    exit_slippage_rate: Decimal = Decimal("0.0005")
    same_candle_resolution: str = "stop_first"
    entry_timing: str = "next_candle_open"


@dataclass(frozen=True, slots=True)
class CandidatePolicy:
    """Immutable baseline entry policy; environment files cannot alter it."""

    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    ema_fast: int = 20
    ema_slow: int = 50
    ema_trend: int = 200
    volume_window: int = 20
    volume_multiplier: float = 1.2
    atr_window: int = 14
    strategy_name: str = "EmaVolumeAtrStrategy"
    strategy_version: str = "1.0.0"
    entry_rule_version: str = CANDIDATE_POLICY_VERSION


DEFAULT_CANDIDATE_POLICY: Final = CandidatePolicy()
DEFAULT_LABEL_POLICY: Final = LabelPolicy()


@dataclass(frozen=True, slots=True)
class DatasetBuildSummary:
    generation_id: str
    source_generation_id: str
    source_candle_count: int
    segment_count: int
    candidate_counts: dict[str, int]
    label_counts: dict[str, dict[str, int]]
    exclusion_counts: dict[str, int]
    output_checksums: dict[str, str]
    trainable_splits: dict[str, bool]
    candidate_policy: CandidatePolicy
    label_policy: LabelPolicy


@dataclass(frozen=True, slots=True)
class VerifiedSourceSnapshot:
    """Frozen provenance captured before any feature or label computation."""

    input_path: Path
    metadata_path: Path
    anomaly_report_path: Path
    candles: tuple[Candle, ...]
    metadata: Mapping[str, Any]
    missing_open_times: frozenset[datetime]
    source_generation_id: str
    csv_sha256: str
    metadata_sha256: str
    anomaly_report_sha256: str
    requested_start: datetime
    effective_end: datetime


@dataclass(frozen=True, slots=True)
class DatasetFileStats:
    row_count: int
    positive_count: int | None
    negative_count: int | None
    first_signal_time: datetime | None
    last_signal_time: datetime | None


@dataclass(frozen=True, slots=True)
class _Segment:
    split: str
    segment_id: str
    candles: list[Candle]


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _file_sha256(path: Path) -> str:
    return csv_sha256(path)


def _generation_id(
    source_checksums: dict[str, str],
    source_generation_id: str,
    splits: dict[str, tuple[datetime, datetime]],
    candidate_policy: CandidatePolicy,
    label_policy: LabelPolicy,
) -> str:
    identity = hashlib.sha256(
        json.dumps(
            {
                "source": source_checksums,
                "source_generation_id": source_generation_id,
                "features": FEATURE_COLUMNS,
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "dataset_schema_version": DATASET_SCHEMA_VERSION,
                "candidate_policy": asdict(candidate_policy),
                "label_policy": asdict(label_policy),
                "candidate_policy_version": CANDIDATE_POLICY_VERSION,
                "label_policy_version": LABEL_POLICY_VERSION,
                "segmentation_policy_version": SEGMENTATION_POLICY_VERSION,
                "holdout_policy_version": HOLDOUT_POLICY_VERSION,
                "splits": {name: (_iso(start), _iso(end)) for name, (start, end) in splits.items()},
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest()
    return identity[:32]


def _code_commit() -> str | None:
    """Best-effort provenance, intentionally omitted outside a Git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True,
            check=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value if len(value) == 40 else None


def _verified_report_path(input_path: Path, metadata: dict[str, Any]) -> Path:
    name = metadata.get("anomaly_report")
    if not isinstance(name, str) or Path(name).name != name:
        raise DatasetBuildError("verified anomaly report reference is invalid")
    report_path = input_path.parent / name
    if not report_path.exists():
        raise DatasetBuildError("verified anomaly report is missing")
    return report_path


def _read_verified_source(
    input_path: Path,
) -> tuple[list[Candle], dict[str, Any], set[datetime], Path]:
    sidecar = metadata_path(input_path)
    if not input_path.exists() or not sidecar.exists():
        raise DatasetBuildError("verified CSV, metadata, and anomaly report are all required")
    try:
        verified = verify_metadata_checksum(input_path)
        metadata_raw = json.loads(sidecar.read_text(encoding="utf-8"))
    except (CsvDataError, OSError, json.JSONDecodeError) as exc:
        raise DatasetBuildError("source verification failed") from exc
    if not verified or not isinstance(metadata_raw, dict):
        raise DatasetBuildError("source metadata verification failed")
    report_path = _verified_report_path(input_path, metadata_raw)
    if metadata_raw.get("internal_symbol") != "BTC/USDT" or metadata_raw.get("timeframe") != "1h":
        raise DatasetBuildError("source must be BTC/USDT 1h")
    allowed_missing = verified_missing_open_times(input_path)
    try:
        candles = read_candles(input_path, allowed_missing_open_times=allowed_missing)
        validate_candles(candles, allowed_missing_open_times=allowed_missing)
    except (CsvDataError, ValueError) as exc:
        raise DatasetBuildError("source candles are not valid closed UTC candles") from exc
    if any(candle.symbol != "BTC/USDT" or candle.timeframe != "1h" for candle in candles):
        raise DatasetBuildError("source contains an unsupported market")
    return candles, metadata_raw, allowed_missing, report_path


def _non_tradable_open_times(report_path: Path) -> set[datetime]:
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetBuildError("invalid verified anomaly report") from exc
    records = report.get("market_interruptions") if isinstance(report, dict) else None
    if not isinstance(records, list):
        raise DatasetBuildError("verified anomaly report has no market interruption records")
    result: set[datetime] = set()
    for record in records:
        if not isinstance(record, dict):
            raise DatasetBuildError("invalid market interruption record")
        if record.get("tradable") is False:
            value = record.get("open_time")
            if not isinstance(value, str):
                raise DatasetBuildError("non-tradable interruption lacks open time")
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
                raise DatasetBuildError("non-tradable interruption time must be UTC")
            result.add(parsed.astimezone(UTC))
    return result


def _capture_verified_source(input_path: Path) -> VerifiedSourceSnapshot:
    # Obtain a stable generation twice.  The second candle read is retained
    # only when all three artefact hashes stayed identical across the window.
    candles, metadata, missing, report_path = _read_verified_source(input_path)
    first_checksums = (
        _file_sha256(input_path),
        _file_sha256(metadata_path(input_path)),
        _file_sha256(report_path),
    )
    candles, metadata, missing, report_path = _read_verified_source(input_path)
    second_checksums = (
        _file_sha256(input_path),
        _file_sha256(metadata_path(input_path)),
        _file_sha256(report_path),
    )
    if first_checksums != second_checksums:
        raise DatasetBuildError("verified source generation changed while capturing snapshot")
    try:
        generation_id = metadata["generation_id"]
    except KeyError as exc:
        raise DatasetBuildError("verified source generation ID is missing") from exc
    if not isinstance(generation_id, str) or not generation_id:
        raise DatasetBuildError("verified source generation ID is invalid")
    requested_start = _metadata_hour(metadata, "requested_start")
    effective_end = _metadata_hour(metadata, "effective_end")
    return VerifiedSourceSnapshot(
        input_path=input_path,
        metadata_path=metadata_path(input_path),
        anomaly_report_path=report_path,
        candles=tuple(candles),
        metadata=MappingProxyType(dict(metadata)),
        missing_open_times=frozenset(missing),
        source_generation_id=generation_id,
        csv_sha256=second_checksums[0],
        metadata_sha256=second_checksums[1],
        anomaly_report_sha256=second_checksums[2],
        requested_start=requested_start,
        effective_end=effective_end,
    )


def _reverify_source_snapshot(snapshot: VerifiedSourceSnapshot) -> None:
    """Fail closed if any raw generation artefact changed during a build."""
    try:
        if not verify_metadata_checksum(snapshot.input_path):
            raise DatasetBuildError("source metadata verification failed during recheck")
        raw = json.loads(snapshot.metadata_path.read_text(encoding="utf-8"))
    except (CsvDataError, OSError, json.JSONDecodeError) as exc:
        raise DatasetBuildError("source verification failed during recheck") from exc
    if not isinstance(raw, dict):
        raise DatasetBuildError("source metadata is invalid during recheck")
    report_path = _verified_report_path(snapshot.input_path, raw)
    if (
        report_path != snapshot.anomaly_report_path
        or raw.get("generation_id") != snapshot.source_generation_id
        or _file_sha256(snapshot.input_path) != snapshot.csv_sha256
        or _file_sha256(snapshot.metadata_path) != snapshot.metadata_sha256
        or _file_sha256(report_path) != snapshot.anomaly_report_sha256
    ):
        raise DatasetBuildError("verified source generation changed during dataset build")


def _before_publish_source_recheck(_snapshot: VerifiedSourceSnapshot) -> None:
    """Dedicated deterministic test seam; production intentionally does nothing."""


def _require_supported_policies(
    candidate_policy: CandidatePolicy, label_policy: LabelPolicy
) -> None:
    if candidate_policy != DEFAULT_CANDIDATE_POLICY:
        raise DatasetBuildError("unsupported candidate policy")
    if label_policy != DEFAULT_LABEL_POLICY:
        raise DatasetBuildError("unsupported label policy")


def _effective_splits(metadata: dict[str, Any]) -> dict[str, tuple[datetime, datetime]]:
    requested_start = _metadata_hour(metadata, "requested_start")
    if requested_start > SPLITS["train"][0]:
        raise DatasetBuildError("verified source starts after the fixed train range")
    effective_end = _metadata_hour(metadata, "effective_end")
    if effective_end <= SPLITS["test"][1]:
        raise DatasetBuildError("verified source does not cover the fixed test range")
    return {**SPLITS, FINAL_HOLDOUT: (SPLITS["test"][1], effective_end)}


def _metadata_hour(metadata: dict[str, Any], key: str) -> datetime:
    try:
        value = metadata[key]
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (KeyError, ValueError) as exc:
        raise DatasetBuildError(f"metadata {key} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise DatasetBuildError(f"metadata {key} must be UTC")
    parsed = parsed.astimezone(UTC)
    if parsed.minute or parsed.second or parsed.microsecond:
        raise DatasetBuildError(f"metadata {key} must be UTC hour-aligned")
    return parsed


def _validate_fixed_coverage(
    candles: list[Candle],
    splits: dict[str, tuple[datetime, datetime]],
    missing: set[datetime],
    non_tradable: set[datetime],
) -> None:
    actual = {candle.open_time for candle in candles}
    if len(actual) != len(candles):
        raise DatasetBuildError("source contains duplicate candles")
    if not non_tradable <= actual:
        raise DatasetBuildError("verified non-tradable candle is absent from the source")
    for name, (start, end) in splits.items():
        expected: set[datetime] = set()
        cursor = start
        while cursor < end:
            expected.add(cursor)
            cursor += HOUR
        observed = actual & expected
        absent = expected - observed
        audited_absent = missing & expected
        if absent != audited_absent:
            raise DatasetBuildError(f"fixed split coverage is incomplete: {name}")
        if not observed:
            raise DatasetBuildError(f"fixed split is empty: {name}")


def _segments(
    candles: list[Candle],
    splits: dict[str, tuple[datetime, datetime]],
    missing: set[datetime],
    non_tradable: set[datetime],
) -> list[_Segment]:
    result: list[_Segment] = []
    invalid = missing | non_tradable
    for split, (start, end) in splits.items():
        members = [candle for candle in candles if start <= candle.open_time < end]
        current: list[Candle] = []
        number = 0
        previous: datetime | None = None
        for candle in members:
            if candle.open_time in invalid:
                if current:
                    result.append(_Segment(split, f"{split}-{number}", current))
                    number += 1
                    current = []
                previous = None
                continue
            if previous is not None and candle.open_time != previous + HOUR:
                if current:
                    result.append(_Segment(split, f"{split}-{number}", current))
                    number += 1
                    current = []
            current.append(candle)
            previous = candle.open_time
        if current:
            result.append(_Segment(split, f"{split}-{number}", current))
    return result


def _segment_frame(segment: _Segment) -> pd.DataFrame:
    """Use the existing causal definitions, but only within one continuous segment."""
    frame = build_features(segment.candles).copy()
    frame["open_time"] = [candle.open_time for candle in segment.candles]
    frame["segment_id"] = segment.segment_id
    frame["split"] = segment.split
    return frame


def _candidate_indices(frame: pd.DataFrame, policy: CandidatePolicy) -> list[int]:
    mask = long_entry_candidate_mask(frame, policy)
    # `shift(1)` is evaluated only in this split/segment frame, so no
    # crossover comparison can bridge either boundary.
    return [int(position) for position in np.flatnonzero(mask.to_numpy()) if position > 0]


def _exit_fill(reference: Decimal, policy: LabelPolicy) -> Decimal:
    return reference * (Decimal("1") - policy.exit_slippage_rate)


def _label(
    candles: list[Candle], candidate_index: int, atr: Decimal, policy: LabelPolicy
) -> tuple[str, int | None, datetime | None, Decimal | None]:
    """Return outcome/target/end/net return; no intrabar path is invented."""
    entry_index = candidate_index + 1
    final_index = entry_index + policy.max_holding_candles - 1
    if entry_index >= len(candles):
        return "missing_next_open", None, None, None
    if final_index >= len(candles):
        return "horizon_incomplete", None, None, None
    entry_reference = candles[entry_index].open
    entry_fill = entry_reference * (Decimal("1") + policy.entry_slippage_rate)
    entry_cost = entry_fill * (Decimal("1") + policy.entry_fee_rate)
    stop = entry_reference - policy.stop_atr_multiple * atr
    target = entry_reference + policy.profit_atr_multiple * atr
    for candle in candles[entry_index : final_index + 1]:
        # The opening price is known before any later OHLC movement. Both gap
        # outcomes are therefore decided before conservative intrabar ordering.
        if candle.open <= stop:
            exit_reference, outcome, target_value, available_at = (
                candle.open,
                "stop",
                0,
                candle.open_time,
            )
        elif candle.open >= target:
            exit_reference, outcome, target_value, available_at = (
                target,
                "profit",
                1,
                candle.open_time,
            )
        elif candle.low <= stop:
            exit_reference, outcome, target_value, available_at = (
                stop,
                "stop",
                0,
                candle.close_time,
            )
        elif candle.high >= target:
            exit_reference, outcome, target_value, available_at = (
                target,
                "profit",
                1,
                candle.close_time,
            )
        else:
            continue
        exit_fill = _exit_fill(exit_reference, policy)
        proceeds = exit_fill * (Decimal("1") - policy.exit_fee_rate)
        return outcome, target_value, available_at, proceeds / entry_cost - Decimal("1")
    return "timeout", None, candles[final_index].close_time, None


def _as_feature_values(row: pd.Series) -> dict[str, float]:
    values: dict[str, float] = {}
    for column in FEATURE_COLUMNS:
        value = row[column]
        if pd.isna(value) or not np.isfinite(float(value)):
            raise DatasetBuildError("candidate contains a missing or non-finite feature")
        values[column] = float(value)
    return values


def _build_rows(
    segments: list[_Segment], candidate_policy: CandidatePolicy, policy: LabelPolicy
) -> tuple[
    dict[str, list[dict[str, Any]]], dict[str, int], dict[str, dict[str, int]], dict[str, int]
]:
    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in [*SPLITS, FINAL_HOLDOUT]}
    candidates = {name: 0 for name in rows}
    labels = {name: {"positive": 0, "negative": 0, "timeout": 0} for name in SPLITS}
    exclusions: dict[str, int] = {}
    for segment in segments:
        frame = _segment_frame(segment)
        for index in _candidate_indices(frame, candidate_policy):
            candidates[segment.split] += 1
            candle = segment.candles[index]
            entry_index = index + 1
            # The entry must be a candle in the same immutable split/segment.
            if entry_index >= len(segment.candles):
                exclusions["missing_next_open"] = exclusions.get("missing_next_open", 0) + 1
                continue
            row = frame.iloc[index]
            payload: dict[str, Any] = {
                "split": segment.split,
                "segment_id": segment.segment_id,
                "signal_time": _iso(candle.close_time),
                "entry_time": _iso(segment.candles[entry_index].open_time),
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                **_as_feature_values(row),
            }
            if segment.split == FINAL_HOLDOUT:
                rows[segment.split].append(payload)
                continue
            atr = Decimal(str(row["atr14"]))
            outcome, target, end_time, net_return = _label(segment.candles, index, atr, policy)
            if outcome == "timeout":
                labels[segment.split]["timeout"] += 1
                exclusions["timeout"] = exclusions.get("timeout", 0) + 1
                continue
            if outcome in {"missing_next_open", "horizon_incomplete"}:
                exclusions[outcome] = exclusions.get(outcome, 0) + 1
                continue
            if target is None or end_time is None or net_return is None:
                raise DatasetBuildError("invalid development label result")
            payload.update(
                {
                    "label_end_time": _iso(end_time),
                    "target": target,
                    "outcome": outcome,
                    "net_return_after_costs": str(net_return),
                }
            )
            labels[segment.split]["positive" if target else "negative"] += 1
            rows[segment.split].append(payload)
    return rows, candidates, labels, exclusions


def _output_columns(split: str) -> list[str]:
    return [*AUDIT_COLUMNS, *FEATURE_COLUMNS] + ([] if split == FINAL_HOLDOUT else LABEL_COLUMNS)


def _write_csv(path: Path, rows: list[dict[str, Any]], split: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_output_columns(split), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            serialized = {
                key: format(value, ".12g") if isinstance(value, float) else value
                for key, value in row.items()
            }
            writer.writerow(serialized)
        handle.flush()
        os.fsync(handle.fileno())


def _parse_output_time(value: str, field: str) -> datetime:
    if not isinstance(value, str):
        raise DatasetBuildError(f"invalid staged {field}")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DatasetBuildError(f"invalid staged {field}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise DatasetBuildError(f"staged {field} must be UTC")
    return parsed.astimezone(UTC)


def _parse_hour_time(value: object, field: str) -> datetime:
    """Parse an explicit UTC hour boundary from immutable manifest metadata."""
    if not isinstance(value, str):
        raise DatasetBuildError(f"invalid staged {field}")
    parsed = _parse_output_time(value, field)
    if parsed.minute or parsed.second or parsed.microsecond:
        raise DatasetBuildError(f"staged {field} must be UTC hour-aligned")
    return parsed


def _verify_staged_csv(
    path: Path,
    split: str,
    split_bounds: tuple[datetime, datetime],
    segment_bounds: dict[str, tuple[str, datetime, datetime]],
) -> DatasetFileStats:
    """Validate the generated schema and ensure no future label leaks to holdout."""
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != _output_columns(split):
                raise DatasetBuildError("staged dataset schema mismatch")
            previous_signal: datetime | None = None
            first_signal: datetime | None = None
            count = 0
            positive = 0
            negative = 0
            for row in reader:
                if set(row) != set(_output_columns(split)):
                    raise DatasetBuildError("staged dataset row schema mismatch")
                if row["split"] != split or row["feature_schema_version"] != FEATURE_SCHEMA_VERSION:
                    raise DatasetBuildError("staged dataset row policy mismatch")
                signal = _parse_output_time(row["signal_time"], "signal_time")
                entry = _parse_output_time(row["entry_time"], "entry_time")
                start, end = split_bounds
                if not (start <= signal <= entry < end):
                    raise DatasetBuildError("staged signal/entry leaves its split")
                if previous_signal is not None and signal <= previous_signal:
                    raise DatasetBuildError("staged signal rows are not strictly ordered")
                if first_signal is None:
                    first_signal = signal
                previous_signal = signal
                segment = segment_bounds.get(row["segment_id"])
                if (
                    segment is None
                    or segment[0] != split
                    or not (segment[1] <= signal <= entry < segment[2])
                ):
                    raise DatasetBuildError("staged row has an invalid segment")
                for column in FEATURE_COLUMNS:
                    value = float(row[column])
                    if not np.isfinite(value):
                        raise DatasetBuildError("staged dataset has non-finite feature")
                if split != FINAL_HOLDOUT:
                    if row["target"] not in {"0", "1"} or row["outcome"] not in {"stop", "profit"}:
                        raise DatasetBuildError("staged development label is invalid")
                    if (row["target"] == "1") != (row["outcome"] == "profit"):
                        raise DatasetBuildError("staged target and outcome disagree")
                    label_end = _parse_output_time(row["label_end_time"], "label_end_time")
                    if not (entry <= label_end < end and label_end <= segment[2]):
                        raise DatasetBuildError("staged label leaves its split or segment")
                    if not Decimal(row["net_return_after_costs"]).is_finite():
                        raise DatasetBuildError("staged net return is non-finite")
                    if row["target"] == "1":
                        positive += 1
                    else:
                        negative += 1
                count += 1
            return DatasetFileStats(
                count,
                None if split == FINAL_HOLDOUT else positive,
                None if split == FINAL_HOLDOUT else negative,
                first_signal,
                previous_signal,
            )
    except DatasetBuildError:
        raise
    except (OSError, ValueError, TypeError, KeyError, InvalidOperation, csv.Error) as exc:
        raise DatasetBuildError("could not validate staged dataset output") from exc


def _output_times(rows: list[dict[str, Any]]) -> dict[str, str | None]:
    times = [str(row["signal_time"]) for row in rows]
    return {
        "first_signal_time": min(times) if times else None,
        "last_signal_time": max(times) if times else None,
    }


def _manifest_segment_bounds(
    payload: dict[str, Any], splits: dict[str, tuple[datetime, datetime]]
) -> dict[str, tuple[str, datetime, datetime]]:
    records = payload.get("segments")
    if not isinstance(records, dict) or not records:
        raise DatasetBuildError("dataset manifest segments are invalid")
    if list(records) != sorted(records):
        raise DatasetBuildError("dataset manifest segment ordering is invalid")
    result: dict[str, tuple[str, datetime, datetime]] = {}
    for segment_id, record in records.items():
        if (
            not isinstance(segment_id, str)
            or not segment_id.strip()
            or not isinstance(record, dict)
        ):
            raise DatasetBuildError("dataset manifest segment is invalid")
        split = record.get("split")
        if not isinstance(split, str) or split not in splits:
            raise DatasetBuildError("dataset manifest segment split is invalid")
        start = _parse_hour_time(record.get("start"), "segment start")
        end = _parse_hour_time(record.get("end"), "segment end")
        if end <= start:
            raise DatasetBuildError("dataset manifest segment duration is invalid")
        split_start, split_end = splits[split]
        if not (split_start <= start < end <= split_end):
            raise DatasetBuildError("dataset manifest segment leaves its split")
        if segment_id in result:
            raise DatasetBuildError("dataset manifest segment ID is duplicated")
        result[segment_id] = (split, start, end)
    if {record[0] for record in result.values()} != set(CANONICAL_SPLIT_NAMES):
        raise DatasetBuildError("dataset manifest segments do not cover every split")
    by_split: dict[str, list[tuple[datetime, datetime, str]]] = {
        split: [] for split in CANONICAL_SPLIT_NAMES
    }
    for segment_id, (split, start, end) in result.items():
        by_split[split].append((start, end, segment_id))
    for split_records in by_split.values():
        previous_end: datetime | None = None
        for start, end, _ in sorted(split_records):
            if previous_end is not None and start < previous_end:
                raise DatasetBuildError("dataset manifest segments overlap")
            previous_end = end
    return result


def _manifest_splits(payload: dict[str, Any]) -> dict[str, tuple[datetime, datetime]]:
    raw = payload.get("splits")
    if not isinstance(raw, dict) or set(raw) != set(CANONICAL_SPLIT_NAMES):
        raise DatasetBuildError("dataset manifest splits are invalid")
    result: dict[str, tuple[datetime, datetime]] = {}
    for split in CANONICAL_SPLIT_NAMES:
        record = raw.get(split)
        if not isinstance(record, dict):
            raise DatasetBuildError("dataset manifest split is invalid")
        start = _parse_hour_time(record.get("start"), "split start")
        end = _parse_hour_time(record.get("end"), "split end")
        if end <= start:
            raise DatasetBuildError("dataset manifest split duration is invalid")
        result[split] = (start, end)
    for split, bounds in SPLITS.items():
        if result[split] != bounds:
            raise DatasetBuildError("dataset manifest split boundaries are noncanonical")
    holdout_start, holdout_end = result[FINAL_HOLDOUT]
    if holdout_start != SPLITS["test"][1] or holdout_end <= holdout_start:
        raise DatasetBuildError("dataset manifest final holdout boundaries are noncanonical")
    return result


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate manifest keys instead of silently accepting the last one."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DatasetBuildError("dataset generation manifest has duplicate keys")
        result[key] = value
    return result


def _validate_generation(directory: Path) -> None:
    """Validate a published generation, normalizing expected corruption failures."""
    try:
        _validate_generation_contents(directory)
    except DatasetBuildError:
        raise
    except (OSError, ValueError, TypeError, KeyError, InvalidOperation, csv.Error) as exc:
        raise DatasetBuildError("dataset generation is corrupt or incomplete") from exc


def _freeze_manifest_value(value: Any) -> Any:
    """Return an immutable snapshot so downstream consumers cannot alter validation input."""
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_manifest_value(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_manifest_value(item) for item in value)
    return value


def _validate_generation_contents(
    directory: Path, *, development_only: bool = False
) -> Mapping[str, Any]:
    manifest_path = directory / "dataset.manifest.json"
    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8"), object_pairs_hook=_unique_json_object
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetBuildError("dataset generation manifest is invalid") from exc
    if not isinstance(manifest, dict):
        raise DatasetBuildError("dataset generation manifest is invalid")
    required_versions = {
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "candidate_policy_version": CANDIDATE_POLICY_VERSION,
        "label_policy_version": LABEL_POLICY_VERSION,
        "segmentation_policy_version": SEGMENTATION_POLICY_VERSION,
        "holdout_policy_version": HOLDOUT_POLICY_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
    }
    if any(manifest.get(key) != value for key, value in required_versions.items()):
        raise DatasetBuildError("dataset generation schema policy mismatch")
    if (
        not isinstance(manifest.get("dataset_generation_id"), str)
        or not isinstance(manifest.get("candidate_policy"), dict)
        or not isinstance(manifest.get("label_policy"), dict)
        or manifest.get("feature_columns") != FEATURE_COLUMNS
        or any(
            not isinstance(manifest.get(key), str)
            for key in (
                "source_csv_sha256",
                "source_metadata_sha256",
                "anomaly_report_sha256",
                "source_generation_id",
            )
        )
    ):
        raise DatasetBuildError("dataset generation identity metadata is invalid")
    checksums = manifest.get("output_file_sha256")
    row_counts = manifest.get("row_counts")
    if not isinstance(checksums, dict) or not isinstance(row_counts, dict):
        raise DatasetBuildError("dataset generation checksum or count metadata is invalid")
    splits = _manifest_splits(manifest)
    segments = _manifest_segment_bounds(manifest, splits)
    if development_only:
        expected_files = {f"{split}.csv" for split in CANONICAL_SPLIT_NAMES}
        candidates = manifest.get("candidate_counts")
        coverage = manifest.get("split_signal_coverage")
        exclusions = manifest.get("excluded_row_counts")
        if (
            set(checksums) != expected_files
            or not all(
                isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
                for value in checksums.values()
            )
            or set(row_counts) != set(CANONICAL_SPLIT_NAMES)
            or not isinstance(candidates, dict)
            or set(candidates) != set(CANONICAL_SPLIT_NAMES)
            or not isinstance(coverage, dict)
            or set(coverage) != set(CANONICAL_SPLIT_NAMES)
            or type(row_counts[FINAL_HOLDOUT]) is not int
            or row_counts[FINAL_HOLDOUT] < 0
            or type(candidates[FINAL_HOLDOUT]) is not int
            or candidates[FINAL_HOLDOUT] < 0
            or candidates[FINAL_HOLDOUT] < row_counts[FINAL_HOLDOUT]
            or not isinstance(exclusions, dict)
        ):
            raise DatasetBuildError("development final-holdout metadata is invalid")
        missing_next_open = exclusions.get("missing_next_open", 0)
        if (
            type(missing_next_open) is not int
            or missing_next_open < 0
            or candidates[FINAL_HOLDOUT] - row_counts[FINAL_HOLDOUT] > missing_next_open
        ):
            raise DatasetBuildError("development final-holdout metadata is invalid")
        final_coverage = coverage[FINAL_HOLDOUT]
        if not isinstance(final_coverage, dict) or set(final_coverage) != {
            "first_signal_time",
            "last_signal_time",
        }:
            raise DatasetBuildError("development final-holdout coverage is invalid")
        first, last = (
            final_coverage.get("first_signal_time"),
            final_coverage.get("last_signal_time"),
        )
        if row_counts[FINAL_HOLDOUT] == 0:
            if first is not None or last is not None:
                raise DatasetBuildError("empty final-holdout coverage is invalid")
        else:
            final_start, final_end = splits[FINAL_HOLDOUT]
            first_time = _parse_hour_time(first, "final holdout first signal")
            last_time = _parse_hour_time(last, "final holdout last signal")
            first_in_final_segment = any(
                split == FINAL_HOLDOUT and start <= first_time < end
                for split, start, end in segments.values()
            )
            last_in_final_segment = any(
                split == FINAL_HOLDOUT and start <= last_time < end
                for split, start, end in segments.values()
            )
            if not (
                final_start <= first_time <= last_time < final_end
                and first_in_final_segment
                and last_in_final_segment
            ):
                raise DatasetBuildError("development final-holdout coverage is invalid")
    try:
        candidate_policy = CandidatePolicy(**manifest["candidate_policy"])
        label_values = dict(manifest["label_policy"])
        for key in (
            "stop_atr_multiple",
            "profit_atr_multiple",
            "entry_fee_rate",
            "exit_fee_rate",
            "entry_slippage_rate",
            "exit_slippage_rate",
        ):
            label_values[key] = Decimal(str(label_values[key]))
        label_policy = LabelPolicy(**label_values)
        source_checksums = {
            key: manifest[key]
            for key in ("source_csv_sha256", "source_metadata_sha256", "anomaly_report_sha256")
        }
    except (KeyError, TypeError, ValueError, InvalidOperation) as exc:
        raise DatasetBuildError("dataset generation policy metadata is invalid") from exc
    if manifest.get("symbol") != "BTC/USDT" or manifest.get("timeframe") != "1h":
        raise DatasetBuildError("dataset generation market metadata is invalid")
    if (
        candidate_policy.symbol != manifest["symbol"]
        or candidate_policy.timeframe != manifest["timeframe"]
    ):
        raise DatasetBuildError("dataset generation candidate market metadata is invalid")
    _require_supported_policies(candidate_policy, label_policy)
    if manifest["dataset_generation_id"] != _generation_id(
        source_checksums,
        manifest["source_generation_id"],
        splits,
        candidate_policy,
        label_policy,
    ):
        raise DatasetBuildError("dataset generation identity does not match its policy")
    file_stats: dict[str, DatasetFileStats] = {}
    validated_splits = SPLITS if development_only else CANONICAL_SPLIT_NAMES
    for split in validated_splits:
        filename = f"{split}.csv"
        checksum = checksums.get(filename)
        if not isinstance(checksum, str) or checksum != _file_sha256(directory / filename):
            raise DatasetBuildError("dataset generation output checksum mismatch")
        stats = _verify_staged_csv(directory / filename, split, splits[split], segments)
        file_stats[split] = stats
        if row_counts.get(split) != stats.row_count:
            raise DatasetBuildError("dataset generation output row count mismatch")
    coverage = manifest.get("split_signal_coverage")
    if not isinstance(coverage, dict):
        raise DatasetBuildError("dataset generation signal coverage is invalid")
    for split, stats in file_stats.items():
        record = coverage.get(split)
        if (
            not isinstance(record, dict)
            or record.get("first_signal_time")
            != (_iso(stats.first_signal_time) if stats.first_signal_time else None)
            or record.get("last_signal_time")
            != (_iso(stats.last_signal_time) if stats.last_signal_time else None)
        ):
            raise DatasetBuildError("dataset generation signal coverage mismatch")
    label_counts = manifest.get("development_label_counts")
    trainable = manifest.get("development_trainable")
    if (
        not isinstance(label_counts, dict)
        or not isinstance(trainable, dict)
        or set(label_counts) != set(SPLITS)
        or set(trainable) != set(SPLITS)
    ):
        raise DatasetBuildError("dataset generation development metadata is invalid")
    for split in SPLITS:
        counts = label_counts.get(split)
        if not isinstance(counts, dict) or set(counts) != {"positive", "negative", "timeout"}:
            raise DatasetBuildError("dataset generation label counts are invalid")
        positive, negative = counts.get("positive"), counts.get("negative")
        timeout = counts.get("timeout")
        if (
            type(positive) is not int
            or type(negative) is not int
            or type(timeout) is not int
            or positive < 0
            or negative < 0
            or timeout < 0
            or row_counts.get(split) != positive + negative
            or file_stats[split].positive_count != positive
            or file_stats[split].negative_count != negative
        ):
            raise DatasetBuildError("dataset generation label counts are invalid")
        if type(trainable.get(split)) is not bool or trainable[split] is not (
            positive > 0 and negative > 0
        ):
            raise DatasetBuildError("dataset generation trainable status is invalid")
    return cast(Mapping[str, Any], _freeze_manifest_value(manifest))


def validate_development_dataset_generation(directory: Path) -> Mapping[str, Any]:
    """Validate only train/validation/test without any final-holdout filesystem access."""
    try:
        return _validate_generation_contents(directory, development_only=True)
    except DatasetBuildError:
        raise
    except (OSError, ValueError, TypeError, KeyError, InvalidOperation, csv.Error) as exc:
        raise DatasetBuildError("development dataset generation is corrupt or incomplete") from exc


def _generation_is_valid(directory: Path) -> bool:
    try:
        _validate_generation(directory)
    except DatasetBuildError:
        return False
    return True


def _recover_generation(destination: Path) -> None:
    """Recover a complete previous directory after an interrupted Windows rename."""
    backup = destination.with_name(f".{destination.name}.previous")
    destination_exists, backup_exists = destination.exists(), backup.exists()
    destination_valid = destination_exists and _generation_is_valid(destination)
    backup_valid = backup_exists and _generation_is_valid(backup)
    if destination_valid:
        # Destination is authoritative. A stale/corrupt backup must never
        # replace it, and can be removed only after that validation succeeds.
        if backup_exists:
            shutil.rmtree(backup)
        return
    if backup_valid:
        if destination_exists:
            shutil.rmtree(destination)
        os.replace(backup, destination)
        _validate_generation(destination)
        return
    if destination_exists or backup_exists:
        raise DatasetBuildError("no valid dataset generation is available for recovery")


def _atomic_publish(staged: Path, destination: Path) -> None:
    """Promote a verified directory and retain/recover the previous generation."""
    _recover_generation(destination)
    _validate_generation(staged)
    backup = destination.with_name(f".{destination.name}.previous")
    moved_previous = False
    try:
        if destination.exists():
            os.replace(destination, backup)
            moved_previous = True
        os.replace(staged, destination)
    except Exception:
        if moved_previous and backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    try:
        _validate_generation(destination)
    except Exception:
        if moved_previous and backup.exists():
            if destination.exists():
                shutil.rmtree(destination)
            os.replace(backup, destination)
        elif not moved_previous and destination.exists():
            # First publication: do not leave a corrupt directory which would
            # make the next fresh build fail closed forever.
            shutil.rmtree(destination)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def build_ml_dataset(
    input_path: Path,
    output_dir: Path,
    *,
    candidate_policy: CandidatePolicy = DEFAULT_CANDIDATE_POLICY,
    label_policy: LabelPolicy = DEFAULT_LABEL_POLICY,
) -> DatasetBuildSummary:
    """Verify a Vision generation and publish a deterministic ML dataset generation."""
    _require_supported_policies(candidate_policy, label_policy)
    snapshot = _capture_verified_source(input_path)
    non_tradable = _non_tradable_open_times(snapshot.anomaly_report_path)
    splits = _effective_splits(dict(snapshot.metadata))
    _validate_fixed_coverage(
        list(snapshot.candles), splits, set(snapshot.missing_open_times), non_tradable
    )
    segments = _segments(
        list(snapshot.candles), splits, set(snapshot.missing_open_times), non_tradable
    )
    rows, candidate_counts, label_counts, exclusions = _build_rows(
        segments, candidate_policy, label_policy
    )
    trainable_splits = {
        name: counts["positive"] > 0 and counts["negative"] > 0
        for name, counts in label_counts.items()
    }
    segment_bounds = {
        segment.segment_id: (
            segment.split,
            segment.candles[0].open_time,
            segment.candles[-1].close_time,
        )
        for segment in segments
    }
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=parent))
    try:
        checksums: dict[str, str] = {}
        for split, split_rows in rows.items():
            filename = f"{split}.csv"
            target = staging / filename
            _write_csv(target, split_rows, split)
            checksums[filename] = _file_sha256(target)
        source_checksums = {
            "source_csv_sha256": snapshot.csv_sha256,
            "source_metadata_sha256": snapshot.metadata_sha256,
            "anomaly_report_sha256": snapshot.anomaly_report_sha256,
        }
        generation_id = _generation_id(
            source_checksums,
            snapshot.source_generation_id,
            splits,
            candidate_policy,
            label_policy,
        )
        manifest = {
            "dataset_generation_id": generation_id,
            **source_checksums,
            "anomaly_report": snapshot.anomaly_report_path.name,
            "source_generation_id": snapshot.source_generation_id,
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "splits": {
                name: {"start": _iso(start), "end": _iso(end)}
                for name, (start, end) in splits.items()
            },
            "dataset_schema_version": DATASET_SCHEMA_VERSION,
            "candidate_policy_version": CANDIDATE_POLICY_VERSION,
            "label_policy_version": LABEL_POLICY_VERSION,
            "segmentation_policy_version": SEGMENTATION_POLICY_VERSION,
            "holdout_policy_version": HOLDOUT_POLICY_VERSION,
            "candidate_policy": asdict(candidate_policy),
            "label_policy": {
                key: str(value) if isinstance(value, Decimal) else value
                for key, value in asdict(label_policy).items()
            },
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_columns": FEATURE_COLUMNS,
            "code_commit": _code_commit(),
            "row_counts": {name: len(value) for name, value in rows.items()},
            "candidate_counts": candidate_counts,
            "development_label_counts": label_counts,
            "development_trainable": trainable_splits,
            "excluded_row_counts": exclusions,
            "split_signal_coverage": {name: _output_times(value) for name, value in rows.items()},
            "output_file_sha256": checksums,
            "interruption_summary": {
                "policy_version": SEGMENTATION_POLICY_VERSION,
                "verified_missing_open_times": [
                    _iso(value) for value in sorted(snapshot.missing_open_times)
                ],
                "non_tradable_open_times": [_iso(value) for value in sorted(non_tradable)],
                "segment_count": len(segments),
            },
            "segments": {
                segment_id: {
                    "split": split,
                    "start": _iso(start),
                    "end": _iso(end),
                }
                for segment_id, (split, start, end) in segment_bounds.items()
            },
        }
        manifest_path = staging / "dataset.manifest.json"
        payload = json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n"
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # Read the staged artefacts back before publishing.  The manifest is
        # written last and acts as the generation commit marker.
        for split in rows:
            verified_stats = _verify_staged_csv(
                staging / f"{split}.csv", split, splits[split], segment_bounds
            )
            if verified_stats.row_count != len(rows[split]):
                raise DatasetBuildError("staged output row count mismatch")
            if _file_sha256(staging / f"{split}.csv") != checksums[f"{split}.csv"]:
                raise DatasetBuildError("staged output checksum mismatch")
        _before_publish_source_recheck(snapshot)
        _reverify_source_snapshot(snapshot)
        _atomic_publish(staging, output_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return DatasetBuildSummary(
        generation_id,
        snapshot.source_generation_id,
        len(snapshot.candles),
        len(segments),
        candidate_counts,
        label_counts,
        exclusions,
        checksums,
        trainable_splits,
        candidate_policy,
        label_policy,
    )
