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
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Final

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
from trading_bot.settings import BotSettings
from trading_bot.strategy.ema_volume_atr import long_entry_candidate_mask

HOUR: Final = timedelta(hours=1)
SPLITS: Final = {
    "train": (datetime(2022, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC)),
    "validation": (datetime(2025, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC)),
    "test": (datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 5, 1, tzinfo=UTC)),
}
FINAL_HOLDOUT: Final = "final_holdout"
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
class DatasetBuildSummary:
    generation_id: str
    source_generation_id: str
    source_candle_count: int
    segment_count: int
    candidate_counts: dict[str, int]
    label_counts: dict[str, dict[str, int]]
    exclusion_counts: dict[str, int]
    output_checksums: dict[str, str]


@dataclass(frozen=True, slots=True)
class _Segment:
    split: str
    segment_id: str
    candles: list[Candle]


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _file_sha256(path: Path) -> str:
    return csv_sha256(path)


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


def _read_verified_source(input_path: Path) -> tuple[list[Candle], dict[str, Any], set[datetime]]:
    sidecar = metadata_path(input_path)
    report_path = input_path.with_name(f"{input_path.stem}.anomalies.json")
    if not input_path.exists() or not sidecar.exists() or not report_path.exists():
        raise DatasetBuildError("verified CSV, metadata, and anomaly report are all required")
    try:
        verified = verify_metadata_checksum(input_path)
        metadata_raw = json.loads(sidecar.read_text(encoding="utf-8"))
    except (CsvDataError, OSError, json.JSONDecodeError) as exc:
        raise DatasetBuildError("source verification failed") from exc
    if not verified or not isinstance(metadata_raw, dict):
        raise DatasetBuildError("source metadata verification failed")
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
    return candles, metadata_raw, allowed_missing


def _non_tradable_open_times(input_path: Path) -> set[datetime]:
    report_path = input_path.with_name(f"{input_path.stem}.anomalies.json")
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


def _effective_splits(metadata: dict[str, Any]) -> dict[str, tuple[datetime, datetime]]:
    try:
        effective_end = datetime.fromisoformat(
            str(metadata["effective_end"]).replace("Z", "+00:00")
        )
    except (KeyError, ValueError) as exc:
        raise DatasetBuildError("metadata effective_end is invalid") from exc
    if effective_end.tzinfo is None or effective_end.utcoffset() != timedelta(0):
        raise DatasetBuildError("metadata effective_end must be UTC")
    effective_end = effective_end.astimezone(UTC)
    if effective_end <= SPLITS["test"][1]:
        raise DatasetBuildError("verified source does not cover the fixed test range")
    return {**SPLITS, FINAL_HOLDOUT: (SPLITS["test"][1], effective_end)}


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


def _candidate_indices(frame: pd.DataFrame, settings: BotSettings) -> list[int]:
    mask = long_entry_candidate_mask(frame, settings)
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
        # Gap through stop cannot obtain the better barrier fill.  A target is
        # deliberately capped at its barrier, avoiding an optimistic gap price.
        if candle.open <= stop:
            exit_reference, outcome, target_value = candle.open, "stop", 0
        elif candle.low <= stop:
            exit_reference, outcome, target_value = stop, "stop", 0
        elif candle.open >= target or candle.high >= target:
            exit_reference, outcome, target_value = target, "profit", 1
        else:
            continue
        exit_fill = _exit_fill(exit_reference, policy)
        proceeds = exit_fill * (Decimal("1") - policy.exit_fee_rate)
        return outcome, target_value, candle.open_time, proceeds / entry_cost - Decimal("1")
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
    segments: list[_Segment], settings: BotSettings, policy: LabelPolicy
) -> tuple[
    dict[str, list[dict[str, Any]]], dict[str, int], dict[str, dict[str, int]], dict[str, int]
]:
    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in [*SPLITS, FINAL_HOLDOUT]}
    candidates = {name: 0 for name in rows}
    labels = {name: {"positive": 0, "negative": 0, "timeout": 0} for name in SPLITS}
    exclusions: dict[str, int] = {}
    for segment in segments:
        frame = _segment_frame(segment)
        for index in _candidate_indices(frame, settings):
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


def _verify_staged_csv(path: Path, split: str) -> None:
    """Validate the generated schema and ensure no future label leaks to holdout."""
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != _output_columns(split):
                raise DatasetBuildError("staged dataset schema mismatch")
            for row in reader:
                if set(row) != set(_output_columns(split)):
                    raise DatasetBuildError("staged dataset row schema mismatch")
                for column in FEATURE_COLUMNS:
                    value = float(row[column])
                    if not np.isfinite(value):
                        raise DatasetBuildError("staged dataset has non-finite feature")
                if split != FINAL_HOLDOUT:
                    if row["target"] not in {"0", "1"} or row["outcome"] not in {"stop", "profit"}:
                        raise DatasetBuildError("staged development label is invalid")
                    if not Decimal(row["net_return_after_costs"]).is_finite():
                        raise DatasetBuildError("staged net return is non-finite")
    except (OSError, ValueError, KeyError, InvalidOperation) as exc:
        raise DatasetBuildError("could not validate staged dataset output") from exc


def _output_times(rows: list[dict[str, Any]]) -> dict[str, str | None]:
    times = [str(row["signal_time"]) for row in rows]
    return {
        "first_signal_time": min(times) if times else None,
        "last_signal_time": max(times) if times else None,
    }


def _atomic_publish(staged: Path, destination: Path) -> None:
    backup = destination.with_name(f".{destination.name}.previous")
    if backup.exists():
        shutil.rmtree(backup)
    replaced = False
    try:
        if destination.exists():
            os.replace(destination, backup)
            replaced = True
        os.replace(staged, destination)
    except Exception:
        if replaced and backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    else:
        if backup.exists():
            shutil.rmtree(backup)


def build_ml_dataset(input_path: Path, output_dir: Path) -> DatasetBuildSummary:
    """Verify a Vision generation and publish a deterministic ML dataset generation."""
    candles, metadata, missing = _read_verified_source(input_path)
    non_tradable = _non_tradable_open_times(input_path)
    splits = _effective_splits(metadata)
    segments = _segments(candles, splits, missing, non_tradable)
    settings = BotSettings()
    policy = LabelPolicy(
        entry_fee_rate=settings.entry_fee_rate,
        exit_fee_rate=settings.exit_fee_rate,
        entry_slippage_rate=settings.entry_slippage_rate,
        exit_slippage_rate=settings.exit_slippage_rate,
    )
    rows, candidate_counts, label_counts, exclusions = _build_rows(segments, settings, policy)
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
            "source_csv_sha256": _file_sha256(input_path),
            "source_metadata_sha256": _file_sha256(metadata_path(input_path)),
            "anomaly_report_sha256": _file_sha256(
                input_path.with_name(f"{input_path.stem}.anomalies.json")
            ),
        }
        identity = hashlib.sha256(
            json.dumps(
                {
                    "source": source_checksums,
                    "features": FEATURE_COLUMNS,
                    "feature_schema_version": FEATURE_SCHEMA_VERSION,
                    "policy": asdict(policy),
                    "splits": {
                        name: (_iso(start), _iso(end)) for name, (start, end) in splits.items()
                    },
                },
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode()
        ).hexdigest()
        generation_id = identity[:32]
        manifest = {
            "dataset_generation_id": generation_id,
            **source_checksums,
            "source_generation_id": metadata["generation_id"],
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "splits": {
                name: {"start": _iso(start), "end": _iso(end)}
                for name, (start, end) in splits.items()
            },
            "label_policy": {
                key: str(value) if isinstance(value, Decimal) else value
                for key, value in asdict(policy).items()
            },
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_columns": FEATURE_COLUMNS,
            "code_commit": _code_commit(),
            "row_counts": {name: len(value) for name, value in rows.items()},
            "candidate_counts": candidate_counts,
            "development_label_counts": label_counts,
            "excluded_row_counts": exclusions,
            "split_signal_coverage": {name: _output_times(value) for name, value in rows.items()},
            "output_file_sha256": checksums,
            "interruption_summary": {
                "verified_missing_open_times": [_iso(value) for value in sorted(missing)],
                "non_tradable_open_times": [_iso(value) for value in sorted(non_tradable)],
                "segment_count": len(segments),
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
            _verify_staged_csv(staging / f"{split}.csv", split)
            if _file_sha256(staging / f"{split}.csv") != checksums[f"{split}.csv"]:
                raise DatasetBuildError("staged output checksum mismatch")
        _atomic_publish(staging, output_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return DatasetBuildSummary(
        generation_id,
        str(metadata["generation_id"]),
        len(candles),
        len(segments),
        candidate_counts,
        label_counts,
        exclusions,
        checksums,
    )
