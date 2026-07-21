from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from trading_bot.data.market_interruptions import KNOWN_MARKET_INTERRUPTIONS, find_interruption
from trading_bot.data.validation import CandleValidationError, validate_candles
from trading_bot.domain.models import Candle

CSV_FIELDS = [
    "open_time",
    "close_time",
    "symbol",
    "timeframe",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "is_closed",
]
SCHEMA_VERSION = "1.0"


class CsvDataError(ValueError):
    """A historical candle CSV is malformed or internally inconsistent."""


def _iso_utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_time(value: str, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CsvDataError(f"invalid {field}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CsvDataError(f"{field} must be timezone-aware")
    if parsed.utcoffset() != timedelta(0):
        raise CsvDataError(f"{field} must be UTC")
    return parsed.astimezone(UTC)


def _parse_row(row: dict[str, str]) -> Candle:
    if set(row) != set(CSV_FIELDS):
        raise CsvDataError("CSV schema does not match expected candle columns")
    try:
        is_closed = {"true": True, "false": False}[row["is_closed"].lower()]
        open_ = Decimal(row["open"])
        high = Decimal(row["high"])
        low = Decimal(row["low"])
        close = Decimal(row["close"])
        volume = Decimal(row["volume"])
        if not all(value.is_finite() for value in (open_, high, low, close, volume)):
            raise ValueError("non-finite decimal")
        return Candle(
            open_time=_parse_time(row["open_time"], "open_time"),
            close_time=_parse_time(row["close_time"], "close_time"),
            symbol=row["symbol"],
            timeframe=row["timeframe"],
            open=open_,
            high=high,
            low=low,
            close=close,
            volume=volume,
            is_closed=is_closed,
        )
    except (KeyError, InvalidOperation, ValueError) as exc:
        raise CsvDataError("malformed candle CSV row") from exc


def merge_candles(
    *groups: Iterable[Candle], allowed_missing_open_times: set[datetime] | None = None
) -> list[Candle]:
    by_open: dict[datetime, Candle] = {}
    for candle in (item for group in groups for item in group):
        previous = by_open.get(candle.open_time)
        if previous is not None and previous != candle:
            raise CsvDataError(f"conflicting candles at {candle.open_time.isoformat()}")
        by_open[candle.open_time] = candle
    merged = [by_open[key] for key in sorted(by_open)]
    try:
        validate_candles(merged, allowed_missing_open_times=allowed_missing_open_times)
    except CandleValidationError as exc:
        raise CsvDataError(f"invalid merged candles: {exc}") from exc
    return merged


def read_candles(
    path: Path, allowed_missing_open_times: set[datetime] | None = None
) -> list[Candle]:
    if not path.exists():
        raise CsvDataError(f"CSV file does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != CSV_FIELDS:
                raise CsvDataError("CSV schema does not match expected candle columns")
            candles = [_parse_row(row) for row in reader]
    except OSError as exc:
        raise CsvDataError(f"could not read CSV: {path}") from exc
    if not candles:
        raise CsvDataError("CSV contains no candles")
    return merge_candles(candles, allowed_missing_open_times=allowed_missing_open_times)


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_candles_atomic(
    path: Path, candles: Iterable[Candle], allowed_missing_open_times: set[datetime] | None = None
) -> list[Candle]:
    normalized = merge_candles(candles, allowed_missing_open_times=allowed_missing_open_times)
    rows = [
        {
            "open_time": _iso_utc(candle.open_time),
            "close_time": _iso_utc(candle.close_time),
            "symbol": candle.symbol,
            "timeframe": candle.timeframe,
            "open": str(candle.open),
            "high": str(candle.high),
            "low": str(candle.low),
            "close": str(candle.close),
            "volume": str(candle.volume),
            "is_closed": str(candle.is_closed).lower(),
        }
        for candle in normalized
    ]
    from io import StringIO

    stream = StringIO()
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_write(path, stream.getvalue())
    return normalized


def metadata_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.metadata.json")


def csv_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise CsvDataError(f"could not checksum CSV: {path}") from exc
    return digest.hexdigest()


def verify_metadata_checksum(path: Path) -> bool:
    sidecar = metadata_path(path)
    if not sidecar.exists():
        return False
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CsvDataError("invalid metadata JSON") from exc
    checksum = metadata.get("csv_sha256") if isinstance(metadata, dict) else None
    if not isinstance(checksum, str) or checksum != csv_sha256(path):
        raise CsvDataError("metadata CSV checksum mismatch")
    allowed_missing = _verify_anomaly_sidecar(path, metadata)
    read_candles(path, allowed_missing_open_times=allowed_missing)
    return True


def verified_missing_open_times(path: Path) -> set[datetime]:
    """Return only sidecar-verified gaps caused by an audited interruption."""
    sidecar = metadata_path(path)
    if not sidecar.exists():
        return set()
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CsvDataError("invalid metadata JSON") from exc
    if not isinstance(metadata, dict) or metadata.get("csv_sha256") != csv_sha256(path):
        raise CsvDataError("metadata CSV checksum mismatch")
    return _verify_anomaly_sidecar(path, metadata)


def contains_non_tradable_intervals(path: Path) -> bool:
    sidecar = metadata_path(path)
    if not sidecar.exists():
        return False
    verified_missing_open_times(path)
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CsvDataError("invalid metadata JSON") from exc
    return metadata.get("contains_non_tradable_intervals") is True


def _verify_anomaly_sidecar(path: Path, metadata: dict[str, Any]) -> set[datetime]:
    report_name = metadata.get("anomaly_report")
    report_checksum = metadata.get("anomaly_report_sha256")
    if report_name is None and report_checksum is None:
        return set()
    if not isinstance(report_name, str) or not isinstance(report_checksum, str):
        raise CsvDataError("metadata anomaly sidecar reference is invalid")
    report_path = path.parent / report_name
    if not report_path.exists():
        raise CsvDataError("metadata anomaly report is missing")
    if csv_sha256(report_path) != report_checksum:
        raise CsvDataError("metadata anomaly report checksum mismatch")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CsvDataError("invalid anomaly report JSON") from exc
    if not isinstance(report, dict) or not isinstance(report.get("summary"), dict):
        raise CsvDataError("invalid anomaly report summary")
    policy = report.get("policy")
    metadata_mode = metadata.get("checksum_verification_mode")
    report_mode = policy.get("checksum_verification_mode") if isinstance(policy, dict) else None
    if (
        not isinstance(policy, dict)
        or not isinstance(metadata_mode, str)
        or not isinstance(report_mode, str)
        or metadata_mode != report_mode
        or metadata_mode != "official_online"
    ):
        raise CsvDataError("invalid checksum verification mode")
    if metadata.get("generation_id") != report.get("generation_id"):
        raise CsvDataError("metadata and anomaly report generation mismatch")
    summary = report["summary"]
    records = report.get("anomalies")
    if not isinstance(records, list):
        raise CsvDataError("invalid anomaly report records")
    interruptions = report.get("market_interruptions")
    if not isinstance(interruptions, list):
        raise CsvDataError("invalid market interruption records")
    allowed_missing = _verify_market_interruptions(metadata, report, interruptions)
    accepted = sum(
        isinstance(record, dict) and record.get("accepted") is True for record in records
    )
    rejected = sum(
        isinstance(record, dict) and record.get("accepted") is False for record in records
    )
    deviations: list[int] = []
    for record in [*records, *interruptions]:
        if not isinstance(record, dict):
            raise CsvDataError("invalid anomaly report record")
        deviation_us = record.get("early_close_deviation_us")
        deviation_ms = record.get("early_close_deviation_ms")
        if (
            not isinstance(deviation_us, int)
            or deviation_us < 0
            or not isinstance(deviation_ms, str)
        ):
            raise CsvDataError("invalid anomaly deviation")
        if _milliseconds_string(deviation_us) != deviation_ms:
            raise CsvDataError("anomaly millisecond deviation does not match microseconds")
        deviations.append(deviation_us)
    maximum_deviation_us = max(deviations, default=0)
    archive_summary_count = _required_nonnegative_int(summary, "archive_candle_count")
    exact_summary_count = _required_nonnegative_int(summary, "exact_archive_timestamp_candle_count")
    expected: dict[str, int | str] = {
        "archive_candle_count": archive_summary_count,
        "exact_archive_timestamp_candle_count": exact_summary_count,
        "accepted_archive_anomaly_count": accepted,
        "rejected_timestamp_anomalies": rejected,
        "maximum_observed_early_close_us": maximum_deviation_us,
        "maximum_observed_early_close_ms": _milliseconds_string(maximum_deviation_us),
    }
    if any(summary.get(key) != value for key, value in expected.items()):
        raise CsvDataError("anomaly report summary does not match records")
    if exact_summary_count + accepted + len(interruptions) != archive_summary_count:
        raise CsvDataError("anomaly report archive counts are inconsistent")
    stored_count = _required_nonnegative_int(metadata, "stored_candle_count")
    archive_count = _required_nonnegative_int(metadata, "archive_candle_count")
    exact_count = _required_nonnegative_int(metadata, "exact_archive_timestamp_candle_count")
    accepted_count = _required_nonnegative_int(metadata, "accepted_archive_anomaly_count")
    rest_count = _required_nonnegative_int(metadata, "rest_suffix_candle_count")
    interruption_count = _required_nonnegative_int(metadata, "market_interruption_candle_count")
    if (
        archive_count != exact_count + accepted_count + interruption_count
        or archive_count + rest_count != stored_count
    ):
        raise CsvDataError("metadata source counts are inconsistent")
    candles = read_candles(path, allowed_missing_open_times=allowed_missing)
    _verify_requested_coverage(metadata, candles, allowed_missing)
    metadata_expected = {
        "accepted_anomaly_count": accepted,
        "rejected_anomaly_count": rejected,
        "maximum_timestamp_deviation_us": maximum_deviation_us,
        "maximum_timestamp_deviation_ms": _milliseconds_string(maximum_deviation_us),
        "archive_candle_count": archive_summary_count,
        "exact_archive_timestamp_candle_count": exact_summary_count,
        "accepted_archive_anomaly_count": accepted,
        "market_interruption_event_count": len(
            {record.get("event_id") for record in interruptions if isinstance(record, dict)}
        ),
        "market_interruption_candle_count": len(interruptions),
        "contains_non_tradable_intervals": bool(interruptions),
    }
    if any(metadata.get(key) != value for key, value in metadata_expected.items()):
        raise CsvDataError("metadata anomaly summary does not match report")
    return allowed_missing


def _verify_requested_coverage(
    metadata: dict[str, Any], candles: list[Candle], allowed_missing: set[datetime]
) -> None:
    requested_start = _required_utc_hour(metadata, "requested_start")
    effective_end = _required_utc_hour(metadata, "effective_end")
    if effective_end <= requested_start:
        raise CsvDataError("invalid requested metadata range")
    expected_open_times = _hourly_open_times(requested_start, effective_end)
    actual_open_times = {candle.open_time for candle in candles}
    unexpected = actual_open_times - expected_open_times
    actual_missing = expected_open_times - actual_open_times
    if any(value not in expected_open_times for value in allowed_missing):
        raise CsvDataError("verified missing timestamp is outside requested range")
    if unexpected:
        raise CsvDataError("CSV contains timestamp outside requested range")
    if actual_missing != allowed_missing:
        raise CsvDataError("CSV gaps do not match verified market interruption records")
    requested_count = _required_nonnegative_int(metadata, "requested_range_candle_count")
    stored_count = _required_nonnegative_int(metadata, "stored_candle_count")
    missing_count = _required_nonnegative_int(metadata, "missing_candle_count")
    if (
        requested_count != len(expected_open_times)
        or stored_count != len(actual_open_times)
        or missing_count != len(actual_missing)
        or stored_count + missing_count != requested_count
    ):
        raise CsvDataError("metadata requested coverage counts are inconsistent")


def _required_utc_hour(payload: dict[str, Any], key: str) -> datetime:
    value = payload.get(key)
    if not isinstance(value, str):
        raise CsvDataError(f"invalid {key}")
    parsed = _parse_time(value, key)
    if parsed.minute or parsed.second or parsed.microsecond:
        raise CsvDataError(f"{key} must be UTC hour-aligned")
    return parsed


def _hourly_open_times(start: datetime, end: datetime) -> set[datetime]:
    result: set[datetime] = set()
    cursor = start
    while cursor < end:
        result.add(cursor)
        cursor += timedelta(hours=1)
    return result


def _verify_market_interruptions(
    metadata: dict[str, Any], report: dict[str, Any], records: list[Any]
) -> set[datetime]:
    summary = report["summary"]
    if not isinstance(summary, dict):
        raise CsvDataError("invalid anomaly report summary")
    event_ids: set[str] = set()
    missing: set[datetime] = set()
    for record in records:
        if not isinstance(record, dict):
            raise CsvDataError("invalid market interruption record")
        try:
            event = find_interruption(
                KNOWN_MARKET_INTERRUPTIONS,
                archive_name=record["archive"],
                archive_sha256=record["archive_sha256"],
                raw_open_timestamp=record["raw_open_timestamp"],
                raw_close_timestamp=record["raw_close_timestamp"],
            )
        except (KeyError, TypeError):
            event = None
        if (
            event is None
            or record.get("event_id") != event.event_id
            or record.get("event_type") != "market_interruption"
            or record.get("tradable") is not False
            or record.get("market_type") != event.market_type
            or record.get("symbol") != event.symbol
            or record.get("timeframe") != event.timeframe
            or record.get("open_time")
            != _iso_utc(datetime.fromtimestamp(event.raw_open_timestamp / 1000, UTC))
            or record.get("official_source_urls") != list(event.official_source_urls)
            or record.get("event_start") != _iso_utc(event.event_start)
            or record.get("event_end") != _iso_utc(event.event_end)
            or record.get("missing_open_times")
            != [_iso_utc(value) for value in event.missing_open_times]
            or record.get("quality") != "known_market_interruption"
            or record.get("accepted") is not True
            or not isinstance(record.get("row_number"), int)
            or record["row_number"] < 0
            or record.get("timestamp_unit") != "milliseconds"
            or record.get("expected_close_timestamp") != event.raw_open_timestamp + 3_599_999
            or record.get("early_close_deviation_us")
            != (event.raw_open_timestamp + 3_599_999 - event.raw_close_timestamp) * 1_000
            or record.get("early_close_deviation_ms") != "1218353"
        ):
            raise CsvDataError("invalid market interruption identity")
        event_ids.add(event.event_id)
        missing.update(event.missing_open_times)
    expected = {
        "market_interruption_event_count": len(event_ids),
        "market_interruption_candle_count": len(records),
        "contains_non_tradable_intervals": bool(records),
    }
    if any(summary.get(key) != value for key, value in expected.items()):
        raise CsvDataError("market interruption summary does not match records")
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise CsvDataError("metadata market interruption summary does not match report")
    return missing


def _required_nonnegative_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or value < 0:
        raise CsvDataError(f"invalid {key}")
    return value


def _milliseconds_string(microseconds: int) -> str:
    return format(Decimal(microseconds) / Decimal(1_000), "f")


def write_metadata_atomic(path: Path, metadata: dict[str, Any]) -> None:
    _atomic_write(path, json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write(path, json.dumps(payload, indent=2, allow_nan=False) + "\n")
