from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from trading_bot.data.csv_store import (
    SCHEMA_VERSION,
    csv_sha256,
    merge_candles,
    metadata_path,
    read_candles,
    write_candles_atomic,
    write_json_atomic,
    write_metadata_atomic,
)
from trading_bot.data.time_ranges import validate_hour_aligned_range
from trading_bot.domain.models import Candle

_INTERVAL = timedelta(hours=1)


class ClosedCandleClient(Protocol):
    """Minimal interface shared by public historical data sources."""

    now: Callable[[], datetime]

    async def fetch_closed(self, start_time: datetime, end_time: datetime) -> list[Candle]: ...


@dataclass(frozen=True, slots=True)
class DownloadSummary:
    old_candles: int
    new_candles: int
    total_candles: int
    first_candle_open: datetime | None
    last_candle_close: datetime | None
    requested_start: datetime
    requested_end: datetime
    effective_end: datetime
    requested_range_candle_count: int
    output: Path


class DataCoverageError(RuntimeError):
    """The exchange response cannot prove complete coverage of the requested range."""


async def download_vision_historical_csv(
    client: ClosedCandleClient,
    start: datetime,
    end: datetime,
    output: Path,
    overwrite: bool = False,
) -> DownloadSummary:
    """Publish CSV, audit sidecar and metadata as one verified Vision generation."""
    if not overwrite:
        raise ValueError(
            "Binance Vision publication requires --overwrite to avoid mixing unaudited generations"
        )
    start, end = validate_hour_aligned_range(start, end)
    effective_end = _effective_end(client, end)
    if effective_end <= start:
        raise ValueError("no closed candles exist in the requested range")
    downloaded = await client.fetch_closed(start, effective_end)
    merged = merge_candles(downloaded)
    _validate_requested_coverage(merged, start, effective_end)
    _publish_vision_generation(client, output, merged, start, end, effective_end)
    return _summary(merged, 0, start, end, effective_end, output)


def _publish_vision_generation(
    client: ClosedCandleClient,
    output: Path,
    candles: list[Candle],
    requested_start: datetime,
    requested_end: datetime,
    effective_end: datetime,
) -> None:
    """Stage every artifact, then use metadata as the final commit marker."""
    from trading_bot.data.binance_vision import anomaly_report

    parsed = getattr(client, "parsed", None)
    if not isinstance(parsed, list):
        raise TypeError("Binance Vision publisher requires archive audit records")
    verification_mode = getattr(client, "checksum_verification_mode", "official_online")
    if verification_mode not in {"official_online", "cached_offline"}:
        raise DataCoverageError("invalid checksum verification mode")
    report = anomaly_report(parsed, verification_mode)
    generation_id = uuid.uuid4().hex
    report["generation_id"] = generation_id
    report["requested_range"] = {"start": _iso(requested_start), "end": _iso(effective_end)}
    anomaly_path = output.with_suffix(".anomalies.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.generation.", dir=output.parent, ignore_cleanup_errors=True
    ) as directory:
        staging = Path(directory)
        staged_csv = staging / output.name
        staged_anomaly = staging / anomaly_path.name
        staged_metadata = staging / metadata_path(output).name
        write_candles_atomic(staged_csv, candles)
        write_json_atomic(staged_anomaly, report)
        summary = report["summary"]
        policy = report["policy"]
        if not isinstance(summary, dict) or not isinstance(policy, dict):
            raise DataCoverageError("invalid generated anomaly report")
        archive_candle_count = len(parsed)
        exact_archive_candle_count = sum(item.quality.value == "exact" for item in parsed)
        accepted_archive_anomaly_count = archive_candle_count - exact_archive_candle_count
        rest_suffix_candle_count = getattr(client, "rest_suffix_candle_count", None)
        if not isinstance(rest_suffix_candle_count, int) or rest_suffix_candle_count < 0:
            raise DataCoverageError("invalid REST suffix candle count")
        if archive_candle_count + rest_suffix_candle_count != len(candles):
            raise DataCoverageError("archive and REST counts do not match stored candle count")
        if (
            summary["archive_candle_count"] != archive_candle_count
            or summary["exact_archive_timestamp_candle_count"] != exact_archive_candle_count
            or summary["accepted_archive_anomaly_count"] != accepted_archive_anomaly_count
        ):
            raise DataCoverageError("generated anomaly report counts do not match archive records")
        metadata = {
            "generation_id": generation_id,
            "source": "Binance Vision verified archives plus Binance Spot public REST suffix",
            "market_type": "spot",
            "exchange_symbol": "BTCUSDT",
            "internal_symbol": "BTC/USDT",
            "timeframe": "1h",
            "requested_start": _iso(requested_start),
            "requested_end": _iso(requested_end),
            "effective_end": _iso(effective_end),
            "stored_first_candle_open": _iso(candles[0].open_time),
            "stored_last_candle_close": _iso(candles[-1].close_time),
            "stored_candle_count": len(candles),
            "archive_candle_count": archive_candle_count,
            "exact_archive_timestamp_candle_count": exact_archive_candle_count,
            "accepted_archive_anomaly_count": accepted_archive_anomaly_count,
            "rest_suffix_candle_count": rest_suffix_candle_count,
            "requested_range_candle_count": _requested_count(requested_start, effective_end),
            "missing_candle_count": 0,
            "duplicate_candle_count": 0,
            "conflicting_candle_count": 0,
            "source_archives": list(getattr(client, "archive_urls", [])),
            "monthly_archives": list(getattr(client, "monthly_archives", [])),
            "daily_archives": list(getattr(client, "daily_archives", [])),
            "archive_checksums": dict(getattr(client, "archive_checksums", {})),
            "checksum_verification_mode": verification_mode,
            "rest_suffix": _rest_suffix_metadata(getattr(client, "rest_suffix", None)),
            "timestamp_policy": policy,
            "timestamp_policy_version": policy["version"],
            "accepted_anomaly_count": summary["accepted_archive_anomaly_count"],
            "rejected_anomaly_count": summary["rejected_timestamp_anomalies"],
            "maximum_timestamp_deviation_us": summary["maximum_observed_early_close_us"],
            "maximum_timestamp_deviation_ms": summary["maximum_observed_early_close_ms"],
            "anomaly_report": anomaly_path.name,
            "anomaly_report_sha256": _sha256(staged_anomaly),
            "csv_sha256": csv_sha256(staged_csv),
            "downloaded_at": _iso(datetime.now(UTC)),
            "schema_version": SCHEMA_VERSION,
        }
        write_metadata_atomic(staged_metadata, metadata)
        _verify_staged_generation(staged_csv, staged_anomaly, staged_metadata)
        targets = (output, anomaly_path, metadata_path(output))
        staged = (staged_csv, staged_anomaly, staged_metadata)
        backups = tuple(staging / f"{target.name}.previous" for target in targets)
        for target, backup in zip(targets, backups, strict=True):
            if target.exists():
                shutil.copy2(target, backup)
        try:
            # Metadata is the commit marker.  Before it is replaced, a reader
            # rejects staged payloads by checksum.  A failed replacement is
            # rolled back to the prior complete generation before returning.
            for source, target in zip(staged, targets, strict=True):
                _replace_generation_file(source, target)
        except OSError:
            for target, backup in zip(targets, backups, strict=True):
                if backup.exists():
                    _replace_generation_file(backup, target)
                elif target.exists():
                    target.unlink()
            raise


def _rest_suffix_metadata(value: object) -> dict[str, str] | None:
    if not isinstance(value, tuple) or len(value) != 2:
        return None
    start, end = value
    if not isinstance(start, datetime) or not isinstance(end, datetime):
        return None
    return {"start": _iso(start), "end": _iso(end)}


def _verify_staged_generation(csv_path: Path, anomaly_path: Path, metadata_file: Path) -> None:
    try:
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        report = json.loads(anomaly_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataCoverageError("could not verify staged generation") from exc
    if metadata.get("csv_sha256") != csv_sha256(csv_path):
        raise DataCoverageError("staged CSV checksum mismatch")
    if metadata.get("anomaly_report_sha256") != _sha256(anomaly_path):
        raise DataCoverageError("staged anomaly checksum mismatch")
    if metadata.get("generation_id") != report.get("generation_id"):
        raise DataCoverageError("staged generation identifiers do not match")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _replace_generation_file(source: Path, target: Path) -> None:
    os.replace(source, target)


async def download_historical_csv(
    client: ClosedCandleClient,
    start: datetime,
    end: datetime,
    output: Path,
    overwrite: bool = False,
) -> DownloadSummary:
    """Download closed 1h candles and atomically maintain one normalized CSV."""
    start, end = validate_hour_aligned_range(start, end)
    effective_end = _effective_end(client, end)
    if effective_end <= start:
        raise ValueError("no closed candles exist in the requested range")
    existing: list[Candle] = []
    if output.exists() and not overwrite:
        existing = read_candles(output)
    requests = _missing_ranges(existing, start, effective_end, overwrite)
    downloaded: list[Candle] = []
    for request_start, request_end in requests:
        downloaded.extend(await client.fetch_closed(request_start, request_end))
    merged = merge_candles(downloaded) if overwrite else merge_candles(existing, downloaded)
    _validate_requested_coverage(merged, start, effective_end)
    # Do not replace a valid existing file until the complete merged data is valid.
    persisted = write_candles_atomic(output, merged) if requests or overwrite else existing
    write_metadata_atomic(
        metadata_path(output),
        {
            "source": "Binance Spot public klines",
            "market_type": "spot",
            "exchange_symbol": "BTCUSDT",
            "internal_symbol": "BTC/USDT",
            "timeframe": "1h",
            "requested_start": _iso(start),
            "requested_end": _iso(end),
            "effective_end": _iso(effective_end),
            "stored_first_candle_open": _iso(persisted[0].open_time),
            "stored_last_candle_close": _iso(persisted[-1].close_time),
            "stored_candle_count": len(persisted),
            "requested_range_candle_count": _requested_count(start, effective_end),
            "csv_sha256": csv_sha256(output),
            "downloaded_at": _iso(datetime.now(UTC)),
            "schema_version": SCHEMA_VERSION,
        },
    )
    return _summary(persisted, 0 if overwrite else len(existing), start, end, effective_end, output)


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _effective_end(client: ClosedCandleClient, end: datetime) -> datetime:
    now = getattr(client, "now", lambda: end)().astimezone(UTC)
    closed_boundary = now.replace(minute=0, second=0, microsecond=0)
    return min(end.astimezone(UTC), closed_boundary)


def _missing_ranges(
    existing: list[Candle], start: datetime, effective_end: datetime, overwrite: bool
) -> list[tuple[datetime, datetime]]:
    if overwrite or not existing:
        return [(start, effective_end)] if effective_end > start else []
    existing_start, existing_end = existing[0].open_time, existing[-1].close_time
    ranges: list[tuple[datetime, datetime]] = []
    if existing_start > start:
        ranges.append((start, min(existing_start, effective_end)))
    if existing_end < effective_end:
        ranges.append((max(existing_end, start), effective_end))
    return [
        (range_start, range_end) for range_start, range_end in ranges if range_start < range_end
    ]


def _requested_count(start: datetime, effective_end: datetime) -> int:
    return int((effective_end - start) / _INTERVAL)


def _validate_requested_coverage(
    candles: list[Candle], start: datetime, effective_end: datetime
) -> None:
    if effective_end == start:
        return
    requested = [candle for candle in candles if start <= candle.open_time < effective_end]
    expected = _requested_count(start, effective_end)
    if (
        len(requested) != expected
        or not requested
        or requested[0].open_time != start
        or requested[-1].close_time != effective_end
    ):
        raise DataCoverageError("incomplete requested range; use --overwrite or retry the download")


def _summary(
    candles: list[Candle],
    old_candles: int,
    requested_start: datetime,
    requested_end: datetime,
    effective_end: datetime,
    output: Path,
) -> DownloadSummary:
    return DownloadSummary(
        old_candles=old_candles,
        new_candles=len(candles) - old_candles,
        total_candles=len(candles),
        first_candle_open=candles[0].open_time if candles else None,
        last_candle_close=candles[-1].close_time if candles else None,
        requested_start=requested_start,
        requested_end=requested_end,
        effective_end=effective_end,
        requested_range_candle_count=_requested_count(requested_start, effective_end),
        output=output,
    )


def summary_json(summary: DownloadSummary) -> dict[str, Any]:
    values = asdict(summary)
    return {
        "old_candles": values["old_candles"],
        "new_candles": values["new_candles"],
        "total_candles": values["total_candles"],
        "requested_start": _iso(values["requested_start"]),
        "requested_end": _iso(values["requested_end"]),
        "effective_end": _iso(values["effective_end"]),
        "requested_range_candle_count": values["requested_range_candle_count"],
        "first_candle_open": _iso(values["first_candle_open"])
        if values["first_candle_open"] is not None
        else None,
        "last_candle_close": _iso(values["last_candle_close"])
        if values["last_candle_close"] is not None
        else None,
        "output": str(values["output"]),
    }
