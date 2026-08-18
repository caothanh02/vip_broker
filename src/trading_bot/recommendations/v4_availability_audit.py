"""Mechanical, public-only archive availability audit for Protocol V4.

This module can establish only whether a *proposed* historical range is
available from Binance Vision.  It never persists candle data, computes a
feature, signal, return, or metric, and cannot select or execute a candidate.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
import zipfile
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import httpx

from trading_bot.data.binance_vision import (
    ArchiveTimestampPolicyError,
    ArchiveTimestampQuality,
    parse_verified_archive_kline,
)
from trading_bot.recommendations.protocol_v4 import (
    ProtocolV4,
    ProtocolV4Error,
    load_protocol_v4,
    require_protocol_v4_availability_audit,
)

_BINANCE_VISION = "https://data.binance.vision/data/spot"
_SYMBOL = "BTCUSDT"
_TIMEFRAME = "1h"
_MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ProtocolV4AvailabilityAuditError(ValueError):
    """A requested V4 availability audit is outside the fail-closed contract."""


@dataclass(frozen=True, slots=True)
class ArchiveAvailability:
    archive_name: str
    sha256: str | None
    expected_candle_count: int
    observed_candle_count: int | None
    accepted_timestamp_anomaly_count: int | None
    missing_open_times: tuple[str, ...]
    duplicate_open_times: tuple[str, ...]
    unexpected_open_times: tuple[str, ...]
    closed_candles_only: bool
    error: str | None

    @property
    def available(self) -> bool:
        return (
            self.error is None
            and self.sha256 is not None
            and self.observed_candle_count == self.expected_candle_count
            and not self.missing_open_times
            and not self.duplicate_open_times
            and not self.unexpected_open_times
            and self.closed_candles_only
        )


def _utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _month_start(value: datetime) -> datetime:
    return value.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def _next_month(value: datetime) -> datetime:
    if value.month == 12:
        return value.replace(year=value.year + 1, month=1)
    return value.replace(month=value.month + 1)


def _expected_opens(start: datetime, end: datetime) -> tuple[datetime, ...]:
    result: list[datetime] = []
    cursor = start
    while cursor < end:
        result.append(cursor)
        cursor += timedelta(hours=1)
    return tuple(result)


def _require_month_range(start: datetime, end: datetime, protocol: ProtocolV4) -> None:
    for value, label in ((start, "start"), (end, "end")):
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ProtocolV4AvailabilityAuditError(f"{label} must be explicitly UTC")
        if value != _month_start(value):
            raise ProtocolV4AvailabilityAuditError(f"{label} must be a UTC month boundary")
    if start >= end:
        raise ProtocolV4AvailabilityAuditError("start must be before end")
    if end > protocol.strict_oos_start:
        raise ProtocolV4AvailabilityAuditError("availability audit must not read strict OOS")


def _checksum(text: str, archive_name: str) -> str:
    lines = text.splitlines()
    if len(lines) != 1:
        raise ProtocolV4AvailabilityAuditError("official checksum sidecar is invalid")
    match = re.fullmatch(r"([0-9a-fA-F]{64})  \*?([^/\\]+)", lines[0])
    if match is None or match.group(2) != archive_name:
        raise ProtocolV4AvailabilityAuditError("official checksum sidecar identity is invalid")
    value = match.group(1).lower()
    if _SHA256.fullmatch(value) is None:
        raise ProtocolV4AvailabilityAuditError("official checksum is invalid")
    return value


def _archive_opens(
    data: bytes, archive_name: str, checksum: str
) -> tuple[tuple[datetime, ...], int]:
    if len(data) > _MAX_ARCHIVE_BYTES:
        raise ProtocolV4AvailabilityAuditError("archive is too large")
    expected_member = archive_name.removesuffix(".zip") + ".csv"
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = archive.namelist()
            if names != [expected_member]:
                raise ProtocolV4AvailabilityAuditError("archive member identity is invalid")
            info = archive.getinfo(expected_member)
            if info.file_size > _MAX_ARCHIVE_BYTES:
                raise ProtocolV4AvailabilityAuditError("archive CSV is too large")
            rows = list(csv.reader(io.StringIO(archive.read(info).decode("utf-8-sig"))))
    except (OSError, UnicodeDecodeError, zipfile.BadZipFile) as exc:
        raise ProtocolV4AvailabilityAuditError("archive is invalid") from exc
    if rows and rows[0] and rows[0][0].lower() in {"open time", "open_time"}:
        rows = rows[1:]

    opens: list[datetime] = []
    accepted_anomalies = 0
    for index, row in enumerate(rows):
        try:
            parsed = parse_verified_archive_kline(
                row,
                archive_name=archive_name,
                archive_sha256=checksum,
                row_number=index,
                checksum_verified=True,
                interruptions=(),
            )
        except ArchiveTimestampPolicyError as exc:
            raise ProtocolV4AvailabilityAuditError(
                f"archive row {index} violates timestamp policy: {exc}"
            ) from exc
        opens.append(parsed.candle.open_time)
        if parsed.quality is not ArchiveTimestampQuality.EXACT:
            accepted_anomalies += 1
    return tuple(opens), accepted_anomalies


def _archive_result(
    archive_name: str, data: bytes, checksum: str, start: datetime, end: datetime
) -> ArchiveAvailability:
    actual_checksum = hashlib.sha256(data).hexdigest()
    if actual_checksum != checksum:
        raise ProtocolV4AvailabilityAuditError("archive checksum mismatch")
    opens, accepted_anomalies = _archive_opens(data, archive_name, checksum)
    expected = _expected_opens(start, end)
    counts = Counter(opens)
    actual = set(opens)
    expected_set = set(expected)
    duplicate = tuple(_utc(value) for value, count in sorted(counts.items()) if count > 1)
    missing = tuple(_utc(value) for value in expected if value not in actual)
    unexpected = tuple(_utc(value) for value in sorted(actual - expected_set))
    return ArchiveAvailability(
        archive_name=archive_name,
        sha256=checksum,
        expected_candle_count=len(expected),
        observed_candle_count=len(opens),
        accepted_timestamp_anomaly_count=accepted_anomalies,
        missing_open_times=missing,
        duplicate_open_times=duplicate,
        unexpected_open_times=unexpected,
        closed_candles_only=True,
        error=None,
    )


def _error_result(
    archive_name: str,
    start: datetime,
    end: datetime,
    error: ValueError,
    checksum: str | None,
) -> ArchiveAvailability:
    return ArchiveAvailability(
        archive_name=archive_name,
        sha256=checksum,
        expected_candle_count=len(_expected_opens(start, end)),
        observed_candle_count=None,
        accepted_timestamp_anomaly_count=None,
        missing_open_times=(),
        duplicate_open_times=(),
        unexpected_open_times=(),
        closed_candles_only=False,
        error=str(error),
    )


async def _fetch(client: httpx.AsyncClient, url: str) -> httpx.Response:
    try:
        response = await client.get(url)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise ProtocolV4AvailabilityAuditError("public archive request failed") from exc
    return response


async def _audit_month(
    client: httpx.AsyncClient, base_url: str, start: datetime
) -> ArchiveAvailability:
    end = _next_month(start)
    stamp = start.strftime("%Y-%m")
    archive_name = f"{_SYMBOL}-{_TIMEFRAME}-{stamp}.zip"
    relative = f"monthly/klines/{_SYMBOL}/{_TIMEFRAME}/{archive_name}"
    checksum: str | None = None
    try:
        checksum_response = await _fetch(client, f"{base_url}/{relative}.CHECKSUM")
        checksum = _checksum(checksum_response.text, archive_name)
        archive_response = await _fetch(client, f"{base_url}/{relative}")
        return _archive_result(archive_name, archive_response.content, checksum, start, end)
    except ProtocolV4AvailabilityAuditError as exc:
        return _error_result(archive_name, start, end, exc, checksum)


def _archive_json(value: ArchiveAvailability) -> dict[str, object]:
    return {
        "archive_name": value.archive_name,
        "checksum_verified": value.sha256 is not None,
        "archive_sha256": value.sha256,
        "expected_candle_count": value.expected_candle_count,
        "observed_candle_count": value.observed_candle_count,
        "accepted_timestamp_anomaly_count": value.accepted_timestamp_anomaly_count,
        "missing_open_times": list(value.missing_open_times),
        "duplicate_open_times": list(value.duplicate_open_times),
        "unexpected_open_times": list(value.unexpected_open_times),
        "closed_candles_only": value.closed_candles_only,
        "available": value.available,
        "error": value.error,
    }


async def audit_protocol_v4_availability(
    start: datetime,
    end: datetime,
    *,
    base_url: str = _BINANCE_VISION,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, object]:
    """Audit complete monthly archives without saving or evaluating market data."""

    protocol = load_protocol_v4()
    try:
        require_protocol_v4_availability_audit(protocol)
    except ProtocolV4Error as exc:
        raise ProtocolV4AvailabilityAuditError(str(exc)) from exc
    _require_month_range(start, end, protocol)
    base = base_url.rstrip("/")
    results: list[ArchiveAvailability] = []
    async with httpx.AsyncClient(timeout=30, transport=transport) as client:
        cursor = start
        while cursor < end:
            results.append(await _audit_month(client, base, cursor))
            cursor = _next_month(cursor)

    continuity = all(item.available for item in results)
    return {
        "schema_version": "1.0",
        "protocol_id": protocol.protocol_id,
        "protocol_status": protocol.status,
        "audit_kind": "mechanical_public_archive_availability_only",
        "symbol": "BTC/USDT",
        "timeframe": _TIMEFRAME,
        "utc_range": {"start": _utc(start), "end": _utc(end)},
        "archives": [_archive_json(item) for item in results],
        "official_checksums_verified": all(item.sha256 is not None for item in results),
        "absolute_continuity": continuity,
        "result": "availability_verified_not_selected"
        if continuity
        else "availability_not_verified",
        "selection_authorized": False,
        "execution_authorized": False,
        "strict_oos_authorized": False,
        "safety_locks": {
            "default_recommendation": "NEUTRAL",
            "recommendation_engine_used": False,
            "signal_or_feature_computed": False,
            "performance_metric_computed": False,
            "data_persisted": False,
            "broker_used": False,
            "orders_submitted": False,
            "ml_used": False,
            "authenticated_api_used": False,
            "strict_oos_read": False,
            "network_used": True,
        },
    }


def assert_no_performance_inputs(values: Iterable[str]) -> None:
    """Reject accidental extension of an audit selector with research metrics."""

    forbidden = {"signal", "return", "accuracy", "backtest", "performance_metric"}
    if forbidden.intersection(values):
        raise ProtocolV4AvailabilityAuditError("performance inputs are prohibited")
