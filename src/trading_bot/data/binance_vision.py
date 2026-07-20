from __future__ import annotations

import csv
import hashlib
import io
import os
import re
import tempfile
import zipfile
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path
from typing import Literal

import httpx

from trading_bot.data.binance_historical import BinanceHistoricalDataClient
from trading_bot.data.csv_store import merge_candles
from trading_bot.data.time_ranges import validate_hour_aligned_range
from trading_bot.domain.models import Candle

POLICY_NAME = "verified-binance-vision-early-close"
POLICY_VERSION = "1.0"
MAX_EARLY_CLOSE_MS = 60_000
MAX_EARLY_CLOSE_US = 60_000_000
BINANCE_VISION = "https://data.binance.vision/data/spot"
MICROSECOND_ARCHIVE_START = datetime(2025, 1, 1, tzinfo=UTC)
_MONTHLY_ARCHIVE = re.compile(r"BTCUSDT-1h-(\d{4})-(\d{2})\.zip\Z")
_DAILY_ARCHIVE = re.compile(r"BTCUSDT-1h-(\d{4})-(\d{2})-(\d{2})\.zip\Z")


class ArchiveTimestampQuality(StrEnum):
    EXACT = "exact"
    EARLY_CLOSE_WITHIN_TOLERANCE = "early_close_within_tolerance"
    INVALID_EARLY_CLOSE = "invalid_early_close"
    LATE_CLOSE = "late_close"
    INVALID_OPEN_ALIGNMENT = "invalid_open_alignment"


class ArchiveTimestampPolicyError(ValueError):
    """A verified Binance Vision timestamp violates the fixed archive policy."""


class BinanceVisionError(ValueError):
    """A Binance Vision archive cannot prove a complete, verified 1h range."""


@dataclass(frozen=True, slots=True)
class ParsedArchiveCandle:
    candle: Candle
    raw_open_timestamp: int
    raw_close_timestamp: int
    expected_close_timestamp: int
    timestamp_unit: Literal["milliseconds", "microseconds"]
    early_close_deviation_us: int
    quality: ArchiveTimestampQuality
    archive_name: str
    archive_sha256: str
    row_number: int
    adjacent_continuity_verified: bool = False


@dataclass(frozen=True, slots=True)
class ArchivePeriod:
    cadence: Literal["monthly", "daily"]
    start: datetime
    end: datetime


def parse_verified_archive_kline(
    row: Sequence[str],
    *,
    archive_name: str,
    archive_sha256: str,
    row_number: int,
    checksum_verified: bool,
) -> ParsedArchiveCandle:
    """Parse one official, checksum-verified BTCUSDT Spot 1h archive row.

    The only relaxation is an early raw close of at most 60 seconds. Domain
    candle close time is always canonicalized to the next UTC hour.
    """
    if not checksum_verified:
        raise ArchiveTimestampPolicyError("relaxed archive policy requires verified checksum")
    if len(row) < 7:
        raise ArchiveTimestampPolicyError("archive row has fewer than seven columns")
    open_timestamp = _integer(row[0], "open")
    close_timestamp = _integer(row[6], "close")
    period = parse_archive_period(archive_name)
    unit = detect_archive_timestamp_unit(open_timestamp)
    close_unit = detect_archive_timestamp_unit(close_timestamp)
    if close_unit != unit:
        raise ArchiveTimestampPolicyError("mixed open and close timestamp units")
    open_time = datetime_from_archive_timestamp(open_timestamp, unit)
    validate_timestamp_unit_for_date(open_time, unit)
    if not period.start <= open_time < period.end:
        raise ArchiveTimestampPolicyError(
            f"row open time is outside named {period.cadence} archive period"
        )
    expected, quality, deviation_us = _timestamp_policy(open_timestamp, close_timestamp, unit)
    try:
        open_, high, low, close, volume = (Decimal(row[index]) for index in range(1, 6))
    except (InvalidOperation, ValueError) as exc:
        raise ArchiveTimestampPolicyError("invalid archive OHLCV decimal") from exc
    if not all(value.is_finite() for value in (open_, high, low, close, volume)):
        raise ArchiveTimestampPolicyError("non-finite archive OHLCV")
    if min(open_, high, low, close) <= 0 or volume < 0:
        raise ArchiveTimestampPolicyError("invalid archive OHLCV")
    if high < max(open_, low, close) or low > min(open_, high, close):
        raise ArchiveTimestampPolicyError("invalid archive OHLC")
    return ParsedArchiveCandle(
        Candle(
            open_time,
            open_time + timedelta(hours=1),
            "BTC/USDT",
            "1h",
            open_,
            high,
            low,
            close,
            volume,
            True,
        ),
        open_timestamp,
        close_timestamp,
        expected,
        unit,
        deviation_us,
        quality,
        archive_name,
        archive_sha256,
        row_number,
    )


def detect_archive_timestamp_unit(raw: int) -> Literal["milliseconds", "microseconds"]:
    """Detect only the raw representation; date policy is enforced separately."""
    return "microseconds" if raw >= 1_000_000_000_000_000 else "milliseconds"


def datetime_from_archive_timestamp(
    raw: int, unit: Literal["milliseconds", "microseconds"]
) -> datetime:
    scale = 1_000 if unit == "milliseconds" else 1_000_000
    seconds, remainder = divmod(raw, scale)
    return datetime.fromtimestamp(seconds, UTC) + timedelta(
        microseconds=remainder * (1_000_000 // scale)
    )


def validate_timestamp_unit_for_date(
    open_time: datetime, unit: Literal["milliseconds", "microseconds"]
) -> None:
    required = "milliseconds" if open_time < MICROSECOND_ARCHIVE_START else "microseconds"
    if unit != required:
        raise ArchiveTimestampPolicyError(
            f"{required} required for archive open time {open_time.isoformat()}"
        )


def _timestamp_policy(
    open_timestamp: int, close_timestamp: int, unit: Literal["milliseconds", "microseconds"]
) -> tuple[int, ArchiveTimestampQuality, int]:
    scale, tolerance = (
        (1_000_000, MAX_EARLY_CLOSE_US) if unit == "microseconds" else (1_000, MAX_EARLY_CLOSE_MS)
    )
    if open_timestamp % (3_600 * scale):
        raise ArchiveTimestampPolicyError("invalid_open_alignment")
    expected = open_timestamp + 3_600 * scale - 1
    if close_timestamp > expected:
        raise ArchiveTimestampPolicyError(
            _message(ArchiveTimestampQuality.LATE_CLOSE, expected, close_timestamp)
        )
    early = expected - close_timestamp
    if early > tolerance:
        raise ArchiveTimestampPolicyError(
            _message(ArchiveTimestampQuality.INVALID_EARLY_CLOSE, expected, close_timestamp)
        )
    quality = (
        ArchiveTimestampQuality.EXACT
        if early == 0
        else ArchiveTimestampQuality.EARLY_CLOSE_WITHIN_TOLERANCE
    )
    return expected, quality, early * (1_000 if unit == "milliseconds" else 1)


def parse_archive_period(archive_name: str) -> ArchivePeriod:
    """Parse the exact official filename and derive its UTC half-open period."""
    monthly = _MONTHLY_ARCHIVE.fullmatch(archive_name)
    daily = _DAILY_ARCHIVE.fullmatch(archive_name)
    try:
        if monthly:
            year, month = (int(value) for value in monthly.groups())
            start = datetime(year, month, 1, tzinfo=UTC)
            return ArchivePeriod("monthly", start, _next_month(start))
        if daily:
            year, month, day = (int(value) for value in daily.groups())
            start = datetime(year, month, day, tzinfo=UTC)
            return ArchivePeriod("daily", start, start + timedelta(days=1))
    except ValueError as exc:
        raise ArchiveTimestampPolicyError(f"invalid archive filename: {archive_name}") from exc
    raise ArchiveTimestampPolicyError(f"invalid archive filename: {archive_name}")


def _integer(value: str, field: str) -> int:
    if not isinstance(value, str):
        raise ArchiveTimestampPolicyError(f"invalid {field} timestamp")
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ArchiveTimestampPolicyError(f"invalid {field} timestamp") from exc
    if parsed < 0 or str(parsed) != value:
        raise ArchiveTimestampPolicyError(f"invalid {field} timestamp")
    return parsed


def _message(quality: ArchiveTimestampQuality, expected: int, actual: int) -> str:
    return f"{quality.value}: raw_close={actual} expected_close={expected} policy={POLICY_VERSION}"


def anomaly_report(
    parsed: Sequence[ParsedArchiveCandle], checksum_verification_mode: str = "official_online"
) -> dict[str, object]:
    """Create a deterministic, path-free audit document for accepted archive anomalies."""
    anomalies = [item for item in parsed if item.quality is not ArchiveTimestampQuality.EXACT]
    if any(not item.adjacent_continuity_verified for item in anomalies):
        raise BinanceVisionError("anomaly continuity has not been verified")
    records = [
        {
            "archive": item.archive_name,
            "archive_sha256": item.archive_sha256,
            "row_number": item.row_number,
            "open_time": item.candle.open_time.isoformat().replace("+00:00", "Z"),
            "raw_open_timestamp": item.raw_open_timestamp,
            "raw_close_timestamp": item.raw_close_timestamp,
            "expected_close_timestamp": item.expected_close_timestamp,
            "timestamp_unit": item.timestamp_unit,
            "early_close_deviation_us": item.early_close_deviation_us,
            "early_close_deviation_ms": _milliseconds_string(item.early_close_deviation_us),
            "quality": item.quality.value,
            "accepted": True,
            "adjacent_continuity_verified": True,
            "policy_version": POLICY_VERSION,
        }
        for item in anomalies
    ]
    return {
        "policy": {
            "name": POLICY_NAME,
            "version": POLICY_VERSION,
            "maximum_early_close_ms": MAX_EARLY_CLOSE_MS,
            "maximum_early_close_us": MAX_EARLY_CLOSE_US,
            "late_close_allowed": False,
            "rest_policy": "strict",
            "checksum_verification_mode": checksum_verification_mode,
        },
        "summary": {
            "archive_candle_count": len(parsed),
            "exact_archive_timestamp_candle_count": len(parsed) - len(records),
            "accepted_archive_anomaly_count": len(records),
            "rejected_timestamp_anomalies": 0,
            "maximum_observed_early_close_us": max(
                (item.early_close_deviation_us for item in anomalies), default=0
            ),
            "maximum_observed_early_close_ms": _milliseconds_string(
                max((item.early_close_deviation_us for item in anomalies), default=0)
            ),
        },
        "anomalies": records,
    }


def _milliseconds_string(microseconds: int) -> str:
    return format(Decimal(microseconds) / Decimal(1_000), "f")


class BinanceVisionHistoricalClient:
    """Verified Data Vision archives with strict Binance REST suffix only."""

    def __init__(
        self,
        cache_dir: Path,
        rest_client: BinanceHistoricalDataClient,
        base_url: str = BINANCE_VISION,
        transport: httpx.AsyncBaseTransport | None = None,
        now: Callable[[], datetime] | None = None,
        max_retries: int = 3,
        backoff_seconds: float = 0.25,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        offline_cache: bool = False,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self.cache_dir = cache_dir
        self.rest_client = rest_client
        self.base_url = base_url.rstrip("/")
        self.transport = transport
        self.now = now or (lambda: datetime.now(UTC))
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.sleep = sleep
        self.offline_cache = offline_cache
        self.checksum_verification_mode = "cached_offline" if offline_cache else "official_online"
        self.parsed: list[ParsedArchiveCandle] = []
        self.archive_urls: list[str] = []
        self.archive_checksums: dict[str, str] = {}
        self.monthly_archives: list[str] = []
        self.daily_archives: list[str] = []
        self.rest_suffix: tuple[datetime, datetime] | None = None
        self.rest_suffix_candle_count = 0

    async def fetch_closed(self, start_time: datetime, end_time: datetime) -> list[Candle]:
        start, end = validate_hour_aligned_range(start_time, end_time)
        closed_end = min(end, self.now().astimezone(UTC).replace(minute=0, second=0, microsecond=0))
        self.parsed = []
        self.archive_urls = []
        self.archive_checksums = {}
        self.monthly_archives = []
        self.daily_archives = []
        self.rest_suffix = None
        self.rest_suffix_candle_count = 0
        archive_end = _month_start(closed_end)
        candles: list[Candle] = []
        cursor = _month_start(start)
        while cursor < archive_end:
            try:
                candles.extend(await self._load_archive("monthly", cursor, allow_missing=True))
            except FileNotFoundError:
                candles.extend(
                    await self._load_daily_range(cursor, _next_month(cursor), strict=True)
                )
            cursor = _next_month(cursor)
        if archive_end < closed_end:
            daily_end = closed_end.replace(hour=0)
            candles.extend(await self._load_daily_range(archive_end, daily_end, strict=False))
            suffix_start = _last_close(candles) or start
            if suffix_start < closed_end:
                rest_candles = await self.rest_client.fetch_closed(suffix_start, closed_end)
                expected_rest_count = int((closed_end - suffix_start) / timedelta(hours=1))
                if len(rest_candles) != expected_rest_count:
                    raise BinanceVisionError(
                        "REST suffix does not provide complete hourly coverage"
                    )
                candles.extend(rest_candles)
                self.rest_suffix = (suffix_start, closed_end)
                self.rest_suffix_candle_count = len(rest_candles)
        try:
            merged = merge_candles(candles)
        except ValueError as exc:
            raise BinanceVisionError(str(exc)) from exc
        verified = _verify_anomaly_continuity(self.parsed, merged)
        result = [candle for candle in merged if start <= candle.open_time < closed_end]
        _validate_continuity(result, start, closed_end)
        self.parsed = [item for item in verified if start <= item.candle.open_time < closed_end]
        if len(self.parsed) + self.rest_suffix_candle_count != len(result):
            raise BinanceVisionError("archive and REST source counts do not match stored coverage")
        return result

    async def _load_daily_range(
        self, start: datetime, end: datetime, *, strict: bool
    ) -> list[Candle]:
        candles: list[Candle] = []
        day = start
        while day < end:
            try:
                candles.extend(await self._load_archive("daily", day, allow_missing=True))
            except FileNotFoundError as exc:
                if strict:
                    raise BinanceVisionError(
                        f"missing required daily archive: {day.date()}"
                    ) from exc
                break
            day += timedelta(days=1)
        return candles

    async def _load_archive(
        self, cadence: Literal["monthly", "daily"], value: datetime, *, allow_missing: bool
    ) -> list[Candle]:
        stamp = value.strftime("%Y-%m" if cadence == "monthly" else "%Y-%m-%d")
        name = f"BTCUSDT-1h-{stamp}.zip"
        relative = f"{cadence}/klines/BTCUSDT/1h/{name}"
        archive, checksum = await self._verified_bytes(relative, allow_missing)
        parsed = _parse_archive(archive, name, checksum)
        self.parsed.extend(parsed)
        self.archive_urls.append(f"{self.base_url}/{relative}")
        self.archive_checksums[name] = checksum
        (self.monthly_archives if cadence == "monthly" else self.daily_archives).append(name)
        return [item.candle for item in parsed]

    async def _verified_bytes(self, relative: str, allow_missing: bool) -> tuple[bytes, str]:
        zip_path = self.cache_dir / relative
        checksum_path = zip_path.with_suffix(".zip.CHECKSUM")
        archive_name = Path(relative).name
        if self.offline_cache:
            offline_cached = _read_verified_cache(zip_path, checksum_path, archive_name)
            if offline_cached is None:
                raise BinanceVisionError(f"offline cache unavailable or invalid: {archive_name}")
            return offline_cached
        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=30, transport=self.transport) as client:
                    checksum_response = await client.get(f"{self.base_url}/{relative}.CHECKSUM")
                    if checksum_response.status_code == 404 and allow_missing:
                        raise FileNotFoundError(relative)
                    if checksum_response.status_code >= 500:
                        raise httpx.HTTPStatusError(
                            "server error",
                            request=checksum_response.request,
                            response=checksum_response,
                        )
                    checksum_response.raise_for_status()
            except FileNotFoundError:
                raise
            except httpx.HTTPError as exc:
                if attempt == self.max_retries:
                    raise BinanceVisionError(
                        f"archive request failed after retries: {relative}"
                    ) from exc
                await self._backoff(attempt)
                continue
            checksum = _checksum_from_text(checksum_response.text, archive_name)
            cached = _read_cached_zip(zip_path, checksum)
            if cached is not None:
                _write_cache_file(checksum_path, f"{checksum}  {archive_name}\n".encode())
                return cached, checksum
            try:
                async with httpx.AsyncClient(timeout=30, transport=self.transport) as client:
                    data_response = await client.get(f"{self.base_url}/{relative}")
                    if data_response.status_code == 404 and allow_missing:
                        raise FileNotFoundError(relative)
                    if data_response.status_code >= 500:
                        raise httpx.HTTPStatusError(
                            "server error", request=data_response.request, response=data_response
                        )
                    data_response.raise_for_status()
            except FileNotFoundError:
                raise
            except httpx.HTTPError as exc:
                if attempt == self.max_retries:
                    raise BinanceVisionError(
                        f"archive request failed after retries: {relative}"
                    ) from exc
                await self._backoff(attempt)
                continue
            data = data_response.content
            if hashlib.sha256(data).hexdigest() != checksum:
                raise BinanceVisionError(f"checksum mismatch: {archive_name}")
            _write_verified_cache(zip_path, checksum_path, data, checksum, archive_name)
            return data, checksum
        raise AssertionError("unreachable")

    async def _backoff(self, attempt: int) -> None:
        delay = self.backoff_seconds * (2**attempt)
        if self.sleep is not None:
            await self.sleep(delay)
            return
        import asyncio

        await asyncio.sleep(delay)


def _parse_archive(data: bytes, name: str, checksum: str) -> list[ParsedArchiveCandle]:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = archive.namelist()
            expected_csv_name = Path(name).with_suffix(".csv").name
            if len(names) != 1 or names[0] != expected_csv_name:
                raise BinanceVisionError(f"invalid inner CSV identity: {name}")
            info = archive.getinfo(names[0])
            if info.file_size > 256 * 1024 * 1024 or info.compress_size > 64 * 1024 * 1024:
                raise BinanceVisionError("archive too large")
            with archive.open(info) as raw:
                rows = list(csv.reader(io.TextIOWrapper(raw, encoding="utf-8", newline="")))
    except zipfile.BadZipFile as exc:
        raise BinanceVisionError(f"invalid archive: {name}") from exc
    if rows and rows[0] and rows[0][0].lower() in {"open time", "open_time"}:
        rows = rows[1:]
    parsed: list[ParsedArchiveCandle] = []
    for index, row in enumerate(rows):
        try:
            parsed.append(
                parse_verified_archive_kline(
                    row,
                    archive_name=name,
                    archive_sha256=checksum,
                    row_number=index,
                    checksum_verified=True,
                )
            )
        except ArchiveTimestampPolicyError as exc:
            raw_open = row[0] if row else "<missing>"
            raise BinanceVisionError(f"{name} row {index} raw_open={raw_open}: {exc}") from exc
    _validate_archive_coverage(parsed, parse_archive_period(name))
    return parsed


def _validate_archive_coverage(
    parsed: Sequence[ParsedArchiveCandle], period: ArchivePeriod
) -> None:
    expected_count = int((period.end - period.start) / timedelta(hours=1))
    if len(parsed) != expected_count:
        raise BinanceVisionError(
            f"{period.cadence} archive has {len(parsed)} rows; expected {expected_count}"
        )
    expected = period.start
    for item in parsed:
        if item.candle.open_time != expected:
            raise BinanceVisionError(
                f"{period.cadence} archive coverage error at {expected.isoformat()}"
            )
        expected += timedelta(hours=1)
    if expected != period.end:
        raise BinanceVisionError(f"{period.cadence} archive does not end at its named period")


def _month_start(value: datetime) -> datetime:
    return value.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def _next_month(value: datetime) -> datetime:
    return (
        value.replace(year=value.year + 1, month=1)
        if value.month == 12
        else value.replace(month=value.month + 1)
    )


def _validate_continuity(candles: Sequence[Candle], start: datetime, end: datetime) -> None:
    expected = start
    for candle in candles:
        if candle.open_time != expected:
            raise BinanceVisionError(f"missing candle at {expected.isoformat()}")
        expected = candle.close_time
    if expected != end:
        raise BinanceVisionError(f"missing candle at {expected.isoformat()}")


def _verify_anomaly_continuity(
    parsed: Sequence[ParsedArchiveCandle], merged: Sequence[Candle]
) -> list[ParsedArchiveCandle]:
    """Mark anomalies only after both global chronological neighbours exist."""
    available = {candle.open_time for candle in merged}
    verified: list[ParsedArchiveCandle] = []
    for item in parsed:
        if item.quality is ArchiveTimestampQuality.EXACT:
            verified.append(item)
            continue
        open_time = item.candle.open_time
        previous = open_time - timedelta(hours=1)
        following = open_time + timedelta(hours=1)
        if previous not in available or following not in available:
            raise BinanceVisionError(
                f"cannot verify adjacent continuity for anomaly at {open_time.isoformat()}"
            )
        verified.append(replace(item, adjacent_continuity_verified=True))
    return verified


def _last_close(candles: Sequence[Candle]) -> datetime | None:
    return max((candle.close_time for candle in candles), default=None)


def _checksum_from_text(text: str, archive_name: str) -> str:
    lines = text.splitlines()
    if len(lines) != 1:
        raise BinanceVisionError(f"invalid official checksum sidecar: {archive_name}")
    match = re.fullmatch(
        r"(?P<hash>[0-9a-fA-F]{64})(?:  (?P<plain>[^/\\]+)| \*(?P<star>[^/\\]+))",
        lines[0],
    )
    if match is None:
        raise BinanceVisionError(f"invalid official checksum sidecar: {archive_name}")
    listed_name = match.group("plain") or match.group("star")
    if listed_name != archive_name:
        raise BinanceVisionError(f"checksum sidecar archive identity mismatch: {archive_name}")
    return match.group("hash").lower()


def _read_cached_zip(zip_path: Path, checksum: str) -> bytes | None:
    if not zip_path.exists():
        return None
    try:
        data = zip_path.read_bytes()
    except OSError:
        return None
    return data if hashlib.sha256(data).hexdigest() == checksum else None


def _read_verified_cache(
    zip_path: Path, checksum_path: Path, archive_name: str
) -> tuple[bytes, str] | None:
    if not zip_path.exists() or not checksum_path.exists():
        return None
    try:
        checksum = _checksum_from_text(checksum_path.read_text(encoding="utf-8"), archive_name)
    except (OSError, BinanceVisionError):
        return None
    data = _read_cached_zip(zip_path, checksum)
    return (data, checksum) if data is not None else None


def _write_verified_cache(
    zip_path: Path, checksum_path: Path, data: bytes, checksum: str, archive_name: str
) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = f"{checksum}  {archive_name}\n".encode()
    for path, content in ((zip_path, data), (checksum_path, sidecar)):
        _write_cache_file(path, content)


def _write_cache_file(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
