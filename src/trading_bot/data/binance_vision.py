from __future__ import annotations

import csv
import hashlib
import io
import os
import tempfile
import zipfile
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
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
    unit, expected, quality, deviation_us = _timestamp_policy(open_timestamp, close_timestamp)
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
    scale = 1_000 if unit == "milliseconds" else 1_000_000
    seconds, remainder = divmod(open_timestamp, scale)
    open_time = datetime.fromtimestamp(seconds, UTC) + timedelta(
        microseconds=remainder * (1_000_000 // scale)
    )
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


def _timestamp_policy(
    open_timestamp: int, close_timestamp: int
) -> tuple[Literal["milliseconds", "microseconds"], int, ArchiveTimestampQuality, int]:
    # Both representations are divisible by 3,600,000 at an hourly boundary;
    # infer their precision before checking alignment.  Modern archive rows use
    # epoch microseconds (16 digits), legacy rows use epoch milliseconds.
    unit: Literal["milliseconds", "microseconds"] = (
        "microseconds" if open_timestamp >= 1_000_000_000_000_000 else "milliseconds"
    )
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
    return unit, expected, quality, early * (1_000 if unit == "milliseconds" else 1)


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


def anomaly_report(parsed: Sequence[ParsedArchiveCandle]) -> dict[str, object]:
    """Create a deterministic, path-free audit document for accepted archive anomalies."""
    anomalies = [item for item in parsed if item.quality is not ArchiveTimestampQuality.EXACT]
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
            "early_close_deviation_ms": item.early_close_deviation_us // 1_000,
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
            "late_close_allowed": False,
            "rest_policy": "strict",
        },
        "summary": {
            "exact_timestamp_candles": len(parsed) - len(records),
            "accepted_early_close_anomalies": len(records),
            "rejected_timestamp_anomalies": 0,
            "maximum_observed_early_close_ms": max(
                (record["early_close_deviation_ms"] for record in records), default=0
            ),
        },
        "anomalies": records,
    }


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
        self.parsed: list[ParsedArchiveCandle] = []
        self.archive_urls: list[str] = []
        self.archive_checksums: dict[str, str] = {}
        self.monthly_archives: list[str] = []
        self.daily_archives: list[str] = []
        self.rest_suffix: tuple[datetime, datetime] | None = None

    async def fetch_closed(self, start_time: datetime, end_time: datetime) -> list[Candle]:
        start, end = validate_hour_aligned_range(start_time, end_time)
        closed_end = min(end, self.now().astimezone(UTC).replace(minute=0, second=0, microsecond=0))
        self.parsed = []
        self.archive_urls = []
        self.archive_checksums = {}
        self.monthly_archives = []
        self.daily_archives = []
        self.rest_suffix = None
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
                candles.extend(await self.rest_client.fetch_closed(suffix_start, closed_end))
                self.rest_suffix = (suffix_start, closed_end)
        try:
            merged = merge_candles(candles)
        except ValueError as exc:
            raise BinanceVisionError(str(exc)) from exc
        result = [candle for candle in merged if start <= candle.open_time < closed_end]
        _validate_continuity(result, start, closed_end)
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
        cached = _read_verified_cache(zip_path, checksum_path)
        if cached is not None:
            return cached
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
            checksum = _checksum_from_text(checksum_response.text, relative)
            data = data_response.content
            if hashlib.sha256(data).hexdigest() != checksum:
                raise BinanceVisionError(f"checksum mismatch: {relative}")
            _write_verified_cache(zip_path, checksum_path, data, checksum)
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
            if len(names) != 1 or Path(names[0]).name != names[0] or not names[0].endswith(".csv"):
                raise BinanceVisionError("unsafe archive path")
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
    return parsed


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


def _last_close(candles: Sequence[Candle]) -> datetime | None:
    return max((candle.close_time for candle in candles), default=None)


def _checksum_from_text(text: str, relative: str) -> str:
    parts = text.split()
    if (
        not parts
        or len(parts[0]) != 64
        or any(char not in "0123456789abcdefABCDEF" for char in parts[0])
    ):
        raise BinanceVisionError(f"invalid official checksum: {relative}")
    return parts[0].lower()


def _read_verified_cache(zip_path: Path, checksum_path: Path) -> tuple[bytes, str] | None:
    if not zip_path.exists() or not checksum_path.exists():
        return None
    try:
        checksum = _checksum_from_text(checksum_path.read_text(encoding="utf-8"), str(zip_path))
        data = zip_path.read_bytes()
    except OSError:
        return None
    return (data, checksum) if hashlib.sha256(data).hexdigest() == checksum else None


def _write_verified_cache(zip_path: Path, checksum_path: Path, data: bytes, checksum: str) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    for path, content in ((zip_path, data), (checksum_path, (checksum + "\n").encode())):
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
