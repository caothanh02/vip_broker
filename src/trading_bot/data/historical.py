from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from trading_bot.data.binance_historical import BinanceHistoricalDataClient
from trading_bot.data.csv_store import (
    SCHEMA_VERSION,
    merge_candles,
    metadata_path,
    read_candles,
    write_candles_atomic,
    write_metadata_atomic,
)
from trading_bot.domain.models import Candle

_INTERVAL = timedelta(hours=1)


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


async def download_historical_csv(
    client: BinanceHistoricalDataClient,
    start: datetime,
    end: datetime,
    output: Path,
    overwrite: bool = False,
) -> DownloadSummary:
    """Download closed 1h candles and atomically maintain one normalized CSV."""
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
            "downloaded_at": _iso(datetime.now(UTC)),
            "schema_version": SCHEMA_VERSION,
        },
    )
    return _summary(persisted, 0 if overwrite else len(existing), start, end, effective_end, output)


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _effective_end(client: BinanceHistoricalDataClient, end: datetime) -> datetime:
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


def summary_json(summary: DownloadSummary) -> dict[str, str | int | None]:
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
