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
    output: Path


async def download_historical_csv(
    client: BinanceHistoricalDataClient,
    start: datetime,
    end: datetime,
    output: Path,
    overwrite: bool = False,
) -> DownloadSummary:
    """Download closed 1h candles and atomically maintain one normalized CSV."""
    existing: list[Candle] = []
    if output.exists() and not overwrite:
        existing = read_candles(output)
    request_start = start
    if existing:
        request_start = existing[-1].open_time + _INTERVAL
    if request_start >= end:
        return _summary(existing, len(existing), output)
    downloaded = await client.fetch_closed(request_start, end)
    merged = merge_candles(downloaded) if overwrite else merge_candles(existing, downloaded)
    # Do not replace a valid existing file until the complete merged data is valid.
    persisted = write_candles_atomic(output, merged)
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
            "first_candle_open": _iso(persisted[0].open_time),
            "last_candle_close": _iso(persisted[-1].close_time),
            "candle_count": len(persisted),
            "downloaded_at": _iso(datetime.now(UTC)),
            "schema_version": SCHEMA_VERSION,
        },
    )
    return _summary(persisted, 0 if overwrite else len(existing), output)


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _summary(candles: list[Candle], old_candles: int, output: Path) -> DownloadSummary:
    return DownloadSummary(
        old_candles=old_candles,
        new_candles=len(candles) - old_candles,
        total_candles=len(candles),
        first_candle_open=candles[0].open_time if candles else None,
        last_candle_close=candles[-1].close_time if candles else None,
        output=output,
    )


def summary_json(summary: DownloadSummary) -> dict[str, str | int | None]:
    values = asdict(summary)
    return {
        "old_candles": values["old_candles"],
        "new_candles": values["new_candles"],
        "total_candles": values["total_candles"],
        "first_candle_open": _iso(values["first_candle_open"])
        if values["first_candle_open"] is not None
        else None,
        "last_candle_close": _iso(values["last_candle_close"])
        if values["last_candle_close"] is not None
        else None,
        "output": str(values["output"]),
    }
