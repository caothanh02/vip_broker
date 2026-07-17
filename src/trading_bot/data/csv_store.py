from __future__ import annotations

import csv
import json
import os
import tempfile
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

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
        return Candle(
            open_time=_parse_time(row["open_time"], "open_time"),
            close_time=_parse_time(row["close_time"], "close_time"),
            symbol=row["symbol"],
            timeframe=row["timeframe"],
            open=Decimal(row["open"]),
            high=Decimal(row["high"]),
            low=Decimal(row["low"]),
            close=Decimal(row["close"]),
            volume=Decimal(row["volume"]),
            is_closed=is_closed,
        )
    except (KeyError, InvalidOperation, ValueError) as exc:
        raise CsvDataError("malformed candle CSV row") from exc


def merge_candles(*groups: Iterable[Candle]) -> list[Candle]:
    by_open: dict[datetime, Candle] = {}
    for candle in (item for group in groups for item in group):
        previous = by_open.get(candle.open_time)
        if previous is not None and previous != candle:
            raise CsvDataError(f"conflicting candles at {candle.open_time.isoformat()}")
        by_open[candle.open_time] = candle
    merged = [by_open[key] for key in sorted(by_open)]
    try:
        validate_candles(merged)
    except CandleValidationError as exc:
        raise CsvDataError(f"invalid merged candles: {exc}") from exc
    return merged


def read_candles(path: Path) -> list[Candle]:
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
    return merge_candles(candles)


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


def write_candles_atomic(path: Path, candles: Iterable[Candle]) -> list[Candle]:
    normalized = merge_candles(candles)
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


def write_metadata_atomic(path: Path, metadata: dict[str, Any]) -> None:
    _atomic_write(path, json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write(path, json.dumps(payload, indent=2, allow_nan=False) + "\n")
