from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any

from trading_bot.domain.models import Candle

INTERVAL_MS = 3_600_000
_INTERVAL = timedelta(hours=1)


class BinanceKlineParseError(ValueError):
    pass


def milliseconds_from_value(value: Any, field: str) -> int:
    """Parse an exact non-negative integer millisecond timestamp."""
    if isinstance(value, bool) or isinstance(value, float):
        raise BinanceKlineParseError(f"invalid Binance {field} timestamp")
    if isinstance(value, int):
        milliseconds = value
    elif isinstance(value, str):
        try:
            milliseconds = int(value)
        except ValueError as exc:
            raise BinanceKlineParseError(f"invalid Binance {field} timestamp") from exc
        if str(milliseconds) != value:
            raise BinanceKlineParseError(f"invalid Binance {field} timestamp")
    else:
        raise BinanceKlineParseError(f"invalid Binance {field} timestamp")
    if milliseconds < 0:
        raise BinanceKlineParseError(f"invalid Binance {field} timestamp")
    return milliseconds


def datetime_from_milliseconds(value: Any, field: str = "") -> datetime:
    milliseconds = milliseconds_from_value(value, field or "")
    try:
        return datetime.fromtimestamp(milliseconds // 1000, UTC) + timedelta(
            milliseconds=milliseconds % 1000
        )
    except (ValueError, OSError) as exc:
        raise BinanceKlineParseError("invalid Binance timestamp") from exc


def parse_binance_spot_1h_kline(row: Any) -> Candle:
    """Parse a REST kline into the domain's canonical half-open 1h candle."""
    if not isinstance(row, list) or len(row) < 7:
        raise BinanceKlineParseError("malformed Binance kline")
    open_ms = milliseconds_from_value(row[0], "open")
    raw_close_ms = milliseconds_from_value(row[6], "close")
    if open_ms % INTERVAL_MS or raw_close_ms != open_ms + INTERVAL_MS - 1:
        raise BinanceKlineParseError("invalid Binance 1h kline timestamps")
    open_time = datetime_from_milliseconds(open_ms)
    close_time = open_time + _INTERVAL
    try:
        open_, high, low, close, volume = (Decimal(str(row[index])) for index in range(1, 6))
    except (InvalidOperation, ValueError) as exc:
        raise BinanceKlineParseError("invalid Binance kline decimal") from exc
    if not all(value.is_finite() for value in (open_, high, low, close, volume)):
        raise BinanceKlineParseError("non-finite Binance kline decimal")
    return Candle(open_time, close_time, "BTC/USDT", "1h", open_, high, low, close, volume, True)


def parse_binance_spot_1h_websocket_kline(item: Any) -> Candle:
    """Canonical closed-candle conversion for the legacy public websocket adapter."""
    if not isinstance(item, dict):
        raise BinanceKlineParseError("malformed Binance websocket kline")
    return parse_binance_spot_1h_kline(
        [
            item.get("t"),
            item.get("o"),
            item.get("h"),
            item.get("l"),
            item.get("c"),
            item.get("v"),
            item.get("T"),
        ]
    )
