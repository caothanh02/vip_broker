from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any

from trading_bot.domain.models import Candle

_INTERVAL = timedelta(hours=1)


class BinanceKlineParseError(ValueError):
    pass


def _timestamp(milliseconds: Any, field: str) -> datetime:
    try:
        value = int(milliseconds)
        return datetime.fromtimestamp(value // 1000, UTC) + timedelta(milliseconds=value % 1000)
    except (TypeError, ValueError, OSError) as exc:
        raise BinanceKlineParseError(f"invalid Binance {field} timestamp") from exc


def parse_binance_spot_1h_kline(row: Any) -> Candle:
    """Parse a REST kline into the domain's canonical half-open 1h candle."""
    if not isinstance(row, list) or len(row) < 7:
        raise BinanceKlineParseError("malformed Binance kline")
    open_time = _timestamp(row[0], "open")
    raw_close_time = _timestamp(row[6], "close")
    close_time = open_time + _INTERVAL
    if raw_close_time < open_time or raw_close_time >= close_time:
        raise BinanceKlineParseError("invalid Binance kline close timestamp")
    try:
        open_, high, low, close, volume = (Decimal(str(row[index])) for index in range(1, 6))
    except (InvalidOperation, ValueError) as exc:
        raise BinanceKlineParseError("invalid Binance kline decimal") from exc
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
