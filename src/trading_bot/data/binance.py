from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime, timedelta

import websockets

from trading_bot.data.binance_historical import (
    BinanceDataError,
    BinanceHistoricalDataClient,
    BinanceRateLimitError,
    BinanceResponseError,
)
from trading_bot.data.binance_parser import (
    datetime_from_milliseconds,
    parse_binance_spot_1h_websocket_kline,
)
from trading_bot.domain.models import Candle

__all__ = [
    "BinanceDataError",
    "BinanceHistoricalDataClient",
    "BinancePublicClient",
    "BinanceRateLimitError",
    "BinanceResponseError",
]

_MILLISECOND = timedelta(milliseconds=1)


class BinancePublicClient:
    """Deprecated compatibility wrapper around the safe paginated historical client."""

    def __init__(self, historical_client: BinanceHistoricalDataClient | None = None) -> None:
        self._historical_client = historical_client or BinanceHistoricalDataClient()

    async def historical(
        self, start_ms: int | str, end_ms: int | str | None = None
    ) -> list[Candle]:
        start = datetime_from_milliseconds(start_ms, "start")
        end = (
            datetime_from_milliseconds(end_ms, "end") + _MILLISECOND
            if end_ms is not None
            else datetime.now(UTC)
        )
        return await self._historical_client.fetch_closed(start, end)

    async def closed_klines(
        self,
        on_connected: Callable[[], None] | None = None,
        on_disconnected: Callable[[], None] | None = None,
    ) -> AsyncIterator[Candle]:
        """Yield only closed klines and reconnect after a public-stream disconnect."""
        url = "wss://stream.binance.com:9443/ws/btcusdt@kline_1h"
        delay = 1.0
        while True:
            try:
                async with websockets.connect(url) as socket:
                    delay = 1.0
                    if on_connected is not None:
                        on_connected()
                    async for raw in socket:
                        item = json.loads(raw)["k"]
                        if item["x"]:
                            yield parse_binance_spot_1h_websocket_kline(item)
            except websockets.WebSocketException:
                if on_disconnected is not None:
                    on_disconnected()
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60)
