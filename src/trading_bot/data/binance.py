from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, cast

import httpx
import websockets

from trading_bot.data.binance_historical import (
    BinanceDataError,
    BinanceHistoricalDataClient,
    BinanceRateLimitError,
    BinanceResponseError,
)
from trading_bot.domain.models import Candle

__all__ = [
    "BinanceDataError",
    "BinanceHistoricalDataClient",
    "BinancePublicClient",
    "BinanceRateLimitError",
    "BinanceResponseError",
]

BINANCE_REST = "https://api.binance.com/api/v3"


def _candle(row: list[Any]) -> Candle:
    return Candle(
        open_time=datetime.fromtimestamp(int(row[0]) / 1000, UTC),
        close_time=datetime.fromtimestamp(int(row[6]) / 1000, UTC),
        symbol="BTC/USDT",
        timeframe="1h",
        open=Decimal(str(row[1])),
        high=Decimal(str(row[2])),
        low=Decimal(str(row[3])),
        close=Decimal(str(row[4])),
        volume=Decimal(str(row[5])),
        is_closed=True,
    )


class BinancePublicClient:
    """Public-only market-data client. It never accepts credentials or sends orders."""

    async def historical(self, start_ms: int, end_ms: int | None = None) -> list[Candle]:
        params: dict[str, str | int] = {
            "symbol": "BTCUSDT",
            "interval": "1h",
            "startTime": start_ms,
            "limit": 1000,
        }
        if end_ms is not None:
            params["endTime"] = end_ms
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.get(f"{BINANCE_REST}/klines", params=params)
            response.raise_for_status()
        payload = cast(list[list[Any]], response.json())
        return [_candle(item) for item in payload]

    async def closed_klines(self) -> AsyncIterator[Candle]:
        url = "wss://stream.binance.com:9443/ws/btcusdt@kline_1h"
        delay = 1.0
        while True:
            try:
                async with websockets.connect(url) as socket:
                    delay = 1.0
                    async for raw in socket:
                        item = json.loads(raw)["k"]
                        if item["x"]:
                            yield Candle(
                                datetime.fromtimestamp(item["t"] / 1000, UTC),
                                datetime.fromtimestamp(item["T"] / 1000, UTC),
                                "BTC/USDT",
                                "1h",
                                Decimal(item["o"]),
                                Decimal(item["h"]),
                                Decimal(item["l"]),
                                Decimal(item["c"]),
                                Decimal(item["v"]),
                                True,
                            )
            except websockets.WebSocketException:
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60)
