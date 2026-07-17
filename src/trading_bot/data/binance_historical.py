from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any

import httpx

from trading_bot.domain.models import Candle

BINANCE_REST = "https://api.binance.com/api/v3"
_INTERVAL = timedelta(hours=1)
_MAX_LIMIT = 1000


class BinanceDataError(RuntimeError):
    """A public Binance historical-data request could not be completed safely."""


class BinanceRateLimitError(BinanceDataError):
    """Binance continued returning HTTP 429 after the configured retries."""


class BinanceResponseError(BinanceDataError):
    """Binance returned a malformed or internally inconsistent kline response."""


Sleep = Callable[[float], Awaitable[None]]
Clock = Callable[[], datetime]


def _utc(value: datetime, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise BinanceResponseError(f"{field} must be timezone-aware")
    return value.astimezone(UTC)


class BinanceHistoricalDataClient:
    """BTCUSDT Spot 1h closed-candle downloader using Binance's public REST API only."""

    def __init__(
        self,
        base_url: str = BINANCE_REST,
        timeout_seconds: float = 20.0,
        max_retries: int = 3,
        page_limit: int = _MAX_LIMIT,
        backoff_seconds: float = 0.25,
        max_backoff_seconds: float = 2.0,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Sleep = asyncio.sleep,
        now: Clock | None = None,
    ) -> None:
        if max_retries < 0 or page_limit < 1 or page_limit > _MAX_LIMIT:
            raise ValueError("invalid retry or page limit configuration")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.page_limit = page_limit
        self.backoff_seconds = backoff_seconds
        self.max_backoff_seconds = max_backoff_seconds
        self.transport = transport
        self.sleep = sleep
        self.now = now or (lambda: datetime.now(UTC))

    async def fetch_closed(self, start_time: datetime, end_time: datetime) -> list[Candle]:
        """Fetch the half-open UTC range ``[start_time, end_time)`` page by page."""
        start = _utc(start_time, "start_time")
        end = _utc(end_time, "end_time")
        if end <= start:
            raise ValueError("end_time must be after start_time")
        cursor = start
        candles: list[Candle] = []
        seen_open_times: set[datetime] = set()
        while cursor < end:
            rows = await self._request_page(cursor, end)
            if not rows:
                break
            page_opens: list[datetime] = []
            for row in rows:
                candle = self._parse_kline(row)
                if candle.open_time < cursor:
                    raise BinanceResponseError("pagination returned a timestamp before the cursor")
                if candle.open_time >= end:
                    continue
                if candle.open_time in seen_open_times:
                    raise BinanceResponseError("pagination returned a duplicate timestamp")
                seen_open_times.add(candle.open_time)
                page_opens.append(candle.open_time)
                if candle.close_time <= self.now().astimezone(UTC):
                    candles.append(candle)
            if not page_opens:
                raise BinanceResponseError("pagination made no forward progress")
            next_cursor = max(page_opens) + _INTERVAL
            if next_cursor <= cursor:
                raise BinanceResponseError("pagination made no forward progress")
            cursor = next_cursor
        return candles

    async def _request_page(self, cursor: datetime, end: datetime) -> list[Any]:
        params: dict[str, str | int] = {
            "symbol": "BTCUSDT",
            "interval": "1h",
            "startTime": int(cursor.timestamp() * 1000),
            "endTime": int(end.timestamp() * 1000) - 1,
            "limit": self.page_limit,
        }
        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(
                    base_url=self.base_url,
                    timeout=self.timeout_seconds,
                    transport=self.transport,
                ) as client:
                    response = await client.get("klines", params=params)
            except httpx.HTTPError as exc:
                if attempt == self.max_retries:
                    raise BinanceDataError("Binance request failed after retries") from exc
                await self._backoff(attempt)
                continue
            if response.status_code == 429:
                if attempt == self.max_retries:
                    raise BinanceRateLimitError("Binance rate limit persisted after retries")
                await self._backoff(attempt)
                continue
            if response.status_code >= 500:
                if attempt == self.max_retries:
                    raise BinanceDataError(f"Binance server error: HTTP {response.status_code}")
                await self._backoff(attempt)
                continue
            if response.is_error:
                raise BinanceDataError(f"Binance request failed: HTTP {response.status_code}")
            try:
                payload = response.json()
            except ValueError as exc:
                raise BinanceResponseError("Binance response is not valid JSON") from exc
            if not isinstance(payload, list):
                raise BinanceResponseError("Binance response must be a list of klines")
            return payload
        raise AssertionError("unreachable")

    async def _backoff(self, attempt: int) -> None:
        await self.sleep(min(self.backoff_seconds * (2**attempt), self.max_backoff_seconds))

    @staticmethod
    def _parse_kline(row: Any) -> Candle:
        if not isinstance(row, list) or len(row) < 7:
            raise BinanceResponseError("malformed Binance kline")
        try:
            open_ms = int(row[0])
            raw_close_ms = int(row[6])
            open_time = datetime.fromtimestamp(open_ms / 1000, UTC)
        except (TypeError, ValueError, OSError) as exc:
            raise BinanceResponseError("invalid Binance kline timestamp") from exc
        close_time = open_time + _INTERVAL
        if raw_close_ms < open_ms or raw_close_ms >= int(close_time.timestamp() * 1000):
            raise BinanceResponseError("invalid Binance kline close timestamp")
        try:
            open_, high, low, close, volume = (Decimal(str(row[index])) for index in range(1, 6))
        except (InvalidOperation, ValueError) as exc:
            raise BinanceResponseError("invalid Binance kline decimal") from exc
        return Candle(
            open_time=open_time,
            close_time=close_time,
            symbol="BTC/USDT",
            timeframe="1h",
            open=open_,
            high=high,
            low=low,
            close=close,
            volume=volume,
            is_closed=True,
        )
