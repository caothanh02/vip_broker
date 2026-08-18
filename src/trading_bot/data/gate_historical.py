"""Public, read-only Gate.io Spot BTC/USDT 1-hour candle retrieval.

The client is deliberately narrow: it cannot authenticate, select an account,
or address any trading endpoint. A request is capped at 1,000 candles and
pages are paced at one request per second.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any

import httpx

from trading_bot.data.time_ranges import validate_hour_aligned_range
from trading_bot.domain.models import Candle

GATE_SPOT_REST = "https://api.gateio.ws/api/v4"
_INTERVAL = timedelta(hours=1)
_MAX_PAGE_CANDLES = 1000

Sleep = Callable[[float], Awaitable[None]]
Clock = Callable[[], datetime]


class GateDataError(ValueError):
    """A public Gate candle response did not meet the strict input contract."""


class GateRateLimitError(GateDataError):
    """Gate continued to rate-limit a bounded public request."""


class GateHistoricalDataClient:
    """Fetch only closed BTC_USDT Spot 1-hour candles from Gate's public API."""

    def __init__(
        self,
        base_url: str = GATE_SPOT_REST,
        timeout_seconds: float = 20.0,
        max_retries: int = 3,
        page_candles: int = _MAX_PAGE_CANDLES,
        minimum_request_interval_seconds: float = 1.0,
        backoff_seconds: float = 1.0,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Sleep = asyncio.sleep,
        now: Clock | None = None,
    ) -> None:
        if (
            max_retries < 0
            or page_candles < 1
            or page_candles > _MAX_PAGE_CANDLES
            or minimum_request_interval_seconds < 1.0
            or backoff_seconds < 1.0
        ):
            raise ValueError("invalid Gate public-client configuration")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.page_candles = page_candles
        self.minimum_request_interval_seconds = minimum_request_interval_seconds
        self.backoff_seconds = backoff_seconds
        self.transport = transport
        self.sleep = sleep
        self.now = now or (lambda: datetime.now(UTC))

    async def fetch_closed(self, start_time: datetime, end_time: datetime) -> list[Candle]:
        """Fetch exact, non-overlapping pages for a half-open UTC range."""
        start, requested_end = validate_hour_aligned_range(start_time, end_time)
        effective_end = min(requested_end, _closed_hour_boundary(self.now()))
        if effective_end <= start:
            return []

        candles: list[Candle] = []
        cursor = start
        first_request = True
        while cursor < effective_end:
            page_end = min(cursor + self.page_candles * _INTERVAL, effective_end)
            if not first_request:
                await self.sleep(self.minimum_request_interval_seconds)
            first_request = False
            candles.extend(await self._request_page(cursor, page_end))
            cursor = page_end
        return candles

    async def _request_page(self, start: datetime, end: datetime) -> list[Candle]:
        # Gate's ``to`` value is inclusive; ``end - 1 second`` preserves this
        # project's half-open [start, end) range contract.
        params = {
            "currency_pair": "BTC_USDT",
            "interval": "1h",
            "from": str(int(start.timestamp())),
            "to": str(int(end.timestamp()) - 1),
        }
        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(
                    base_url=self.base_url,
                    timeout=self.timeout_seconds,
                    transport=self.transport,
                ) as client:
                    response = await client.get("spot/candlesticks", params=params)
            except httpx.HTTPError as exc:
                if attempt == self.max_retries:
                    raise GateDataError("Gate request failed after retries") from exc
                await self._backoff(attempt)
                continue
            if response.status_code == 429:
                if attempt == self.max_retries:
                    raise GateRateLimitError("Gate rate limit persisted after retries")
                await self._backoff(attempt)
                continue
            if response.status_code >= 500:
                if attempt == self.max_retries:
                    raise GateDataError(f"Gate server error: HTTP {response.status_code}")
                await self._backoff(attempt)
                continue
            if response.is_error:
                raise GateDataError(f"Gate request failed: HTTP {response.status_code}")
            try:
                payload = response.json()
            except ValueError as exc:
                raise GateDataError("Gate response is not valid JSON") from exc
            if not isinstance(payload, list):
                raise GateDataError("Gate response must be a list of candlesticks")
            return _validate_page(payload, start, end)
        raise AssertionError("unreachable")

    async def _backoff(self, attempt: int) -> None:
        await self.sleep(self.backoff_seconds * (2**attempt))


def _validate_page(payload: list[Any], start: datetime, end: datetime) -> list[Candle]:
    expected_count = int((end - start) / _INTERVAL)
    if len(payload) != expected_count:
        raise GateDataError("Gate page does not contain the expected candle count")
    candles = [_parse_gate_spot_1h_candle(row) for row in payload]
    opens = [candle.open_time for candle in candles]
    if len(set(opens)) != len(opens):
        raise GateDataError("Gate page contains duplicate candle timestamps")
    expected_opens = {start + index * _INTERVAL for index in range(expected_count)}
    if set(opens) != expected_opens:
        raise GateDataError("Gate page has a gap or timestamp outside its requested range")
    return sorted(candles, key=lambda candle: candle.open_time)


def _parse_gate_spot_1h_candle(row: Any) -> Candle:
    # Gate Spot response: timestamp, quote volume, close, high, low, open.
    if not isinstance(row, list) or len(row) != 6:
        raise GateDataError("malformed Gate candlestick")
    open_time = _timestamp_seconds(row[0])
    if open_time.minute or open_time.second or open_time.microsecond:
        raise GateDataError("Gate candle timestamp is not hour-aligned")
    try:
        volume, close, high, low, open_ = (Decimal(str(row[index])) for index in range(1, 6))
    except (InvalidOperation, ValueError) as exc:
        raise GateDataError("invalid Gate OHLCV decimal") from exc
    if not all(value.is_finite() for value in (open_, high, low, close, volume)):
        raise GateDataError("non-finite Gate OHLCV")
    if min(open_, high, low, close) <= 0 or volume < 0:
        raise GateDataError("invalid Gate OHLCV")
    if high < max(open_, close, low) or low > min(open_, close, high):
        raise GateDataError("invalid Gate OHLC")
    return Candle(
        open_time,
        open_time + _INTERVAL,
        "BTC/USDT",
        "1h",
        open_,
        high,
        low,
        close,
        volume,
        True,
    )


def _timestamp_seconds(value: Any) -> datetime:
    if isinstance(value, bool) or isinstance(value, float):
        raise GateDataError("invalid Gate candle timestamp")
    try:
        seconds = int(value)
    except (TypeError, ValueError) as exc:
        raise GateDataError("invalid Gate candle timestamp") from exc
    if str(seconds) != str(value) or seconds < 0:
        raise GateDataError("invalid Gate candle timestamp")
    try:
        return datetime.fromtimestamp(seconds, UTC)
    except (OverflowError, OSError, ValueError) as exc:
        raise GateDataError("invalid Gate candle timestamp") from exc


def _closed_hour_boundary(now: datetime) -> datetime:
    if now.tzinfo is None or now.utcoffset() is None:
        raise GateDataError("clock must return a timezone-aware timestamp")
    return now.astimezone(UTC).replace(minute=0, second=0, microsecond=0)
