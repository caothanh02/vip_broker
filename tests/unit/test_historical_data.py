from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest

from trading_bot.cli import main, parse_utc
from trading_bot.data.binance_historical import (
    BinanceHistoricalDataClient,
    BinanceRateLimitError,
    BinanceResponseError,
)
from trading_bot.data.csv_store import (
    CsvDataError,
    merge_candles,
    read_candles,
    write_candles_atomic,
)
from trading_bot.data.historical import download_historical_csv
from trading_bot.domain.models import Candle

BASE = datetime(2024, 1, 1, tzinfo=UTC)


def candle(hour: int, price: str = "100") -> Candle:
    value = Decimal(price)
    return Candle(
        BASE + timedelta(hours=hour),
        BASE + timedelta(hours=hour + 1),
        "BTC/USDT",
        "1h",
        value,
        value + 1,
        value - 1,
        value,
        Decimal("123.4567890123456789"),
        True,
    )


def kline(hour: int, price: str = "100") -> list[Any]:
    opened = int((BASE + timedelta(hours=hour)).timestamp() * 1000)
    value = Decimal(price)
    return [
        opened,
        price,
        str(value + 1),
        str(value - 1),
        price,
        "123.4567890123456789",
        opened + 3_599_999,
    ]


def client(handler: httpx.MockTransport, **kwargs: Any) -> BinanceHistoricalDataClient:
    return BinanceHistoricalDataClient(
        transport=handler,
        now=lambda: BASE + timedelta(days=30),
        backoff_seconds=0,
        **kwargs,
    )


def test_binance_client_canonicalizes_close_and_preserves_decimal() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json=[kline(0, "100.123456789")])
    )
    candles = asyncio.run(client(transport).fetch_closed(BASE, BASE + timedelta(hours=1)))
    assert candles == [candle(0, "100.123456789")]
    assert candles[0].close_time == candles[0].open_time + timedelta(hours=1)
    assert candles[0].volume == Decimal("123.4567890123456789")


def test_binance_client_paginates_without_duplicates() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        start = int(request.url.params["startTime"])
        if start == int(BASE.timestamp() * 1000):
            return httpx.Response(200, json=[kline(0), kline(1)])
        if start == int((BASE + timedelta(hours=2)).timestamp() * 1000):
            return httpx.Response(200, json=[kline(2)])
        return httpx.Response(200, json=[])

    candles = asyncio.run(
        client(httpx.MockTransport(handler), page_limit=2).fetch_closed(
            BASE, BASE + timedelta(hours=4)
        )
    )
    assert [item.open_time for item in candles] == [
        BASE + timedelta(hours=index) for index in range(3)
    ]


def test_binance_client_omits_current_open_candle() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=[kline(0)]))
    candles = asyncio.run(
        BinanceHistoricalDataClient(
            transport=transport,
            now=lambda: BASE + timedelta(minutes=30),
            backoff_seconds=0,
        ).fetch_closed(BASE, BASE + timedelta(hours=1))
    )
    assert candles == []


def test_binance_client_retries_429_and_fails_malformed_payload() -> None:
    calls = 0
    sleeps: list[float] = []

    async def sleep(value: float) -> None:
        sleeps.append(value)

    def retry_handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(429 if calls == 1 else 200, json=[])

    assert (
        asyncio.run(
            client(httpx.MockTransport(retry_handler), sleep=sleep).fetch_closed(
                BASE, BASE + timedelta(hours=1)
            )
        )
        == []
    )
    assert calls == 2 and sleeps == [0]
    malformed = httpx.MockTransport(lambda request: httpx.Response(200, json={"bad": "payload"}))
    with pytest.raises(BinanceResponseError):
        asyncio.run(client(malformed).fetch_closed(BASE, BASE + timedelta(hours=1)))


def test_binance_client_retries_5xx_and_rejects_invalid_kline_fields() -> None:
    calls = 0

    def retry_handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503 if calls == 1 else 200, json=[])

    assert (
        asyncio.run(
            client(httpx.MockTransport(retry_handler)).fetch_closed(BASE, BASE + timedelta(hours=1))
        )
        == []
    )
    invalid = kline(0)
    invalid[1] = "not-a-decimal"
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=[invalid]))
    with pytest.raises(BinanceResponseError, match="decimal"):
        asyncio.run(client(transport).fetch_closed(BASE, BASE + timedelta(hours=1)))


def test_binance_client_detects_non_progress_and_retry_exhaustion() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=[kline(0)]))
    with pytest.raises(BinanceResponseError, match="before the cursor"):
        asyncio.run(client(transport).fetch_closed(BASE, BASE + timedelta(hours=2)))
    rate_limited = httpx.MockTransport(lambda request: httpx.Response(429, json=[]))
    with pytest.raises(BinanceRateLimitError):
        asyncio.run(
            client(rate_limited, max_retries=0).fetch_closed(BASE, BASE + timedelta(hours=1))
        )


def test_csv_round_trip_deduplicates_and_rejects_conflicts(tmp_path: Path) -> None:
    path = tmp_path / "btc.csv"
    written = write_candles_atomic(path, [candle(1), candle(0), candle(1)])
    assert [item.open_time for item in written] == [BASE, BASE + timedelta(hours=1)]
    assert read_candles(path) == written
    with pytest.raises(CsvDataError, match="conflicting"):
        merge_candles([candle(0)], [candle(0, "101")])
    original = path.read_text(encoding="utf-8")
    with pytest.raises(CsvDataError):
        write_candles_atomic(path, [candle(0), candle(2)])
    assert path.read_text(encoding="utf-8") == original


class FakeClient:
    def __init__(self, response: list[Candle]) -> None:
        self.response = response
        self.calls: list[tuple[datetime, datetime]] = []

    async def fetch_closed(self, start: datetime, end: datetime) -> list[Candle]:
        self.calls.append((start, end))
        return self.response


def test_incremental_download_merges_and_skips_complete_range(tmp_path: Path) -> None:
    path = tmp_path / "btc.csv"
    write_candles_atomic(path, [candle(0), candle(1)])
    fake = FakeClient([candle(2)])
    summary = asyncio.run(download_historical_csv(fake, BASE, BASE + timedelta(hours=3), path))
    assert fake.calls == [(BASE + timedelta(hours=2), BASE + timedelta(hours=3))]
    assert summary.old_candles == 2 and summary.new_candles == 1
    assert len(read_candles(path)) == 3
    metadata = json.loads((tmp_path / "btc.csv.metadata.json").read_text(encoding="utf-8"))
    assert metadata["candle_count"] == 3
    complete = FakeClient([])
    asyncio.run(download_historical_csv(complete, BASE, BASE + timedelta(hours=3), path))
    assert complete.calls == []


def test_incremental_overwrite_replaces_only_after_success(tmp_path: Path) -> None:
    path = tmp_path / "btc.csv"
    write_candles_atomic(path, [candle(0), candle(1)])
    fake = FakeClient([candle(4), candle(5)])
    summary = asyncio.run(
        download_historical_csv(
            fake, BASE + timedelta(hours=4), BASE + timedelta(hours=6), path, True
        )
    )
    assert summary.old_candles == 0
    assert [item.open_time for item in read_candles(path)] == [
        BASE + timedelta(hours=4),
        BASE + timedelta(hours=5),
    ]


def test_cli_backtests_csv_and_writes_report(tmp_path: Path) -> None:
    data = tmp_path / "btc.csv"
    report = tmp_path / "report.json"
    write_candles_atomic(data, [candle(index, str(100 + index)) for index in range(240)])
    assert main(["backtest", "--input", str(data), "--output", str(report)]) == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["input_file"] == str(data)
    assert payload["candle_count"] == 240
    assert main(["validate-data", "--input", str(tmp_path / "missing.csv")]) == 1
    assert parse_utc("2024-01-01").tzinfo is UTC
