from __future__ import annotations

import asyncio
import inspect
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from trading_bot.cli import main
from trading_bot.data.gate_historical import GateDataError, GateHistoricalDataClient

BASE = datetime(2019, 1, 1, tzinfo=UTC)


def _row(hour: int) -> list[str]:
    opened = int((BASE + timedelta(hours=hour)).timestamp())
    return [str(opened), "1000", "100", "101", "99", "100"]


def _client(handler: httpx.MockTransport, sleeps: list[float]) -> GateHistoricalDataClient:
    async def sleep(value: float) -> None:
        sleeps.append(value)

    return GateHistoricalDataClient(
        transport=handler,
        sleep=sleep,
        now=lambda: BASE + timedelta(days=100),
    )


def test_gate_client_uses_half_open_pages_and_conservative_pacing() -> None:
    requests: list[httpx.Request] = []
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        start = int(request.url.params["from"])
        end = int(request.url.params["to"])
        first_hour = int((datetime.fromtimestamp(start, UTC) - BASE) / timedelta(hours=1))
        last_hour = int((datetime.fromtimestamp(end, UTC) - BASE) / timedelta(hours=1))
        return httpx.Response(200, json=[_row(hour) for hour in range(first_hour, last_hour + 1)])

    candles = asyncio.run(
        _client(httpx.MockTransport(handler), sleeps).fetch_closed(
            BASE, BASE + timedelta(hours=1001)
        )
    )

    assert len(candles) == 1001
    assert {item.url.path for item in requests} == {"/api/v4/spot/candlesticks"}
    lengths = [int(item.url.params["to"]) - int(item.url.params["from"]) + 1 for item in requests]
    assert lengths == [3_600_000, 3_600]
    assert sleeps == [1.0]
    assert candles[0].open_time == BASE
    assert candles[-1].close_time == BASE + timedelta(hours=1001)


def test_gate_client_rejects_gap_before_any_publication() -> None:
    sleeps: list[float] = []
    with pytest.raises(GateDataError, match="expected candle count"):
        asyncio.run(
            _client(
                httpx.MockTransport(lambda request: httpx.Response(200, json=[_row(0)])), sleeps
            ).fetch_closed(BASE, BASE + timedelta(hours=2))
        )


def test_gate_client_rejects_invalid_ohlcv() -> None:
    sleeps: list[float] = []
    invalid = _row(0)
    invalid[3] = "99"
    with pytest.raises(GateDataError, match="invalid Gate OHLC"):
        asyncio.run(
            _client(
                httpx.MockTransport(lambda request: httpx.Response(200, json=[invalid])), sleeps
            ).fetch_closed(BASE, BASE + timedelta(hours=1))
        )


def test_gate_client_retries_rate_limit_with_bounded_backoff() -> None:
    sleeps: list[float] = []
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(429 if attempts == 1 else 200, json=[_row(0)])

    candles = asyncio.run(
        _client(httpx.MockTransport(handler), sleeps).fetch_closed(BASE, BASE + timedelta(hours=1))
    )
    assert len(candles) == 1
    assert sleeps == [1.0]


def test_v5_gate_cli_rejects_closed_protocol_before_runtime_dependencies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "gate.csv"
    monkeypatch.setattr(
        "trading_bot.cli._load_non_operational_dependencies",
        lambda: (_ for _ in ()).throw(AssertionError("must not load runtime dependencies")),
    )
    monkeypatch.setattr(
        "trading_bot.data.gate_historical.GateHistoricalDataClient",
        lambda: (_ for _ in ()).throw(AssertionError("must not construct a network client")),
    )

    assert main(["download-protocol-v5-gate-availability", "--output", str(output)]) == 1
    assert "closed" in capsys.readouterr().err
    assert not output.exists()


def test_gate_client_has_no_broker_order_or_credential_dependency() -> None:
    source = inspect.getsource(__import__("trading_bot.data.gate_historical", fromlist=["*"]))
    for forbidden in ("Broker", "Order", "RiskEngine", "api_key", "api_secret", "settings"):
        assert forbidden not in source
