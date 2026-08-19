"""Protocol V7 availability audit is public-only, complete-range, and non-persistent."""

from __future__ import annotations

import asyncio
import inspect
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from trading_bot.cli import main
from trading_bot.domain.models import Candle
from trading_bot.recommendations import v7_rest_availability_audit

START = datetime(2019, 1, 1, tzinfo=UTC)
END = datetime(2022, 1, 1, tzinfo=UTC)


def _candles() -> list[Candle]:
    value = Decimal("100")
    return [
        Candle(
            START + timedelta(hours=index),
            START + timedelta(hours=index + 1),
            "BTC/USDT",
            "1h",
            value,
            value,
            value,
            value,
            Decimal("1"),
            True,
        )
        for index in range(26304)
    ]


def test_v7_audit_requires_complete_closed_continuous_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    class Client:
        def __init__(self, **kwargs: object) -> None:
            observed.update(kwargs)

        async def fetch_closed(self, start: datetime, end: datetime) -> list[Candle]:
            assert (start, end) == (START, END)
            return _candles()

    monkeypatch.setattr(v7_rest_availability_audit, "BinanceHistoricalDataClient", Client)
    payload = asyncio.run(v7_rest_availability_audit.audit_protocol_v7_binance_rest_availability())

    assert observed["base_url"] == "https://api.binance.com/api/v3"
    assert observed["page_limit"] == 1000
    assert payload["observed_candle_count"] == 26304
    assert payload["maximum_request_count"] == 27
    assert payload["data_persisted"] is False
    assert payload["performance_metric_computed"] is False
    assert payload["strict_oos_authorized"] is False


def test_v7_audit_rejects_incomplete_range(monkeypatch: pytest.MonkeyPatch) -> None:
    class Client:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        async def fetch_closed(self, start: datetime, end: datetime) -> list[Candle]:
            del start, end
            return _candles()[:-1]

    monkeypatch.setattr(v7_rest_availability_audit, "BinanceHistoricalDataClient", Client)
    with pytest.raises(v7_rest_availability_audit.ProtocolV7AvailabilityAuditError):
        asyncio.run(v7_rest_availability_audit.audit_protocol_v7_binance_rest_availability())


@pytest.mark.parametrize(
    "candles",
    [
        [
            Candle(
                START,
                START + timedelta(hours=1),
                "BTC/USDT",
                "1h",
                Decimal("100"),
                Decimal("100"),
                Decimal("100"),
                Decimal("100"),
                Decimal("1"),
                False,
            )
        ],
        [
            Candle(
                START,
                START + timedelta(hours=1),
                "BTC/USDT",
                "1h",
                Decimal("100"),
                Decimal("100"),
                Decimal("100"),
                Decimal("100"),
                Decimal("1"),
                True,
            ),
            Candle(
                START + timedelta(hours=2),
                START + timedelta(hours=3),
                "BTC/USDT",
                "1h",
                Decimal("100"),
                Decimal("100"),
                Decimal("100"),
                Decimal("100"),
                Decimal("1"),
                True,
            ),
        ],
    ],
)
def test_v7_audit_rejects_open_or_gapped_candles(
    monkeypatch: pytest.MonkeyPatch, candles: list[Candle]
) -> None:
    class Client:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        async def fetch_closed(self, start: datetime, end: datetime) -> list[Candle]:
            del start, end
            return candles

    monkeypatch.setattr(v7_rest_availability_audit, "BinanceHistoricalDataClient", Client)
    with pytest.raises(
        v7_rest_availability_audit.ProtocolV7AvailabilityAuditError, match="validation"
    ):
        asyncio.run(v7_rest_availability_audit.audit_protocol_v7_binance_rest_availability())


def test_v7_cli_does_not_load_runtime_dependencies(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "trading_bot.cli._load_non_operational_dependencies",
        lambda: (_ for _ in ()).throw(AssertionError("must not load runtime dependencies")),
    )

    async def audit() -> dict[str, object]:
        return {"data_persisted": False, "strict_oos_authorized": False}

    monkeypatch.setattr(
        "trading_bot.recommendations.v7_rest_availability_audit.audit_protocol_v7_binance_rest_availability",
        audit,
    )
    assert main(["audit-protocol-v7-binance-rest-availability"]) == 0
    assert '"data_persisted": false' in capsys.readouterr().out


def test_v7_audit_has_no_broker_order_or_research_execution_dependency() -> None:
    source = inspect.getsource(v7_rest_availability_audit)
    for forbidden in (
        "Broker",
        "Order",
        "RiskEngine",
        "DryRunBroker",
        "trading_bot.backtest",
        "recommendation_engine",
    ):
        assert forbidden not in source
