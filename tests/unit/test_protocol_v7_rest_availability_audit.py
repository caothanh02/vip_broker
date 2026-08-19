"""A closed V7 audit must fail before any public request or runtime import."""

from __future__ import annotations

import asyncio
import inspect

import httpx
import pytest

from trading_bot.cli import main
from trading_bot.recommendations import v7_rest_availability_audit


def test_closed_v7_rejects_before_public_request() -> None:
    transport = httpx.MockTransport(
        lambda request: (_ for _ in ()).throw(AssertionError("must not request"))
    )
    with pytest.raises(v7_rest_availability_audit.ProtocolV7AvailabilityAuditError, match="closed"):
        asyncio.run(
            v7_rest_availability_audit.audit_protocol_v7_binance_rest_availability(
                transport=transport
            )
        )


def test_closed_v7_cli_rejects_before_runtime_or_network_client(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "trading_bot.cli._load_non_operational_dependencies",
        lambda: (_ for _ in ()).throw(AssertionError("must not load runtime dependencies")),
    )
    monkeypatch.setattr(
        "trading_bot.recommendations.v7_rest_availability_audit.BinanceHistoricalDataClient",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("must not construct a client")),
    )
    assert main(["audit-protocol-v7-binance-rest-availability"]) == 1
    assert "closed" in capsys.readouterr().err


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
