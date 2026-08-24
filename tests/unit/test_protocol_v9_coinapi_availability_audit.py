"""A closed V9 audit must fail before credential or network access."""

from __future__ import annotations

import asyncio
import inspect

import httpx
import pytest

from trading_bot.cli import main
from trading_bot.recommendations import v9_coinapi_availability_audit


def test_closed_v9_rejects_before_network_request() -> None:
    transport = httpx.MockTransport(
        lambda request: (_ for _ in ()).throw(AssertionError("must not request"))
    )
    with pytest.raises(
        v9_coinapi_availability_audit.ProtocolV9AvailabilityAuditError, match="closed"
    ):
        asyncio.run(
            v9_coinapi_availability_audit.audit_protocol_v9_coinapi_historical_availability(
                transport=transport
            )
        )


def test_closed_v9_cli_rejects_before_loading_runtime_dependencies_or_local_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "trading_bot.cli._load_non_operational_dependencies",
        lambda: (_ for _ in ()).throw(AssertionError("must not load runtime dependencies")),
    )
    monkeypatch.setattr(
        "trading_bot.recommendations.v6_access_verification.load_local_coinapi_key",
        lambda: (_ for _ in ()).throw(AssertionError("must not load credential")),
    )
    assert main(["audit-protocol-v9-coinapi-historical-availability"]) == 1
    assert "closed" in capsys.readouterr().err


def test_closed_v9_audit_has_no_broker_order_or_research_execution_dependency() -> None:
    source = inspect.getsource(v9_coinapi_availability_audit)
    for forbidden in (
        "Broker",
        "Order",
        "RiskEngine",
        "DryRunBroker",
        "trading_bot.backtest",
        "recommendation_engine",
    ):
        assert forbidden not in source
