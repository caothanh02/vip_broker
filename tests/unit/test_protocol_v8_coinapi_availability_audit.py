"""Protocol V8 reads only a bounded historical sample in memory."""

from __future__ import annotations

import asyncio
import inspect
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from trading_bot.cli import main
from trading_bot.recommendations import protocol_v8, v8_coinapi_availability_audit

_START = datetime(2020, 1, 1, tzinfo=UTC)
_END = _START + timedelta(hours=3)
_NOW = datetime(2021, 1, 1, tzinfo=UTC)


def _protocol() -> protocol_v8.ProtocolV8:
    return protocol_v8.ProtocolV8(
        protocol_v8.PROTOCOL_V8_ID,
        protocol_v8.PROTOCOL_V8_STATUS,
        _START,
        _END,
        datetime(2025, 1, 1, tzinfo=UTC),
        3,
        2,
        3,
    )


def _identity() -> dict[str, str]:
    return {
        "symbol_id": "BINANCE_SPOT_BTC_USDT",
        "exchange_id": "BINANCE",
        "symbol_type": "SPOT",
        "asset_id_base": "BTC",
        "asset_id_quote": "USDT",
    }


def _candle(open_time: datetime) -> dict[str, str]:
    return {
        "time_period_start": open_time.isoformat().replace("+00:00", "Z"),
        "time_period_end": (open_time + timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
        "time_open": (open_time + timedelta(minutes=1)).isoformat().replace("+00:00", "Z"),
        "time_close": (open_time + timedelta(minutes=59)).isoformat().replace("+00:00", "Z"),
        "price_open": "100",
        "price_high": "110",
        "price_low": "90",
        "price_close": "105",
        "volume_traded": "1",
    }


def test_v8_audit_validates_identity_and_full_range_without_persisting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(v8_coinapi_availability_audit, "load_protocol_v8", _protocol)
    api_key = "test-access-key-never-published"
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        assert request.headers["X-CoinAPI-Key"] == api_key
        assert "api_key" not in str(request.url).lower()
        if requests == 1:
            assert (
                str(request.url)
                == "https://rest.coinapi.io/v1/symbols?filter_symbol_id=BINANCE_SPOT_BTC_USDT"
            )
            return httpx.Response(200, json=[_identity()], request=request)
        assert str(request.url).startswith(
            "https://rest.coinapi.io/v1/ohlcv/BINANCE_SPOT_BTC_USDT/history?"
        )
        assert request.url.params["period_id"] == "1HRS"
        assert request.url.params["time_end"] == "2020-01-01T02:59:59.999999Z"
        assert request.url.params["limit"] == "2"
        if requests == 2:
            assert request.url.params["time_start"] == "2020-01-01T00:00:00Z"
            return httpx.Response(
                200, json=[_candle(_START), _candle(_START + timedelta(hours=1))], request=request
            )
        assert request.url.params["time_start"] == "2020-01-01T02:00:00Z"
        return httpx.Response(200, json=[_candle(_START + timedelta(hours=2))], request=request)

    payload = asyncio.run(
        v8_coinapi_availability_audit.audit_protocol_v8_coinapi_historical_availability(
            api_key, transport=httpx.MockTransport(handler), now=_NOW
        )
    )

    assert requests == 3
    assert payload["identity_verified"] is True
    assert payload["historical_ohlcv_access_verified"] is True
    assert payload["observed_candle_count"] == 3
    assert payload["request_count"] == 3
    assert payload["data_persisted"] is False
    assert payload["strict_oos_authorized"] is False
    assert api_key not in str(payload)


@pytest.mark.parametrize(
    "identity, rows, message",
    [
        ([], [_candle(_START)], "metadata"),
        ([_identity(), _identity()], [_candle(_START)], "metadata"),
        ([{**_identity(), "asset_id_quote": "USD"}], [_candle(_START)], "identity"),
        ([_identity()], [_candle(_START), _candle(_START + timedelta(hours=2))], "validation"),
        ([_identity()], [_candle(_START), _candle(_START)], "duplicate"),
        ([_identity()], {"not": "a candle collection"}, "payload"),
    ],
)
def test_v8_audit_rejects_identity_or_continuity_failure(
    monkeypatch: pytest.MonkeyPatch, identity: object, rows: object, message: str
) -> None:
    monkeypatch.setattr(v8_coinapi_availability_audit, "load_protocol_v8", _protocol)
    responses = [identity, rows]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=responses.pop(0), request=request)

    with pytest.raises(
        v8_coinapi_availability_audit.ProtocolV8AvailabilityAuditError, match=message
    ):
        asyncio.run(
            v8_coinapi_availability_audit.audit_protocol_v8_coinapi_historical_availability(
                "test-access-key-never-published", transport=httpx.MockTransport(handler), now=_NOW
            )
        )


def test_v8_provider_rejection_never_echoes_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(v8_coinapi_availability_audit, "load_protocol_v8", _protocol)
    api_key = "super-secret-must-not-appear"
    transport = httpx.MockTransport(
        lambda request: httpx.Response(401, text=api_key, request=request)
    )
    with pytest.raises(v8_coinapi_availability_audit.ProtocolV8AvailabilityAuditError) as raised:
        asyncio.run(
            v8_coinapi_availability_audit.audit_protocol_v8_coinapi_historical_availability(
                api_key, transport=transport, now=_NOW
            )
        )
    assert api_key not in str(raised.value)


def test_v8_missing_key_fails_before_network_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(v8_coinapi_availability_audit, "load_protocol_v8", _protocol)
    transport = httpx.MockTransport(
        lambda request: (_ for _ in ()).throw(AssertionError("must not request"))
    )
    with pytest.raises(
        v8_coinapi_availability_audit.ProtocolV8AvailabilityAuditError, match="required"
    ):
        asyncio.run(
            v8_coinapi_availability_audit.audit_protocol_v8_coinapi_historical_availability(
                " ", transport=transport, now=_NOW
            )
        )


def test_v8_cli_isolated_from_runtime_dependencies(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "trading_bot.cli._load_non_operational_dependencies",
        lambda: (_ for _ in ()).throw(AssertionError("must not load runtime dependencies")),
    )
    monkeypatch.setattr(
        "trading_bot.recommendations.v6_access_verification.load_local_coinapi_key",
        lambda: "test-access-key-never-published",
    )

    async def audit(_: str) -> dict[str, object]:
        return {"protocol_id": "recommendation_research_v8", "data_persisted": False}

    monkeypatch.setattr(
        "trading_bot.recommendations.v8_coinapi_availability_audit.audit_protocol_v8_coinapi_historical_availability",
        audit,
    )
    assert main(["audit-protocol-v8-coinapi-historical-availability"]) == 0
    output = capsys.readouterr().out
    assert "test-access-key-never-published" not in output
    assert '"data_persisted": false' in output


def test_v8_audit_has_no_broker_order_or_research_execution_dependency() -> None:
    source = inspect.getsource(v8_coinapi_availability_audit)
    for forbidden in (
        "Broker",
        "Order",
        "RiskEngine",
        "DryRunBroker",
        "trading_bot.backtest",
        "recommendation_engine",
    ):
        assert forbidden not in source
