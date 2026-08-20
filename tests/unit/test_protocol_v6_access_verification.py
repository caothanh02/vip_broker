"""Protocol V6 may verify one authenticated symbol identity, and nothing more."""

from __future__ import annotations

import asyncio
import inspect

import httpx
import pytest

from trading_bot.cli import main
from trading_bot.recommendations import v6_access_verification


def _symbol_metadata() -> dict[str, str]:
    return {
        "symbol_id": "BINANCE_SPOT_BTC_USDT",
        "exchange_id": "BINANCE",
        "symbol_type": "SPOT",
        "asset_id_base": "BTC",
        "asset_id_quote": "USDT",
    }


def test_access_verification_uses_only_fixed_filtered_symbol_metadata_endpoint() -> None:
    api_key = "test-access-key-never-published"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert str(request.url) == (
            "https://rest.coinapi.io/v1/symbols?filter_symbol_id=BINANCE_SPOT_BTC_USDT"
        )
        assert request.headers["X-CoinAPI-Key"] == api_key
        assert "api_key" not in str(request.url).lower()
        return httpx.Response(200, json=[_symbol_metadata()], request=request)

    payload = asyncio.run(
        v6_access_verification.verify_protocol_v6_coinapi_access(
            api_key, transport=httpx.MockTransport(handler)
        )
    )

    assert payload["symbol_identity_verified"] is True
    assert payload["historical_ohlcv_entitlement_verified"] is False
    assert payload["historical_ohlcv_requested"] is False
    assert payload["data_persisted"] is False
    assert api_key not in str(payload)


def test_missing_key_fails_before_network_request() -> None:
    transport = httpx.MockTransport(
        lambda request: (_ for _ in ()).throw(AssertionError("must not request"))
    )
    with pytest.raises(v6_access_verification.ProtocolV6AccessVerificationError, match="required"):
        asyncio.run(
            v6_access_verification.verify_protocol_v6_coinapi_access("  ", transport=transport)
        )


@pytest.mark.parametrize(
    "payload",
    [
        [],
        [{**_symbol_metadata(), "asset_id_quote": "USD"}],
        [_symbol_metadata(), _symbol_metadata()],
        ["not", "an", "object"],
    ],
)
def test_unexpected_symbol_identity_is_rejected(payload: object) -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json=payload, request=request)
    )
    with pytest.raises(
        v6_access_verification.ProtocolV6AccessVerificationError, match="identity|invalid"
    ):
        asyncio.run(
            v6_access_verification.verify_protocol_v6_coinapi_access(
                "test-access-key-never-published", transport=transport
            )
        )


def test_provider_rejection_never_echoes_credential() -> None:
    api_key = "super-secret-must-not-appear"
    transport = httpx.MockTransport(
        lambda request: httpx.Response(401, text=api_key, request=request)
    )
    with pytest.raises(v6_access_verification.ProtocolV6AccessVerificationError) as raised:
        asyncio.run(
            v6_access_verification.verify_protocol_v6_coinapi_access(api_key, transport=transport)
        )
    assert api_key not in str(raised.value)


def test_cli_isolated_verifier_does_not_load_runtime_dependencies(
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

    async def verify(_: str) -> dict[str, object]:
        return {
            "protocol_id": "recommendation_research_v6",
            "symbol_identity_verified": True,
            "historical_ohlcv_requested": False,
        }

    monkeypatch.setattr(
        "trading_bot.recommendations.v6_access_verification.verify_protocol_v6_coinapi_access",
        verify,
    )
    assert main(["verify-protocol-v6-coinapi-access"]) == 0
    output = capsys.readouterr().out
    assert "test-access-key-never-published" not in output
    assert "historical_ohlcv_requested" in output


def test_access_module_has_no_broker_order_or_research_execution_dependency() -> None:
    source = inspect.getsource(v6_access_verification)
    for forbidden in (
        "Broker",
        "Order",
        "RiskEngine",
        "DryRunBroker",
        "recommendation_engine",
        "trading_bot.backtest",
        "historical_ohlcv/history",
    ):
        assert forbidden not in source
