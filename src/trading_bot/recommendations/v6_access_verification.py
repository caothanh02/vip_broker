"""Bounded authenticated CoinAPI identity verification for Protocol V6.

The only network action is one filtered metadata-collection request for the
preregistered symbol.
It does not request OHLCV, persist data, build features, or authorize research.
"""

from __future__ import annotations

import httpx

from trading_bot.recommendations.protocol_v6 import (
    PROTOCOL_V6_ID,
    ProtocolV6Error,
    load_protocol_v6,
    require_protocol_v6_access_verification,
)

_COINAPI_SYMBOLS_URL = "https://rest.coinapi.io/v1/symbols"
_EXPECTED_IDENTITY = {
    "symbol_id": "BINANCE_SPOT_BTC_USDT",
    "exchange_id": "BINANCE",
    "symbol_type": "SPOT",
    "asset_id_base": "BTC",
    "asset_id_quote": "USDT",
}


class ProtocolV6AccessVerificationError(ValueError):
    """The bounded V6 access check could not prove authenticated identity."""


def load_local_coinapi_key() -> str:
    """Read the local-only credential without returning it in any payload."""

    from trading_bot.settings import BotSettings

    api_key = BotSettings().coinapi_api_key.get_secret_value().strip()
    if not api_key:
        raise ProtocolV6AccessVerificationError(
            "COINAPI_API_KEY is required for V6 access verification"
        )
    return api_key


def _validate_symbol_collection(payload: object) -> None:
    if not isinstance(payload, list) or len(payload) != 1:
        raise ProtocolV6AccessVerificationError("CoinAPI symbol metadata is invalid")
    symbol = payload[0]
    if not isinstance(symbol, dict):
        raise ProtocolV6AccessVerificationError("CoinAPI symbol metadata is invalid")
    for field, expected in _EXPECTED_IDENTITY.items():
        if symbol.get(field) != expected:
            raise ProtocolV6AccessVerificationError(
                "CoinAPI symbol identity does not match Protocol V6"
            )


async def verify_protocol_v6_coinapi_access(
    api_key: str,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, object]:
    """Verify one authenticated filtered-symbol response without reading OHLCV."""

    protocol = load_protocol_v6()
    try:
        require_protocol_v6_access_verification(protocol)
    except ProtocolV6Error as exc:
        raise ProtocolV6AccessVerificationError(str(exc)) from exc
    if not api_key.strip():
        raise ProtocolV6AccessVerificationError(
            "COINAPI_API_KEY is required for V6 access verification"
        )

    try:
        async with httpx.AsyncClient(timeout=20, transport=transport) as client:
            response = await client.get(
                _COINAPI_SYMBOLS_URL,
                params={"filter_symbol_id": _EXPECTED_IDENTITY["symbol_id"]},
                headers={"Accept": "application/json", "X-CoinAPI-Key": api_key},
            )
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise ProtocolV6AccessVerificationError(
            "CoinAPI rejected the V6 access-verification request"
        ) from exc
    except httpx.HTTPError as exc:
        raise ProtocolV6AccessVerificationError(
            "CoinAPI access-verification request failed"
        ) from exc

    try:
        _validate_symbol_collection(response.json())
    except ValueError as exc:
        raise ProtocolV6AccessVerificationError("CoinAPI symbol metadata is invalid") from exc

    return {
        "schema_version": "1.0",
        "protocol_id": PROTOCOL_V6_ID,
        "protocol_status": protocol.status,
        "verification_kind": "authenticated_filtered_symbol_collection_identity_only",
        "provider": "coinapi_historical_ohlcv",
        "symbol_id": _EXPECTED_IDENTITY["symbol_id"],
        "symbol_identity_verified": True,
        "historical_ohlcv_entitlement_verified": False,
        "historical_ohlcv_requested": False,
        "data_persisted": False,
        "candidate_or_parameter_used": False,
        "recommendation_or_backtest_run": False,
        "strict_oos_authorized": False,
        "safety_locks": {
            "default_recommendation": "NEUTRAL",
            "broker_used": False,
            "orders_submitted": False,
            "risk_engine_used": False,
            "dry_run_broker_used": False,
            "ml_used": False,
            "network_used": True,
        },
    }
