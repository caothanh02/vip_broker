"""Bounded in-memory authenticated CoinAPI availability audit for Protocol V8.

It verifies the fixed symbol identity and full preregistered historical range,
but never persists candles or authorizes research, recommendations, or OOS.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation

import httpx

from trading_bot.data.validation import CandleValidationError, validate_candles
from trading_bot.domain.models import Candle
from trading_bot.recommendations.protocol_v8 import (
    ProtocolV8Error,
    load_protocol_v8,
    require_protocol_v8_availability_audit,
)

_IDENTITY_URL = "https://rest.coinapi.io/v1/symbols"
_HISTORY_URL = "https://rest.coinapi.io/v1/ohlcv/BINANCE_SPOT_BTC_USDT/history"
_EXPECTED_IDENTITY = {
    "symbol_id": "BINANCE_SPOT_BTC_USDT",
    "exchange_id": "BINANCE",
    "symbol_type": "SPOT",
    "asset_id_base": "BTC",
    "asset_id_quote": "USDT",
}
_INTERVAL = timedelta(hours=1)


class ProtocolV8AvailabilityAuditError(ValueError):
    """The bounded V8 historical availability audit could not complete safely."""


def _utc_timestamp(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise ProtocolV8AvailabilityAuditError(f"CoinAPI {label} is invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProtocolV8AvailabilityAuditError(f"CoinAPI {label} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ProtocolV8AvailabilityAuditError(f"CoinAPI {label} must be UTC")
    return parsed.astimezone(UTC)


def _decimal(value: object, label: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ProtocolV8AvailabilityAuditError(f"CoinAPI {label} is invalid")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ProtocolV8AvailabilityAuditError(f"CoinAPI {label} is invalid") from exc
    if not parsed.is_finite():
        raise ProtocolV8AvailabilityAuditError(f"CoinAPI {label} is invalid")
    return parsed


def _validate_identity(payload: object) -> None:
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
        raise ProtocolV8AvailabilityAuditError("CoinAPI symbol metadata is invalid")
    symbol = payload[0]
    for field, expected in _EXPECTED_IDENTITY.items():
        if symbol.get(field) != expected:
            raise ProtocolV8AvailabilityAuditError(
                "CoinAPI symbol identity does not match Protocol V8"
            )


def _parse_candle(raw: object, *, now: datetime) -> Candle:
    if not isinstance(raw, dict):
        raise ProtocolV8AvailabilityAuditError("CoinAPI historical payload is invalid")
    start = _utc_timestamp(raw.get("time_period_start"), "time_period_start")
    end = _utc_timestamp(raw.get("time_period_end"), "time_period_end")
    first_trade = _utc_timestamp(raw.get("time_open"), "time_open")
    last_trade = _utc_timestamp(raw.get("time_close"), "time_close")
    if start.minute or start.second or start.microsecond or end - start != _INTERVAL:
        raise ProtocolV8AvailabilityAuditError("CoinAPI historical candle interval is invalid")
    if not start <= first_trade <= last_trade < end:
        raise ProtocolV8AvailabilityAuditError("CoinAPI historical trade timestamps are invalid")
    if end > now:
        raise ProtocolV8AvailabilityAuditError("CoinAPI returned an open candle")
    return Candle(
        open_time=start,
        close_time=end,
        symbol="BTC/USDT",
        timeframe="1h",
        open=_decimal(raw.get("price_open"), "price_open"),
        high=_decimal(raw.get("price_high"), "price_high"),
        low=_decimal(raw.get("price_low"), "price_low"),
        close=_decimal(raw.get("price_close"), "price_close"),
        volume=_decimal(raw.get("volume_traded"), "volume_traded"),
        is_closed=True,
    )


def _iso8601(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


async def _get_json(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: dict[str, str | int],
    api_key: str,
) -> object:
    try:
        response = await client.get(
            url,
            params=params,
            headers={"Accept": "application/json", "X-CoinAPI-Key": api_key},
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise ProtocolV8AvailabilityAuditError(
            "CoinAPI rejected the V8 availability-audit request"
        ) from exc
    except httpx.HTTPError as exc:
        raise ProtocolV8AvailabilityAuditError("CoinAPI availability-audit request failed") from exc
    try:
        return response.json()
    except ValueError as exc:
        raise ProtocolV8AvailabilityAuditError("CoinAPI response is not valid JSON") from exc


async def audit_protocol_v8_coinapi_historical_availability(
    api_key: str | None = None,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    now: datetime | None = None,
) -> dict[str, object]:
    """Audit V8's fixed CoinAPI range in memory without persisting OHLCV."""

    protocol = load_protocol_v8()
    try:
        require_protocol_v8_availability_audit(protocol)
    except ProtocolV8Error as exc:
        raise ProtocolV8AvailabilityAuditError(str(exc)) from exc
    if api_key is None:
        from trading_bot.recommendations.v6_access_verification import load_local_coinapi_key

        api_key = load_local_coinapi_key()
    if not api_key.strip():
        raise ProtocolV8AvailabilityAuditError(
            "COINAPI_API_KEY is required for V8 availability audit"
        )
    clock = now or datetime.now(UTC)
    if clock.tzinfo is None or clock.utcoffset() is None:
        raise ProtocolV8AvailabilityAuditError("audit clock must be timezone-aware")
    audit_now = clock.astimezone(UTC)

    request_count = 0
    candles: list[Candle] = []
    seen_open_times: set[datetime] = set()
    cursor = protocol.availability_start
    exclusive_end = protocol.availability_end - timedelta(microseconds=1)
    try:
        async with httpx.AsyncClient(timeout=20, transport=transport) as client:
            request_count += 1
            identity = await _get_json(
                client,
                _IDENTITY_URL,
                params={"filter_symbol_id": _EXPECTED_IDENTITY["symbol_id"]},
                api_key=api_key,
            )
            _validate_identity(identity)

            while cursor < protocol.availability_end:
                if request_count >= protocol.maximum_request_count:
                    raise ProtocolV8AvailabilityAuditError("CoinAPI request bound was exceeded")
                request_count += 1
                payload = await _get_json(
                    client,
                    _HISTORY_URL,
                    params={
                        "period_id": "1HRS",
                        "time_start": _iso8601(cursor),
                        "time_end": _iso8601(exclusive_end),
                        "limit": protocol.request_page_candles,
                    },
                    api_key=api_key,
                )
                if not isinstance(payload, list) or len(payload) > protocol.request_page_candles:
                    raise ProtocolV8AvailabilityAuditError("CoinAPI historical payload is invalid")
                if not payload:
                    break
                previous_open: datetime | None = None
                for raw in payload:
                    candle = _parse_candle(raw, now=audit_now)
                    if not cursor <= candle.open_time < protocol.availability_end:
                        raise ProtocolV8AvailabilityAuditError(
                            "CoinAPI historical candle is outside the preregistered range"
                        )
                    if candle.open_time in seen_open_times:
                        raise ProtocolV8AvailabilityAuditError(
                            "CoinAPI historical payload has a duplicate timestamp"
                        )
                    if previous_open is not None and candle.open_time <= previous_open:
                        raise ProtocolV8AvailabilityAuditError(
                            "CoinAPI historical payload is not ordered"
                        )
                    seen_open_times.add(candle.open_time)
                    candles.append(candle)
                    previous_open = candle.open_time
                next_cursor = candles[-1].open_time + _INTERVAL
                if next_cursor <= cursor:
                    raise ProtocolV8AvailabilityAuditError(
                        "CoinAPI pagination made no forward progress"
                    )
                cursor = next_cursor
    except CandleValidationError as exc:
        raise ProtocolV8AvailabilityAuditError(
            "CoinAPI historical candles failed validation"
        ) from exc

    try:
        validate_candles(candles)
    except CandleValidationError as exc:
        raise ProtocolV8AvailabilityAuditError(
            "CoinAPI historical candles failed validation"
        ) from exc
    if (
        len(candles) != protocol.expected_candle_count
        or candles[0].open_time != protocol.availability_start
        or candles[-1].close_time != protocol.availability_end
    ):
        raise ProtocolV8AvailabilityAuditError("CoinAPI historical candles are incomplete")

    return {
        "schema_version": "1.0",
        "protocol_id": protocol.protocol_id,
        "protocol_status": protocol.status,
        "audit_kind": "authenticated_full_range_historical_availability_only",
        "provider": "coinapi_historical_ohlcv",
        "identity_verified": True,
        "historical_ohlcv_access_verified": True,
        "license_or_terms_verified": False,
        "symbol_id": _EXPECTED_IDENTITY["symbol_id"],
        "timeframe": "1h",
        "utc_range": {
            "start": _iso8601(protocol.availability_start),
            "end": _iso8601(protocol.availability_end),
        },
        "expected_candle_count": protocol.expected_candle_count,
        "observed_candle_count": len(candles),
        "request_page_candles": protocol.request_page_candles,
        "request_count": request_count,
        "maximum_request_count": protocol.maximum_request_count,
        "absolute_continuity": True,
        "result": "availability_verified_not_selected",
        "data_persisted": False,
        "candidate_or_parameter_used": False,
        "signal_or_feature_computed": False,
        "performance_metric_computed": False,
        "selection_authorized": False,
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
