"""Closed Protocol V7 public Binance REST availability-audit command.

V7's prior in-memory audit did not meet its fixed validation contract. This
command is retained only to reject a retry before constructing a network client.
"""

from __future__ import annotations

import math

import httpx

from trading_bot.data.binance_historical import (
    BINANCE_REST,
    BinanceDataError,
    BinanceHistoricalDataClient,
)
from trading_bot.data.validation import CandleValidationError, validate_candles
from trading_bot.recommendations.protocol_v7 import (
    ProtocolV7Error,
    load_protocol_v7,
    require_protocol_v7_availability_audit,
)


class ProtocolV7AvailabilityAuditError(ValueError):
    """The fixed V7 public-source availability audit could not complete safely."""


async def audit_protocol_v7_binance_rest_availability(
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, object]:
    """Fetch and validate V7's fixed range in memory without persisting it."""

    protocol = load_protocol_v7()
    try:
        require_protocol_v7_availability_audit(protocol)
    except ProtocolV7Error as exc:
        raise ProtocolV7AvailabilityAuditError(str(exc)) from exc

    client = BinanceHistoricalDataClient(
        base_url=BINANCE_REST,
        page_limit=1000,
        max_retries=3,
        transport=transport,
    )
    try:
        candles = await client.fetch_closed(protocol.availability_start, protocol.availability_end)
        validate_candles(candles)
    except (BinanceDataError, CandleValidationError) as exc:
        raise ProtocolV7AvailabilityAuditError("V7 public REST candles failed validation") from exc

    if (
        len(candles) != protocol.expected_candle_count
        or candles[0].open_time != protocol.availability_start
        or candles[-1].close_time != protocol.availability_end
    ):
        raise ProtocolV7AvailabilityAuditError("V7 public REST candles are incomplete")

    return {
        "schema_version": "1.0",
        "protocol_id": protocol.protocol_id,
        "protocol_status": protocol.status,
        "audit_kind": "full_range_public_rest_availability_only",
        "provider": "binance_spot_public_rest",
        "endpoint": f"{BINANCE_REST}/klines",
        "authentication": "none",
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "utc_range": {
            "start": protocol.availability_start.isoformat().replace("+00:00", "Z"),
            "end": protocol.availability_end.isoformat().replace("+00:00", "Z"),
        },
        "expected_candle_count": protocol.expected_candle_count,
        "observed_candle_count": len(candles),
        "request_page_candles": 1000,
        "maximum_request_count": math.ceil(protocol.expected_candle_count / 1000),
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
