"""Immutable closure record for Protocol V8.

The one authorized CoinAPI availability audit was rejected by the provider.
V8 now permits no credential access, network retry, persistence, research or OOS.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml

PROTOCOL_V8_ID = "recommendation_research_v8"
PROTOCOL_V8_SCHEMA_VERSION = "1.1"
PROTOCOL_V8_CLOSED_STATUS = "closed_input_unavailable"


class ProtocolV8Error(ValueError):
    """Protocol V8 configuration is incomplete or an unsafe action was requested."""


@dataclass(frozen=True, slots=True)
class ProtocolV8:
    protocol_id: str
    status: str
    availability_start: datetime
    availability_end: datetime
    strict_oos_start: datetime
    expected_candle_count: int
    request_page_candles: int
    maximum_request_count: int

    @property
    def executable(self) -> bool:
        return False


_EXPECTED: dict[str, object] = {
    "schema_version": PROTOCOL_V8_SCHEMA_VERSION,
    "protocol_id": PROTOCOL_V8_ID,
    "status": PROTOCOL_V8_CLOSED_STATUS,
    "source_selection": {
        "selection_basis": (
            "license_provenance_authenticated_identity_and_mechanical_availability_only"
        ),
        "required_facts": [
            "license_or_terms_verified",
            "source_provenance_verified",
            "authenticated_symbol_identity",
            "authenticated_historical_read_access",
            "btc_usdt_spot_1h_utc",
            "closed_candles_only",
            "absolute_continuity",
        ],
        "prohibited_selection_inputs": [
            "signal",
            "return",
            "accuracy",
            "backtest",
            "pnl",
            "performance_metric",
        ],
    },
    "data_source": {
        "provider": "coinapi_historical_ohlcv",
        "identity_endpoint": "https://rest.coinapi.io/v1/symbols?filter_symbol_id=BINANCE_SPOT_BTC_USDT",
        "historical_endpoint": "https://rest.coinapi.io/v1/ohlcv/BINANCE_SPOT_BTC_USDT/history",
        "exchange_symbol": "BINANCE_SPOT_BTC_USDT",
        "internal_symbol": "BTC/USDT",
        "market_type": "spot",
        "timeframe": "1h",
        "period_id": "1HRS",
        "timezone": "UTC",
        "authentication": "local_env_key_for_availability_audit_only",
    },
    "availability_audit": {
        "utc_range": {"start": "2019-01-01T00:00:00Z", "end": "2022-01-01T00:00:00Z"},
        "request_page_candles": 1000,
        "expected_candle_count": 26304,
        "maximum_request_count": 28,
        "outcome": "failed_provider_rejection",
        "observed_at": "2026-08-20T06:06:10Z",
        "data_persistence_authorized": False,
        "candidate_or_parameter_authorized": False,
        "recommendation_or_backtest_authorized": False,
        "strict_oos_authorized": False,
    },
    "closure": {
        "reason": "authenticated_coinapi_availability_audit_request_rejected",
        "audit_retry_authorized": False,
        "input_freeze_authorized": False,
        "execution_authorized": False,
        "selection_authorized": False,
        "strict_oos_authorized": False,
    },
    "candidate": None,
    "parameters": None,
    "development_dataset_range": None,
    "required_input_lock": None,
    "selection_artifact": None,
    "strict_oos": {
        "start": "2025-01-01T00:00:00Z",
        "status": "sealed",
        "evaluation_authorized": False,
    },
    "safety_locks": {
        "live_trading_enabled": False,
        "broker_used": False,
        "orders_submitted": False,
        "risk_engine_used": False,
        "dry_run_broker_used": False,
        "ml_used": False,
        "network_used": False,
    },
}


def _utc(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise ProtocolV8Error(f"{label} must be a UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProtocolV8Error(f"{label} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ProtocolV8Error(f"{label} must be UTC")
    return parsed.astimezone(UTC)


def validate_protocol_v8(raw: object) -> ProtocolV8:
    """Validate the exact V8 availability-audit governance without reading data."""

    if not isinstance(raw, dict) or raw != _EXPECTED:
        raise ProtocolV8Error(
            "protocol v8 differs from its immutable availability-audit governance"
        )
    audit = raw["availability_audit"]
    strict_oos = raw["strict_oos"]
    assert isinstance(audit, dict) and isinstance(strict_oos, dict)
    utc_range = audit["utc_range"]
    assert isinstance(utc_range, dict)
    start = _utc(utc_range["start"], "availability start")
    end = _utc(utc_range["end"], "availability end")
    strict_oos_start = _utc(strict_oos["start"], "strict OOS start")
    count = audit["expected_candle_count"]
    page = audit["request_page_candles"]
    maximum_requests = audit["maximum_request_count"]
    if not all(isinstance(value, int) and value > 0 for value in (count, page, maximum_requests)):
        raise ProtocolV8Error("availability audit count configuration is invalid")
    expected_pages = (count + page - 1) // page
    if start >= end or end > strict_oos_start or (end - start) // timedelta(hours=1) != count:
        raise ProtocolV8Error("availability audit range is invalid")
    if maximum_requests != 1 + expected_pages:
        raise ProtocolV8Error("availability audit request bound is invalid")
    return ProtocolV8(
        PROTOCOL_V8_ID,
        PROTOCOL_V8_CLOSED_STATUS,
        start,
        end,
        strict_oos_start,
        count,
        page,
        maximum_requests,
    )


def load_protocol_v8(path: Path = Path("config/recommendation_protocol_v8.yaml")) -> ProtocolV8:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ProtocolV8Error("could not read protocol v8 availability-audit governance") from exc
    return validate_protocol_v8(raw)


def require_protocol_v8_availability_audit(protocol: ProtocolV8) -> None:
    del protocol
    raise ProtocolV8Error("protocol v8 is closed and cannot audit an input source")


def _blocked(protocol: ProtocolV8, action: str) -> None:
    del protocol
    raise ProtocolV8Error(f"protocol v8 cannot {action}")


def require_protocol_v8_input_freeze(protocol: ProtocolV8) -> None:
    _blocked(protocol, "freeze an input")


def require_protocol_v8_execution(protocol: ProtocolV8) -> None:
    _blocked(protocol, "be executed")


def require_protocol_v8_selection(protocol: ProtocolV8) -> None:
    _blocked(protocol, "select a policy")


def require_protocol_v8_oos_authorization(protocol: ProtocolV8) -> None:
    _blocked(protocol, "authorize strict OOS")
