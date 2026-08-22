"""Fail-closed CoinAPI paid-entitlement preregistration for Protocol V9.

V9 is a pure governance record. It has no credential, network, market-data,
candidate, broker, order, ML, or OOS runtime dependency. A later, separately
reviewed availability-audit implementation must call its narrow authorization.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml

PROTOCOL_V9_ID = "recommendation_research_v9"
PROTOCOL_V9_SCHEMA_VERSION = "1.0"
PROTOCOL_V9_STATUS = "source_selected_availability_audit_authorized"


class ProtocolV9Error(ValueError):
    """Protocol V9 configuration is incomplete or an unsafe action was requested."""


@dataclass(frozen=True, slots=True)
class ProtocolV9:
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
        """V9 never authorizes research or trading execution."""

        return False


_EXPECTED: dict[str, object] = {
    "schema_version": PROTOCOL_V9_SCHEMA_VERSION,
    "protocol_id": PROTOCOL_V9_ID,
    "status": PROTOCOL_V9_STATUS,
    "source_selection": {
        "selection_basis": "license_provenance_paid_entitlement_and_mechanical_availability_only",
        "required_facts": [
            "license_or_terms_verified",
            "source_provenance_verified",
            "paid_entitlement_context",
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
    "entitlement_context": {
        "basis": "separately_confirmed_paid_coinapi_credit_balance_after_v8_closure",
        "credential_name": "COINAPI_API_KEY",
        "historical_access_verified": False,
    },
    "availability_audit": {
        "utc_range": {"start": "2019-01-01T00:00:00Z", "end": "2022-01-01T00:00:00Z"},
        "request_page_candles": 1000,
        "expected_candle_count": 26304,
        "maximum_request_count": 28,
        "data_persistence_authorized": False,
        "candidate_or_parameter_authorized": False,
        "recommendation_or_backtest_authorized": False,
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
        raise ProtocolV9Error(f"{label} must be a UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProtocolV9Error(f"{label} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ProtocolV9Error(f"{label} must be UTC")
    return parsed.astimezone(UTC)


def validate_protocol_v9(raw: object) -> ProtocolV9:
    """Validate V9's immutable, availability-only governance record."""

    if not isinstance(raw, dict) or raw != _EXPECTED:
        raise ProtocolV9Error("protocol v9 differs from its immutable preregistration")
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
        raise ProtocolV9Error("availability audit count configuration is invalid")
    expected_pages = (count + page - 1) // page
    if start >= end or end > strict_oos_start or (end - start) // timedelta(hours=1) != count:
        raise ProtocolV9Error("availability audit range is invalid")
    if maximum_requests != 1 + expected_pages:
        raise ProtocolV9Error("availability audit request bound is invalid")
    return ProtocolV9(
        PROTOCOL_V9_ID,
        PROTOCOL_V9_STATUS,
        start,
        end,
        strict_oos_start,
        count,
        page,
        maximum_requests,
    )


def load_protocol_v9(path: Path = Path("config/recommendation_protocol_v9.yaml")) -> ProtocolV9:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ProtocolV9Error("could not read protocol v9 preregistration") from exc
    return validate_protocol_v9(raw)


def require_protocol_v9_availability_audit(protocol: ProtocolV9) -> None:
    """Authorize only a separately implemented bounded availability audit."""

    if protocol.status != PROTOCOL_V9_STATUS:
        raise ProtocolV9Error("protocol v9 availability audit is not authorized")


def require_protocol_v9_blocked(protocol: ProtocolV9, action: str) -> None:
    """Fail closed for every action other than the preregistered audit."""

    del protocol
    raise ProtocolV9Error(f"protocol v9 cannot {action}")
