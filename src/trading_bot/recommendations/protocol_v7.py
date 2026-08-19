"""Immutable public-REST availability governance for Protocol V7.

V7 permits only a mechanical audit of one fixed historical range. It does not
authorize persistence, research execution, policy selection, or strict OOS.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml

PROTOCOL_V7_ID = "recommendation_research_v7"
PROTOCOL_V7_SCHEMA_VERSION = "1.0"
PROTOCOL_V7_SOURCE_AUDIT_STATUS = "source_selected_availability_audit_authorized"


class ProtocolV7Error(ValueError):
    """Protocol V7 configuration is incomplete or an unsafe action was requested."""


@dataclass(frozen=True, slots=True)
class ProtocolV7:
    protocol_id: str
    status: str
    availability_start: datetime
    availability_end: datetime
    strict_oos_start: datetime
    expected_candle_count: int

    @property
    def executable(self) -> bool:
        return False


_EXPECTED: dict[str, object] = {
    "schema_version": PROTOCOL_V7_SCHEMA_VERSION,
    "protocol_id": PROTOCOL_V7_ID,
    "status": PROTOCOL_V7_SOURCE_AUDIT_STATUS,
    "source_selection": {
        "selection_basis": "license_provenance_and_mechanical_availability_only",
        "required_facts": [
            "public_unauthenticated_endpoint",
            "source_provenance_verified",
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
        "provider": "binance_spot_public_rest",
        "endpoint": "https://api.binance.com/api/v3/klines",
        "exchange_symbol": "BTCUSDT",
        "internal_symbol": "BTC/USDT",
        "market_type": "spot",
        "timeframe": "1h",
        "timezone": "UTC",
        "authentication": "none",
    },
    "availability_audit": {
        "utc_range": {"start": "2019-01-01T00:00:00Z", "end": "2022-01-01T00:00:00Z"},
        "request_page_candles": 1000,
        "expected_candle_count": 26304,
        "outcome": "pending",
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
        raise ProtocolV7Error(f"{label} must be a UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProtocolV7Error(f"{label} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ProtocolV7Error(f"{label} must be UTC")
    return parsed.astimezone(UTC)


def validate_protocol_v7(raw: object) -> ProtocolV7:
    """Validate the exact V7 source-audit governance without reading market data."""

    if not isinstance(raw, dict) or raw != _EXPECTED:
        raise ProtocolV7Error("protocol v7 differs from its immutable source-audit governance")
    audit = raw["availability_audit"]
    strict_oos = raw["strict_oos"]
    assert isinstance(audit, dict) and isinstance(strict_oos, dict)
    utc_range = audit["utc_range"]
    assert isinstance(utc_range, dict)
    start = _utc(utc_range["start"], "availability start")
    end = _utc(utc_range["end"], "availability end")
    oos_start = _utc(strict_oos["start"], "strict OOS start")
    count = audit["expected_candle_count"]
    if not isinstance(count, int) or count <= 0:
        raise ProtocolV7Error("expected candle count is invalid")
    if start >= end or end > oos_start or (end - start) // timedelta(hours=1) != count:
        raise ProtocolV7Error("availability audit range is invalid")
    return ProtocolV7(PROTOCOL_V7_ID, PROTOCOL_V7_SOURCE_AUDIT_STATUS, start, end, oos_start, count)


def load_protocol_v7(path: Path = Path("config/recommendation_protocol_v7.yaml")) -> ProtocolV7:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ProtocolV7Error("could not read protocol v7 source-audit governance") from exc
    return validate_protocol_v7(raw)


def require_protocol_v7_availability_audit(protocol: ProtocolV7) -> None:
    if protocol.status != PROTOCOL_V7_SOURCE_AUDIT_STATUS:
        raise ProtocolV7Error("protocol v7 does not authorize a source availability audit")


def _blocked(protocol: ProtocolV7, action: str) -> None:
    del protocol
    raise ProtocolV7Error(f"protocol v7 has no candidate and cannot {action}")


def require_protocol_v7_input_freeze(protocol: ProtocolV7) -> None:
    _blocked(protocol, "freeze an input")


def require_protocol_v7_execution(protocol: ProtocolV7) -> None:
    _blocked(protocol, "be executed")


def require_protocol_v7_selection(protocol: ProtocolV7) -> None:
    _blocked(protocol, "select a policy")


def require_protocol_v7_oos_authorization(protocol: ProtocolV7) -> None:
    _blocked(protocol, "authorize strict OOS")
