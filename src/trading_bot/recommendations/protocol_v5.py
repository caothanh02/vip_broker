"""Immutable closure record for Protocol V5.

Gate's public endpoint could not cover V5's preregistered history. V5 is now
closed and cannot make another network, source, candidate or OOS action.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml

PROTOCOL_V5_ID = "recommendation_research_v5"
PROTOCOL_V5_SCHEMA_VERSION = "1.2"
PROTOCOL_V5_CLOSED_STATUS = "closed_input_unavailable"


class ProtocolV5Error(ValueError):
    """Protocol V5 is incomplete or an unsafe action was requested."""


@dataclass(frozen=True, slots=True)
class ProtocolV5:
    protocol_id: str
    status: str
    strict_oos_start: datetime
    availability_start: datetime
    availability_end: datetime

    @property
    def executable(self) -> bool:
        return False


_EXPECTED_CONFIG: dict[str, object] = {
    "schema_version": PROTOCOL_V5_SCHEMA_VERSION,
    "protocol_id": PROTOCOL_V5_ID,
    "status": PROTOCOL_V5_CLOSED_STATUS,
    "source_selection": {
        "selection_basis": "license_provenance_and_mechanical_availability_only",
        "required_facts": [
            "license_or_terms_verified",
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
        "provider": "gate_io_public_spot_rest",
        "endpoint": "https://api.gateio.ws/api/v4/spot/candlesticks",
        "exchange_symbol": "BTC_USDT",
        "internal_symbol": "BTC/USDT",
        "market_type": "spot",
        "timeframe": "1h",
        "authentication": "none",
        "selection_evidence": [
            "public_market_candlesticks_endpoint",
            "btc_usdt_spot_1h_utc_contract",
            "closed_candle_boundary_enforced_locally",
            "one_request_per_second_pacing",
        ],
    },
    "availability_audit": {
        "utc_range": {
            "start": "2019-01-01T00:00:00Z",
            "end": "2022-01-01T00:00:00Z",
        },
        "request_page_candles": 1000,
        "minimum_request_interval_seconds": 1,
        "outcome": "closed_source_history_window",
        "observed_response": {
            "checked_at": "2026-08-18T04:02:47Z",
            "http_status": 400,
            "provider_error": "Candlestick too long ago. Maximum 10000 points ago are allowed",
        },
    },
    "closure": {
        "reason": "public_source_history_window_does_not_cover_preregistered_range",
        "strategy_result_evaluated": False,
        "source_selection_authorized": False,
        "input_freeze_authorized": False,
        "execution_authorized": False,
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


def _parse_utc(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise ProtocolV5Error(f"{label} must be a UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProtocolV5Error(f"{label} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ProtocolV5Error(f"{label} must be UTC")
    return parsed.astimezone(UTC)


def validate_protocol_v5(raw: object) -> ProtocolV5:
    """Validate the exact V5 closure record without reading market input."""
    if not isinstance(raw, dict) or raw != _EXPECTED_CONFIG:
        raise ProtocolV5Error("protocol v5 differs from its immutable closure record")
    strict_oos = raw["strict_oos"]
    availability_audit = raw["availability_audit"]
    assert isinstance(strict_oos, dict)
    assert isinstance(availability_audit, dict)
    utc_range = availability_audit["utc_range"]
    assert isinstance(utc_range, dict)
    strict_oos_start = _parse_utc(strict_oos["start"], "strict OOS start")
    availability_start = _parse_utc(utc_range["start"], "availability audit start")
    availability_end = _parse_utc(utc_range["end"], "availability audit end")
    if availability_start >= availability_end or availability_end > strict_oos_start:
        raise ProtocolV5Error("availability audit range is invalid")
    return ProtocolV5(
        PROTOCOL_V5_ID,
        PROTOCOL_V5_CLOSED_STATUS,
        strict_oos_start,
        availability_start,
        availability_end,
    )


def load_protocol_v5(path: Path = Path("config/recommendation_protocol_v5.yaml")) -> ProtocolV5:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ProtocolV5Error("could not read protocol v5 closure record") from exc
    return validate_protocol_v5(raw)


def require_protocol_v5_availability_audit(protocol: ProtocolV5) -> None:
    del protocol
    raise ProtocolV5Error("protocol v5 is closed and cannot audit an input source")


def _blocked(protocol: ProtocolV5, action: str) -> None:
    del protocol
    raise ProtocolV5Error(f"protocol v5 is closed and cannot {action}")


def require_protocol_v5_source_ingest(protocol: ProtocolV5) -> None:
    _blocked(protocol, "ingest a source")


def require_protocol_v5_input_freeze(protocol: ProtocolV5) -> None:
    _blocked(protocol, "freeze an input")


def require_protocol_v5_execution(protocol: ProtocolV5) -> None:
    _blocked(protocol, "be executed")


def require_protocol_v5_selection(protocol: ProtocolV5) -> None:
    _blocked(protocol, "select a policy")


def require_protocol_v5_oos_authorization(protocol: ProtocolV5) -> None:
    _blocked(protocol, "authorize strict OOS")
