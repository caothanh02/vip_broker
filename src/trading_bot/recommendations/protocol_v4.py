"""Fail-closed governance contract for unscoped Protocol V4.

V4 has no candidate or input.  It can only be advanced after a separately
reviewed, mechanical public-archive availability audit; this module never
opens data, computes a signal, or imports runtime/trading dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml

PROTOCOL_V4_ID = "recommendation_research_v4"
PROTOCOL_V4_SCHEMA_VERSION = "1.0"
PROTOCOL_V4_DRAFT_STATUS = "draft_availability_audit_required"


class ProtocolV4Error(ValueError):
    """Protocol V4 is incomplete or an unsafe activation was requested."""


@dataclass(frozen=True, slots=True)
class ProtocolV4:
    protocol_id: str
    status: str
    strict_oos_start: datetime

    @property
    def executable(self) -> bool:
        return False


_EXPECTED_CONFIG: dict[str, object] = {
    "schema_version": PROTOCOL_V4_SCHEMA_VERSION,
    "protocol_id": PROTOCOL_V4_ID,
    "status": PROTOCOL_V4_DRAFT_STATUS,
    "availability_audit": {
        "source": "public_binance_vision_archive_only",
        "selection_basis": "mechanical_availability_only",
        "required_checks": [
            "official_checksum_verified",
            "btc_usdt_1h_utc",
            "closed_candles_only",
            "absolute_continuity",
        ],
        "prohibited_selection_inputs": [
            "signal",
            "return",
            "accuracy",
            "backtest",
            "performance_metric",
        ],
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


def _parse_utc(value: object) -> datetime:
    if not isinstance(value, str):
        raise ProtocolV4Error("strict OOS start must be a UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProtocolV4Error("strict OOS start is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ProtocolV4Error("strict OOS start must be UTC")
    return parsed.astimezone(UTC)


def validate_protocol_v4(raw: object) -> ProtocolV4:
    """Validate the exact draft without reading an input or candidate result."""

    if not isinstance(raw, dict) or raw != _EXPECTED_CONFIG:
        raise ProtocolV4Error("protocol v4 differs from its immutable draft governance")
    strict_oos = raw["strict_oos"]
    assert isinstance(strict_oos, dict)
    return ProtocolV4(PROTOCOL_V4_ID, PROTOCOL_V4_DRAFT_STATUS, _parse_utc(strict_oos["start"]))


def load_protocol_v4(path: Path = Path("config/recommendation_protocol_v4.yaml")) -> ProtocolV4:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ProtocolV4Error("could not read protocol v4 draft") from exc
    return validate_protocol_v4(raw)


def require_protocol_v4_input_freeze(protocol: ProtocolV4) -> None:
    del protocol
    raise ProtocolV4Error("protocol v4 is draft and cannot freeze an input")


def require_protocol_v4_execution(protocol: ProtocolV4) -> None:
    del protocol
    raise ProtocolV4Error("protocol v4 is draft and cannot be executed")


def require_protocol_v4_oos_authorization(protocol: ProtocolV4) -> None:
    del protocol
    raise ProtocolV4Error("protocol v4 is draft and cannot authorize strict OOS")
