"""Fail-closed source-selection governance for unscoped Protocol V5.

V5 does not authorize a source, dataset, candidate, or research action. It
records only the criteria for a future source-governance decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml

PROTOCOL_V5_ID = "recommendation_research_v5"
PROTOCOL_V5_SCHEMA_VERSION = "1.0"
PROTOCOL_V5_DRAFT_STATUS = "draft_source_selection_required"


class ProtocolV5Error(ValueError):
    """Protocol V5 is incomplete or an unsafe action was requested."""


@dataclass(frozen=True, slots=True)
class ProtocolV5:
    protocol_id: str
    status: str
    strict_oos_start: datetime

    @property
    def executable(self) -> bool:
        return False


_EXPECTED_CONFIG: dict[str, object] = {
    "schema_version": PROTOCOL_V5_SCHEMA_VERSION,
    "protocol_id": PROTOCOL_V5_ID,
    "status": PROTOCOL_V5_DRAFT_STATUS,
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
    "data_source": None,
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
        raise ProtocolV5Error("strict OOS start must be a UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProtocolV5Error("strict OOS start is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ProtocolV5Error("strict OOS start must be UTC")
    return parsed.astimezone(UTC)


def validate_protocol_v5(raw: object) -> ProtocolV5:
    """Validate the exact V5 draft without reading data or source metadata."""

    if not isinstance(raw, dict) or raw != _EXPECTED_CONFIG:
        raise ProtocolV5Error("protocol v5 differs from its immutable draft governance")
    strict_oos = raw["strict_oos"]
    assert isinstance(strict_oos, dict)
    return ProtocolV5(PROTOCOL_V5_ID, PROTOCOL_V5_DRAFT_STATUS, _parse_utc(strict_oos["start"]))


def load_protocol_v5(path: Path = Path("config/recommendation_protocol_v5.yaml")) -> ProtocolV5:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ProtocolV5Error("could not read protocol v5 draft") from exc
    return validate_protocol_v5(raw)


def _blocked(protocol: ProtocolV5, action: str) -> None:
    del protocol
    raise ProtocolV5Error(f"protocol v5 is draft and cannot {action}")


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
