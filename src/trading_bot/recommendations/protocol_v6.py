"""Fail-closed CoinAPI source preregistration for Protocol V6.

This module is a pure governance record. It must not read credentials, make a
network request, or authorize any data, candidate, recommendation or OOS work.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml

PROTOCOL_V6_ID = "recommendation_research_v6"
PROTOCOL_V6_SCHEMA_VERSION = "1.0"
PROTOCOL_V6_STATUS = "source_selected_access_verification_required"


class ProtocolV6Error(ValueError):
    """Protocol V6 has not completed its required independent governance gates."""


@dataclass(frozen=True, slots=True)
class ProtocolV6:
    protocol_id: str
    status: str
    proposed_start: datetime
    proposed_end: datetime
    strict_oos_start: datetime

    @property
    def executable(self) -> bool:
        return False


_EXPECTED: dict[str, object] = {
    "schema_version": PROTOCOL_V6_SCHEMA_VERSION,
    "protocol_id": PROTOCOL_V6_ID,
    "status": PROTOCOL_V6_STATUS,
    "source_selection": {
        "selection_basis": "license_provenance_entitlement_and_mechanical_availability_only",
        "required_facts": [
            "license_or_terms_verified",
            "source_provenance_verified",
            "entitlement_verified",
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
        "exchange_symbol": "BINANCE_SPOT_BTC_USDT",
        "internal_symbol": "BTC/USDT",
        "market_type": "spot",
        "timeframe": "1h",
        "authentication": "required_later_not_loaded_by_v6_governance",
        "endpoint": "not_enabled_until_access_verification",
    },
    "proposed_independent_range": {"start": "2019-01-01T00:00:00Z", "end": "2022-01-01T00:00:00Z"},
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
        raise ProtocolV6Error(f"{label} must be a UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProtocolV6Error(f"{label} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ProtocolV6Error(f"{label} must be UTC")
    return parsed.astimezone(UTC)


def validate_protocol_v6(raw: object) -> ProtocolV6:
    if not isinstance(raw, dict) or raw != _EXPECTED:
        raise ProtocolV6Error("protocol v6 differs from its immutable preregistration")
    proposed = raw["proposed_independent_range"]
    oos = raw["strict_oos"]
    assert isinstance(proposed, dict) and isinstance(oos, dict)
    start, end, oos_start = (
        _utc(proposed["start"], "proposed start"),
        _utc(proposed["end"], "proposed end"),
        _utc(oos["start"], "strict OOS start"),
    )
    if start >= end or end > oos_start:
        raise ProtocolV6Error("proposed range is invalid")
    return ProtocolV6(PROTOCOL_V6_ID, PROTOCOL_V6_STATUS, start, end, oos_start)


def load_protocol_v6(path: Path = Path("config/recommendation_protocol_v6.yaml")) -> ProtocolV6:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ProtocolV6Error("could not read protocol v6 preregistration") from exc
    return validate_protocol_v6(raw)


def require_protocol_v6_access_verified(protocol: ProtocolV6, action: str) -> None:
    del protocol
    raise ProtocolV6Error(f"protocol v6 access is unverified and cannot {action}")
