"""Fail-closed Tardis source preregistration for Protocol V10.

This pure governance record authorizes no data download, candle aggregation,
research, recommendation, broker, order, ML, or OOS runtime activity.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml

PROTOCOL_V10_ID = "recommendation_research_v10"
PROTOCOL_V10_SCHEMA_VERSION = "1.0"
PROTOCOL_V10_STATUS = "source_selected_metadata_identity_probe_authorized"


class ProtocolV10Error(ValueError):
    """Protocol V10 has not completed its independent governance gates."""


@dataclass(frozen=True, slots=True)
class ProtocolV10:
    protocol_id: str
    status: str
    proposed_start: datetime
    proposed_end: datetime
    strict_oos_start: datetime
    expected_candle_count: int
    maximum_request_count: int

    @property
    def executable(self) -> bool:
        """V10 never authorizes research or trading execution."""

        return False


_EXPECTED: dict[str, object] = {
    "schema_version": PROTOCOL_V10_SCHEMA_VERSION,
    "protocol_id": PROTOCOL_V10_ID,
    "status": PROTOCOL_V10_STATUS,
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
        "provider": "tardis_binance_spot_trade_csv",
        "metadata_endpoint": "https://api.tardis.dev/v1/instruments/binance/BTCUSDT",
        "dataset_url_template": (
            "https://datasets.tardis.dev/v1/binance/trades/{year}/{month}/{day}/BTCUSDT.csv.gz"
        ),
        "exchange": "binance",
        "exchange_symbol": "BTCUSDT",
        "internal_symbol": "BTC/USDT",
        "market_type": "spot",
        "source_data_type": "trades",
        "candle_aggregation": "exchange_event_timestamp_utc_to_1h_ohlcv",
        "timezone": "UTC",
        "authentication": "local_env_key_for_metadata_identity_probe_only",
    },
    "entitlement_context": {
        "credential_name": "TARDIS_API_KEY",
        "historical_access_verified": False,
        "metadata_identity_verified": False,
        "metadata_probe_proves_historical_access": False,
    },
    "proposed_independent_range": {
        "start": "2019-04-01T00:00:00Z",
        "end": "2022-04-01T00:00:00Z",
        "expected_candle_count": 26304,
    },
    "metadata_identity_probe": {
        "maximum_request_count": 1,
        "response_contract": {
            "required_fields": [
                "id",
                "datasetId",
                "exchange",
                "baseCurrency",
                "quoteCurrency",
                "type",
                "availableSince",
            ],
            "expected_id": "BTCUSDT",
            "expected_dataset_id": "BTCUSDT",
            "expected_exchange": "binance",
            "expected_base_currency": "BTC",
            "expected_quote_currency": "USDT",
            "expected_type": "spot",
            "available_since_must_be_at_or_before": "2019-04-01T00:00:00Z",
        },
        "data_persistence_authorized": False,
        "candle_aggregation_authorized": False,
        "candidate_or_parameter_authorized": False,
        "recommendation_or_backtest_authorized": False,
        "strict_oos_authorized": False,
    },
    "raw_trade_boundary_policy": {
        "source_partition_timestamp": "provider_local_timestamp_utc",
        "target_candle_timestamp": "exchange_event_timestamp_utc",
        "finite_boundary_partition_set_preregistered": False,
        "exchange_event_filter_preregistered": False,
        "duplicate_trade_policy_preregistered": False,
        "aggregation_authorized": False,
        "fail_closed_rule": (
            "a later independently preregistered protocol must bind a documented finite "
            "partition set, exchange-event filter, deduplication rule, and continuity evidence "
            "before any raw trade download or candle aggregation"
        ),
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
        raise ProtocolV10Error(f"{label} must be a UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProtocolV10Error(f"{label} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ProtocolV10Error(f"{label} must be UTC")
    return parsed.astimezone(UTC)


def validate_protocol_v10(raw: object) -> ProtocolV10:
    """Validate the exact V10 metadata-identity-probe preregistration."""

    if not isinstance(raw, dict) or raw != _EXPECTED:
        raise ProtocolV10Error("protocol v10 differs from its immutable preregistration")
    proposed = raw["proposed_independent_range"]
    probe = raw["metadata_identity_probe"]
    strict_oos = raw["strict_oos"]
    assert isinstance(proposed, dict) and isinstance(probe, dict) and isinstance(strict_oos, dict)
    start = _utc(proposed["start"], "proposed start")
    end = _utc(proposed["end"], "proposed end")
    strict_oos_start = _utc(strict_oos["start"], "strict OOS start")
    count = proposed["expected_candle_count"]
    maximum_requests = probe["maximum_request_count"]
    if not isinstance(count, int) or count <= 0:
        raise ProtocolV10Error("expected candle count is invalid")
    if not isinstance(maximum_requests, int) or maximum_requests != 1:
        raise ProtocolV10Error("metadata identity probe request bound is invalid")
    if start >= end or end > strict_oos_start or (end - start) // timedelta(hours=1) != count:
        raise ProtocolV10Error("proposed range is invalid")
    return ProtocolV10(
        PROTOCOL_V10_ID,
        PROTOCOL_V10_STATUS,
        start,
        end,
        strict_oos_start,
        count,
        maximum_requests,
    )


def load_protocol_v10(path: Path = Path("config/recommendation_protocol_v10.yaml")) -> ProtocolV10:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ProtocolV10Error("could not read protocol v10 preregistration") from exc
    return validate_protocol_v10(raw)


def require_protocol_v10_metadata_identity_probe(protocol: ProtocolV10) -> None:
    """Permit only a future separately implemented single metadata check."""

    if protocol.status != PROTOCOL_V10_STATUS:
        raise ProtocolV10Error("protocol v10 metadata identity probe is not authorized")


def require_protocol_v10_blocked(protocol: ProtocolV10, action: str) -> None:
    """Fail closed for every action other than the preregistered metadata check."""

    del protocol
    raise ProtocolV10Error(f"protocol v10 cannot {action}")
