"""Immutable, unexecutable preregistration contract for recommendation Protocol V3.

Protocol V3 records one candidate hypothesis before a new independent development
input is available.  This module validates only tracked configuration or a
future input-lock object supplied in memory; it never opens market data or
imports an engine, model, broker, order, risk, settings, or network client.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from pathlib import Path

import yaml

PROTOCOL_V3_ID = "recommendation_research_v3"
PROTOCOL_V3_CANDIDATE_ID = "dual_regime_reclaim_avoid_v3"
PROTOCOL_V3_SCHEMA_VERSION = "1.2"
PROTOCOL_V3_INPUT_LOCK_SCHEMA_VERSION = "1.0"
PROTOCOL_V3_UNFROZEN_STATUS = "candidate_preregistered_input_unfrozen"
PROTOCOL_V3_LOCKED_STATUS = "candidate_preregistered_input_locked"
PROTOCOL_V3_CLOSED_STATUS = "closed_input_unavailable"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GENERATION_ID_RE = re.compile(r"^[0-9a-f]{32,64}$")


class ProtocolV3Error(ValueError):
    """Protocol V3 configuration or an input lock is incomplete or unsafe."""


@dataclass(frozen=True, slots=True)
class ProtocolV3Fold:
    identifier: str
    calibration_start: datetime
    validation_start: datetime
    validation_end: datetime


@dataclass(frozen=True, slots=True)
class ProtocolV3Candidate:
    identifier: str
    outputs: tuple[str, ...]
    parameters: tuple[tuple[str, str], ...]
    cost_model: tuple[tuple[str, Decimal], ...]


@dataclass(frozen=True, slots=True)
class ProtocolV3SelectionGate:
    minimum_fold_applicable_resolved: int
    minimum_each_direction_count: int
    minimum_pooled_applicable_resolved: int
    coverage_minimum: Decimal
    coverage_maximum: Decimal
    directional_accuracy_threshold: Decimal
    mean_after_cost_return_threshold: Decimal
    confidence_lower_bound_threshold: Decimal


@dataclass(frozen=True, slots=True)
class ProtocolV3:
    protocol_id: str
    status: str
    development_start: datetime
    development_end: datetime
    strict_oos_start: datetime
    audited_interruption_ids: tuple[str, ...]
    horizons: tuple[int, ...]
    folds: tuple[ProtocolV3Fold, ...]
    candidate: ProtocolV3Candidate
    selection_gate: ProtocolV3SelectionGate
    required_input_lock_fields: tuple[str, ...]

    @property
    def executable(self) -> bool:
        """Whether a future input-lock transition has been separately approved."""

        return self.status == PROTOCOL_V3_LOCKED_STATUS


@dataclass(frozen=True, slots=True)
class ProtocolV3InputLock:
    protocol_config_sha256: str
    frozen_manifest_sha256: str
    generation_id: str
    csv_sha256: str
    metadata_sha256: str
    anomaly_sha256: str
    symbol: str
    timeframe: str
    start: datetime
    end: datetime
    audited_interruption_ids: tuple[str, ...]


_EXPECTED_CONFIG: dict[str, object] = {
    "schema_version": PROTOCOL_V3_SCHEMA_VERSION,
    "protocol_id": PROTOCOL_V3_ID,
    "status": PROTOCOL_V3_CLOSED_STATUS,
    "goal": "after_cost_buy_bias_and_avoid_quality_not_accuracy_maximization",
    "development_target": {
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "timezone": "UTC",
        "start": "2019-01-01T00:00:00Z",
        "end": "2022-01-01T00:00:00Z",
        "audited_interruptions": [],
    },
    "strict_oos": {
        "start": "2025-01-01T00:00:00Z",
        "status": "sealed",
        "evaluation_authorized": False,
        "selection_artifact_required": True,
    },
    "horizons": [1, 4, 24],
    "folds": [
        {
            "id": "fold_1",
            "calibration_start": "2019-01-01T00:00:00Z",
            "validation_start": "2020-01-01T00:00:00Z",
            "validation_end": "2020-09-01T00:00:00Z",
        },
        {
            "id": "fold_2",
            "calibration_start": "2019-01-01T00:00:00Z",
            "validation_start": "2020-09-01T00:00:00Z",
            "validation_end": "2021-05-01T00:00:00Z",
        },
        {
            "id": "fold_3",
            "calibration_start": "2019-01-01T00:00:00Z",
            "validation_start": "2021-05-01T00:00:00Z",
            "validation_end": "2022-01-01T00:00:00Z",
        },
    ],
    "candidate": {
        "id": PROTOCOL_V3_CANDIDATE_ID,
        "outputs": ["BUY_BIAS", "AVOID", "NEUTRAL"],
        "rules": {
            "candle_state": "closed_utc_1h_only",
            "feature_scope": "causal_same_continuous_segment_only",
            "buy_bias": [
                "close_gt_ema200",
                "ema20_gt_ema50",
                "previous_close_lte_ema20",
                "current_close_gt_ema20",
                "current_volume_gte_1_2_times_volume_sma20",
                "atr14_gt_zero",
            ],
            "avoid": [
                "close_lt_ema200",
                "ema20_lt_ema50",
                "previous_close_gte_ema20",
                "current_close_lt_ema20",
                "current_volume_gte_1_2_times_volume_sma20",
                "atr14_gt_zero",
            ],
        },
        "parameters": {
            "ema_fast": 20,
            "ema_slow": 50,
            "ema_trend": 200,
            "volume_window": 20,
            "volume_multiplier": "1.2",
            "atr_window": 14,
        },
        "cost_model": {
            "entry_fee_rate": "0.001",
            "exit_fee_rate": "0.001",
            "entry_slippage_rate": "0.0005",
            "exit_slippage_rate": "0.0005",
        },
        "ml_allowed": False,
    },
    "selection_gate": {
        "minimum_fold_applicable_resolved": 30,
        "minimum_each_direction_count": 10,
        "minimum_pooled_applicable_resolved": 100,
        "non_neutral_coverage": {
            "comparator": "inclusive_range",
            "minimum": "0.01",
            "maximum": "0.50",
        },
        "after_cost_directional_accuracy": {
            "comparator": "strictly_greater_than",
            "value": "0.50",
        },
        "mean_after_cost_return": {
            "comparator": "strictly_greater_than",
            "value": "0",
        },
        "two_sided_95_percent_exact_lower_bound": {
            "comparator": "strictly_greater_than",
            "value": "0.50",
        },
        "requirements_apply_to": "every_fold_every_horizon_and_pooled_every_horizon",
    },
    "required_input_lock": {
        "status": "unbound",
        "schema_version": PROTOCOL_V3_INPUT_LOCK_SCHEMA_VERSION,
        "required_fields": [
            "protocol_config_sha256",
            "frozen_manifest_sha256",
            "generation_id",
            "csv_sha256",
            "metadata_sha256",
            "anomaly_sha256",
            "symbol",
            "timeframe",
            "utc_range",
            "audited_interruption_ids",
        ],
    },
    "failure_rule": "any_failure_closes_v3_and_requires_v4_for_a_new_idea",
    "closure": {
        "reason": "independent_input_continuity_unavailable",
        "strategy_result_evaluated": False,
        "freeze_authorized": False,
        "execution_authorized": False,
        "selection_authorized": False,
        "strict_oos_authorized": False,
        "verified_binance_vision_findings": {
            "monthly_archive": "BTCUSDT-1h-2019-03.zip",
            "expected_rows": 744,
            "observed_rows": 738,
            "missing_open_start": "2019-03-12T02:00:00Z",
            "missing_open_end": "2019-03-12T07:00:00Z",
            "daily_1h_rows": "18/24",
            "daily_1m_rows": "1080/1440",
            "official_checksums_verified": True,
        },
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


def _require_mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ProtocolV3Error(f"{label} must be an object")
    return value


def _require_list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ProtocolV3Error(f"{label} must be a list")
    return value


def _require_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolV3Error(f"{label} must be an integer")
    return value


def _parse_utc(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise ProtocolV3Error(f"{label} must be an ISO UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProtocolV3Error(f"{label} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ProtocolV3Error(f"{label} must be UTC")
    return parsed.astimezone(UTC)


def _decimal(value: object, label: str) -> Decimal:
    if not isinstance(value, str):
        raise ProtocolV3Error(f"{label} must be an exact decimal string")
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise ProtocolV3Error(f"{label} is invalid") from exc
    if not result.is_finite():
        raise ProtocolV3Error(f"{label} must be finite")
    return result


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ProtocolV3Error(f"{label} must be a lowercase SHA-256")
    return value


def _generation_id(value: object) -> str:
    if not isinstance(value, str) or _GENERATION_ID_RE.fullmatch(value) is None:
        raise ProtocolV3Error("generation_id must be a lowercase hexadecimal identifier")
    return value


def _fold(raw: object) -> ProtocolV3Fold:
    value = _require_mapping(raw, "protocol v3 fold")
    identifier = value.get("id")
    if not isinstance(identifier, str):
        raise ProtocolV3Error("protocol v3 fold id must be a string")
    return ProtocolV3Fold(
        identifier,
        _parse_utc(value.get("calibration_start"), "fold calibration start"),
        _parse_utc(value.get("validation_start"), "fold validation start"),
        _parse_utc(value.get("validation_end"), "fold validation end"),
    )


def validate_protocol_v3(raw: object) -> ProtocolV3:
    """Validate the exact candidate preregistration without opening market data."""

    config = _require_mapping(raw, "protocol v3")
    if config != _EXPECTED_CONFIG:
        raise ProtocolV3Error("protocol v3 differs from its immutable preregistration")
    development = _require_mapping(config["development_target"], "development target")
    strict_oos = _require_mapping(config["strict_oos"], "strict OOS")
    candidate = _require_mapping(config["candidate"], "candidate")
    gate = _require_mapping(config["selection_gate"], "selection gate")
    coverage = _require_mapping(gate["non_neutral_coverage"], "coverage comparator")
    accuracy = _require_mapping(gate["after_cost_directional_accuracy"], "accuracy comparator")
    mean_return = _require_mapping(gate["mean_after_cost_return"], "return comparator")
    confidence = _require_mapping(
        gate["two_sided_95_percent_exact_lower_bound"], "confidence comparator"
    )
    lock = _require_mapping(config["required_input_lock"], "required input lock")

    return ProtocolV3(
        protocol_id=PROTOCOL_V3_ID,
        status=PROTOCOL_V3_CLOSED_STATUS,
        development_start=_parse_utc(development["start"], "development target start"),
        development_end=_parse_utc(development["end"], "development target end"),
        strict_oos_start=_parse_utc(strict_oos["start"], "strict OOS start"),
        audited_interruption_ids=tuple(
            str(item)
            for item in _require_list(
                development["audited_interruptions"], "development target interruptions"
            )
        ),
        horizons=tuple(
            _require_integer(item, "horizon")
            for item in _require_list(config["horizons"], "horizons")
        ),
        folds=tuple(_fold(item) for item in _require_list(config["folds"], "folds")),
        candidate=ProtocolV3Candidate(
            identifier=str(candidate["id"]),
            outputs=tuple(
                str(item) for item in _require_list(candidate["outputs"], "candidate outputs")
            ),
            parameters=tuple(
                sorted(
                    (str(key), str(value))
                    for key, value in _require_mapping(
                        candidate["parameters"], "candidate parameters"
                    ).items()
                )
            ),
            cost_model=tuple(
                sorted(
                    (str(key), _decimal(value, f"cost model {key}"))
                    for key, value in _require_mapping(
                        candidate["cost_model"], "cost model"
                    ).items()
                )
            ),
        ),
        selection_gate=ProtocolV3SelectionGate(
            _require_integer(gate["minimum_fold_applicable_resolved"], "minimum fold samples"),
            _require_integer(gate["minimum_each_direction_count"], "minimum direction samples"),
            _require_integer(gate["minimum_pooled_applicable_resolved"], "minimum pooled samples"),
            _decimal(coverage["minimum"], "minimum coverage"),
            _decimal(coverage["maximum"], "maximum coverage"),
            _decimal(accuracy["value"], "accuracy threshold"),
            _decimal(mean_return["value"], "mean return threshold"),
            _decimal(confidence["value"], "confidence threshold"),
        ),
        required_input_lock_fields=tuple(
            str(item) for item in _require_list(lock["required_fields"], "required lock fields")
        ),
    )


def load_protocol_v3(path: Path = Path("config/recommendation_protocol_v3.yaml")) -> ProtocolV3:
    """Load only the tracked preregistration configuration; never execute it."""

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ProtocolV3Error("could not read protocol v3 preregistration") from exc
    return validate_protocol_v3(raw)


def protocol_v3_config_sha256(path: Path) -> str:
    """Return the exact byte digest that a future input-lock must bind."""

    try:
        return sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ProtocolV3Error("could not read protocol v3 preregistration bytes") from exc


def validate_protocol_v3_input_lock(
    raw: object, protocol: ProtocolV3, expected_config_sha256: str
) -> ProtocolV3InputLock:
    """Validate an in-memory future input lock without reading its dataset paths."""

    lock = _require_mapping(raw, "protocol v3 input lock")
    expected_keys = {
        "schema_version",
        "protocol_id",
        "protocol_config_sha256",
        "frozen_manifest_sha256",
        "dataset",
    }
    if set(lock) != expected_keys:
        raise ProtocolV3Error("protocol v3 input lock has an invalid field set")
    if lock["schema_version"] != PROTOCOL_V3_INPUT_LOCK_SCHEMA_VERSION:
        raise ProtocolV3Error("protocol v3 input lock schema is unsupported")
    if lock["protocol_id"] != protocol.protocol_id:
        raise ProtocolV3Error("protocol v3 input lock protocol id does not match")
    config_sha = _sha256(lock["protocol_config_sha256"], "protocol_config_sha256")
    if config_sha != _sha256(expected_config_sha256, "expected config digest"):
        raise ProtocolV3Error("protocol v3 input lock config digest does not match")
    manifest_sha = _sha256(lock["frozen_manifest_sha256"], "frozen_manifest_sha256")
    dataset = _require_mapping(lock["dataset"], "protocol v3 input lock dataset")
    expected_dataset_keys = {
        "generation_id",
        "csv_sha256",
        "metadata_sha256",
        "anomaly_sha256",
        "symbol",
        "timeframe",
        "utc_range",
        "audited_interruption_ids",
    }
    if set(dataset) != expected_dataset_keys:
        raise ProtocolV3Error("protocol v3 input lock dataset has an invalid field set")
    if dataset["symbol"] != "BTC/USDT" or dataset["timeframe"] != "1h":
        raise ProtocolV3Error("protocol v3 input lock dataset symbol or timeframe does not match")
    utc_range = _require_mapping(dataset["utc_range"], "protocol v3 input lock UTC range")
    if set(utc_range) != {"start", "end"}:
        raise ProtocolV3Error("protocol v3 input lock UTC range has an invalid field set")
    start = _parse_utc(utc_range["start"], "input lock range start")
    end = _parse_utc(utc_range["end"], "input lock range end")
    if start != protocol.development_start or end != protocol.development_end:
        raise ProtocolV3Error("protocol v3 input lock range does not match the independent target")
    interruptions = tuple(
        str(item)
        for item in _require_list(
            dataset["audited_interruption_ids"], "input lock audited interruptions"
        )
    )
    if interruptions != protocol.audited_interruption_ids:
        raise ProtocolV3Error("protocol v3 input lock interruptions do not match")
    return ProtocolV3InputLock(
        protocol_config_sha256=config_sha,
        frozen_manifest_sha256=manifest_sha,
        generation_id=_generation_id(dataset["generation_id"]),
        csv_sha256=_sha256(dataset["csv_sha256"], "csv_sha256"),
        metadata_sha256=_sha256(dataset["metadata_sha256"], "metadata_sha256"),
        anomaly_sha256=_sha256(dataset["anomaly_sha256"], "anomaly_sha256"),
        symbol="BTC/USDT",
        timeframe="1h",
        start=start,
        end=end,
        audited_interruption_ids=interruptions,
    )


def require_protocol_v3_execution(
    protocol: ProtocolV3, raw_input_lock: object, expected_config_sha256: str
) -> ProtocolV3InputLock:
    """Fail closed until a separately reviewed input-locked V3 transition exists."""

    if protocol.status == PROTOCOL_V3_CLOSED_STATUS:
        raise ProtocolV3Error("protocol v3 is closed and cannot be executed")
    if not protocol.executable:
        raise ProtocolV3Error("protocol v3 is input-unfrozen and cannot be executed")
    return validate_protocol_v3_input_lock(raw_input_lock, protocol, expected_config_sha256)
