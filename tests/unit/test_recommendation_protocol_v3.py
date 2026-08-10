"""Regression coverage for the unexecutable Protocol V3 preregistration.

The suite validates configuration and in-memory lock objects only. It never
constructs a recommendation engine or opens market data, especially not OOS.
"""

from __future__ import annotations

import inspect
from copy import deepcopy
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from trading_bot.recommendations import candidate_rules, experiments, protocol_v3

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config/recommendation_protocol_v3.yaml"
CONFIG_DIGEST = "a" * 64


def _raw_protocol() -> dict[str, object]:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return raw


def _mapping(raw: dict[str, object], key: str) -> dict[str, object]:
    value = raw[key]
    assert isinstance(value, dict)
    return value


def _valid_input_lock(protocol: protocol_v3.ProtocolV3) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "protocol_id": protocol.protocol_id,
        "protocol_config_sha256": CONFIG_DIGEST,
        "frozen_manifest_sha256": "b" * 64,
        "dataset": {
            "generation_id": "c" * 32,
            "csv_sha256": "d" * 64,
            "metadata_sha256": "e" * 64,
            "anomaly_sha256": "f" * 64,
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "utc_range": {
                "start": "2019-01-01T00:00:00Z",
                "end": "2022-01-01T00:00:00Z",
            },
            "audited_interruption_ids": [],
        },
    }


def test_protocol_v3_is_candidate_preregistration_with_an_independent_unfrozen_target() -> None:
    protocol = protocol_v3.load_protocol_v3(CONFIG_PATH)

    assert protocol.protocol_id == "recommendation_research_v3"
    assert protocol.status == "candidate_preregistered_input_unfrozen"
    assert protocol.executable is False
    assert protocol.development_start.isoformat() == "2019-01-01T00:00:00+00:00"
    assert protocol.development_end.isoformat() == "2022-01-01T00:00:00+00:00"
    assert protocol.strict_oos_start.isoformat() == "2025-01-01T00:00:00+00:00"
    assert protocol.audited_interruption_ids == ()
    assert protocol.horizons == (1, 4, 24)
    assert [fold.identifier for fold in protocol.folds] == ["fold_1", "fold_2", "fold_3"]
    assert protocol.folds[-1].validation_end == protocol.development_end
    assert protocol.candidate.identifier == "dual_regime_reclaim_avoid_v3"
    assert protocol.candidate.outputs == ("BUY_BIAS", "AVOID", "NEUTRAL")
    assert dict(protocol.candidate.cost_model) == {
        "entry_fee_rate": Decimal("0.001"),
        "exit_fee_rate": Decimal("0.001"),
        "entry_slippage_rate": Decimal("0.0005"),
        "exit_slippage_rate": Decimal("0.0005"),
    }
    assert protocol.selection_gate.minimum_fold_applicable_resolved == 30
    assert protocol.selection_gate.minimum_each_direction_count == 10
    assert protocol.selection_gate.minimum_pooled_applicable_resolved == 100
    assert protocol.selection_gate.coverage_minimum == Decimal("0.01")
    assert protocol.selection_gate.coverage_maximum == Decimal("0.50")
    assert protocol.selection_gate.directional_accuracy_threshold == Decimal("0.50")
    assert protocol.selection_gate.mean_after_cost_return_threshold == Decimal("0")
    assert protocol.selection_gate.confidence_lower_bound_threshold == Decimal("0.50")


def test_protocol_v3_machine_readable_comparators_are_exact() -> None:
    raw = _raw_protocol()
    gate = _mapping(raw, "selection_gate")

    assert gate["non_neutral_coverage"] == {
        "comparator": "inclusive_range",
        "minimum": "0.01",
        "maximum": "0.50",
    }
    assert gate["after_cost_directional_accuracy"] == {
        "comparator": "strictly_greater_than",
        "value": "0.50",
    }
    assert gate["mean_after_cost_return"] == {
        "comparator": "strictly_greater_than",
        "value": "0",
    }
    assert gate["two_sided_95_percent_exact_lower_bound"] == {
        "comparator": "strictly_greater_than",
        "value": "0.50",
    }


@pytest.mark.parametrize(
    "mutation",
    [
        lambda raw: raw.__setitem__("status", "candidate_preregistered_input_locked"),
        lambda raw: _mapping(raw, "development_target").__setitem__("end", "2025-01-01T00:00:00Z"),
        lambda raw: raw.__setitem__("horizons", [1, 4]),
        lambda raw: _mapping(raw, "candidate").__setitem__("id", "other_candidate"),
        lambda raw: _mapping(_mapping(raw, "candidate"), "parameters").__setitem__("ema_fast", 21),
        lambda raw: _mapping(_mapping(raw, "candidate"), "cost_model").__setitem__(
            "entry_fee_rate", "0.002"
        ),
        lambda raw: _mapping(raw, "candidate").__setitem__("ml_allowed", True),
        lambda raw: _mapping(raw, "strict_oos").__setitem__("status", "open"),
        lambda raw: _mapping(raw, "selection_gate").__setitem__(
            "minimum_pooled_applicable_resolved", 99
        ),
        lambda raw: _mapping(_mapping(raw, "selection_gate"), "non_neutral_coverage").__setitem__(
            "comparator", "strictly_greater_than"
        ),
        lambda raw: raw.__setitem__("unregistered_parameter_sweep", True),
    ],
)
def test_protocol_v3_rejects_mutated_target_candidate_cost_gate_or_oos_lock(
    mutation: object,
) -> None:
    raw = deepcopy(_raw_protocol())
    assert callable(mutation)
    mutation(raw)

    with pytest.raises(protocol_v3.ProtocolV3Error, match="immutable preregistration"):
        protocol_v3.validate_protocol_v3(raw)


def test_protocol_v3_input_lock_is_pure_and_binds_full_independent_dataset_identity() -> None:
    protocol = protocol_v3.load_protocol_v3(CONFIG_PATH)
    raw_lock = _valid_input_lock(protocol)
    config_digest = protocol_v3.protocol_v3_config_sha256(CONFIG_PATH)
    raw_lock["protocol_config_sha256"] = config_digest

    lock = protocol_v3.validate_protocol_v3_input_lock(raw_lock, protocol, config_digest)

    assert lock.protocol_config_sha256 == config_digest
    assert lock.frozen_manifest_sha256 == "b" * 64
    assert lock.generation_id == "c" * 32
    assert lock.csv_sha256 == "d" * 64
    assert lock.metadata_sha256 == "e" * 64
    assert lock.anomaly_sha256 == "f" * 64
    assert lock.symbol == "BTC/USDT"
    assert lock.timeframe == "1h"
    assert lock.start == protocol.development_start
    assert lock.end == protocol.development_end
    assert lock.audited_interruption_ids == ()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda lock: lock.pop("frozen_manifest_sha256"),
        lambda lock: lock.__setitem__("protocol_config_sha256", "not-a-sha"),
        lambda lock: _mapping(lock, "dataset").__setitem__("csv_sha256", "A" * 64),
        lambda lock: _mapping(lock, "dataset").__setitem__("symbol", "ETH/USDT"),
        lambda lock: _mapping(lock, "dataset").__setitem__("timeframe", "4h"),
        lambda lock: _mapping(_mapping(lock, "dataset"), "utc_range").__setitem__(
            "end", "2022-01-02T00:00:00Z"
        ),
        lambda lock: _mapping(lock, "dataset").__setitem__(
            "audited_interruption_ids", ["binance-spot-2023-03-24-trailing-stop-maintenance"]
        ),
        lambda lock: _mapping(lock, "dataset").__setitem__("unexpected", "field"),
    ],
)
def test_protocol_v3_input_lock_rejects_missing_or_mismatched_identity(mutation: object) -> None:
    protocol = protocol_v3.load_protocol_v3(CONFIG_PATH)
    raw_lock = _valid_input_lock(protocol)
    assert callable(mutation)
    mutation(raw_lock)

    with pytest.raises(protocol_v3.ProtocolV3Error):
        protocol_v3.validate_protocol_v3_input_lock(raw_lock, protocol, CONFIG_DIGEST)


def test_protocol_v3_activation_fails_before_input_lock_or_dataset_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = protocol_v3.load_protocol_v3(CONFIG_PATH)

    def reject_lock_read(*args: object, **kwargs: object) -> protocol_v3.ProtocolV3InputLock:
        raise AssertionError("input lock must not be inspected for an unfrozen protocol")

    monkeypatch.setattr(protocol_v3, "validate_protocol_v3_input_lock", reject_lock_read)

    with pytest.raises(protocol_v3.ProtocolV3Error, match="input-unfrozen"):
        protocol_v3.require_protocol_v3_execution(protocol, object(), CONFIG_DIGEST)


def test_protocol_v3_is_not_registered_or_runnable_before_separate_implementation() -> None:
    candidate_source = inspect.getsource(candidate_rules)
    experiment_source = inspect.getsource(experiments)

    assert protocol_v3.PROTOCOL_V3_CANDIDATE_ID not in candidate_source
    assert protocol_v3.PROTOCOL_V3_CANDIDATE_ID not in experiment_source


def test_protocol_v3_validation_has_no_market_execution_model_or_network_dependencies() -> None:
    source = inspect.getsource(protocol_v3)

    for forbidden in (
        "RecommendationEngine",
        "RiskEngine",
        "BinanceBroker",
        "DryRunBroker",
        "OrderRequest",
        "ProbabilityModel",
        "api_key",
        "api_secret",
        "httpx",
        "requests",
        "websocket",
        "trading_bot.settings",
        "trading_bot.data",
    ):
        assert forbidden not in source
