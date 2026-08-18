"""Regression coverage for candidate-free Protocol V5 source-audit governance."""

from __future__ import annotations

import inspect
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from trading_bot.recommendations import protocol_v5

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config/recommendation_protocol_v5.yaml"


def _raw_protocol() -> dict[str, object]:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return raw


def test_v5_selects_only_a_source_availability_audit() -> None:
    protocol = protocol_v5.load_protocol_v5(CONFIG_PATH)
    raw = _raw_protocol()

    assert protocol.status == protocol_v5.PROTOCOL_V5_SOURCE_AUDIT_STATUS
    assert protocol.executable is False
    for key in (
        "candidate",
        "parameters",
        "development_dataset_range",
        "required_input_lock",
        "selection_artifact",
    ):
        assert raw[key] is None
    assert raw["data_source"] == protocol_v5._EXPECTED_CONFIG["data_source"]
    assert protocol.availability_start.isoformat() == "2019-01-01T00:00:00+00:00"
    assert protocol.availability_end.isoformat() == "2022-01-01T00:00:00+00:00"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda raw: raw["data_source"].__setitem__("endpoint", "https://other.example"),
        lambda raw: raw.__setitem__("candidate", {"id": "candidate"}),
        lambda raw: raw.__setitem__("parameters", {"ema": 20}),
        lambda raw: raw.__setitem__("development_dataset_range", {"start": "2019"}),
        lambda raw: raw.__setitem__("required_input_lock", {"sha": "a" * 64}),
        lambda raw: raw.__setitem__("selection_artifact", {"selected": True}),
        lambda raw: raw["strict_oos"].__setitem__("evaluation_authorized", True),
        lambda raw: raw["source_selection"].__setitem__("selection_basis", "accuracy"),
        lambda raw: raw["availability_audit"].__setitem__("outcome", "available"),
    ],
)
def test_v5_rejects_source_candidate_or_oos_mutation(mutation: object) -> None:
    raw = deepcopy(_raw_protocol())
    assert callable(mutation)
    mutation(raw)
    with pytest.raises(protocol_v5.ProtocolV5Error, match="immutable source-audit"):
        protocol_v5.validate_protocol_v5(raw)


def test_v5_allows_only_availability_audit_not_candidate_work() -> None:
    protocol = protocol_v5.load_protocol_v5(CONFIG_PATH)
    protocol_v5.require_protocol_v5_availability_audit(protocol)
    for action in (
        protocol_v5.require_protocol_v5_source_ingest,
        protocol_v5.require_protocol_v5_input_freeze,
        protocol_v5.require_protocol_v5_execution,
        protocol_v5.require_protocol_v5_selection,
        protocol_v5.require_protocol_v5_oos_authorization,
    ):
        with pytest.raises(protocol_v5.ProtocolV5Error, match="no candidate"):
            action(protocol)


def test_v5_governance_has_no_data_model_or_market_execution_dependencies() -> None:
    source = inspect.getsource(protocol_v5)
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
        "trading_bot.data",
        "trading_bot.settings",
    ):
        assert forbidden not in source
