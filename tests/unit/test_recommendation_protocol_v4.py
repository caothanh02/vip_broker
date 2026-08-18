"""Regression coverage for the immutable, closed Protocol V4 record."""

from __future__ import annotations

import inspect
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from trading_bot.recommendations import protocol_v4

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config/recommendation_protocol_v4.yaml"


def _raw_protocol() -> dict[str, object]:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return raw


def test_v4_is_closed_after_a_facts_only_availability_audit() -> None:
    protocol = protocol_v4.load_protocol_v4(CONFIG_PATH)
    raw = _raw_protocol()

    assert protocol.status == protocol_v4.PROTOCOL_V4_CLOSED_STATUS
    assert protocol.executable is False
    assert raw["candidate"] is None
    assert raw["parameters"] is None
    assert raw["development_dataset_range"] is None
    closure = raw["closure"]
    assert isinstance(closure, dict)
    assert closure["strategy_result_evaluated"] is False
    assert closure["source_selection_authorized"] is False
    assert closure["input_freeze_authorized"] is False
    assert closure["execution_authorized"] is False
    assert closure["strict_oos_authorized"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda raw: raw.__setitem__("status", "draft_availability_audit_required"),
        lambda raw: raw.__setitem__("candidate", {"id": "candidate"}),
        lambda raw: raw.__setitem__("development_dataset_range", {"start": "2019"}),
        lambda raw: raw["closure"].__setitem__("execution_authorized", True),
        lambda raw: raw["strict_oos"].__setitem__("evaluation_authorized", True),
    ],
)
def test_v4_rejects_reopening_or_authorizing_mutation(mutation: object) -> None:
    raw = deepcopy(_raw_protocol())
    assert callable(mutation)
    mutation(raw)
    with pytest.raises(protocol_v4.ProtocolV4Error, match="immutable closure"):
        protocol_v4.validate_protocol_v4(raw)


def test_v4_closed_cannot_audit_freeze_execute_or_authorize_oos() -> None:
    protocol = protocol_v4.load_protocol_v4(CONFIG_PATH)
    for action in (
        protocol_v4.require_protocol_v4_availability_audit,
        protocol_v4.require_protocol_v4_input_freeze,
        protocol_v4.require_protocol_v4_execution,
        protocol_v4.require_protocol_v4_oos_authorization,
    ):
        with pytest.raises(protocol_v4.ProtocolV4Error, match="closed"):
            action(protocol)


def test_v4_closure_has_no_data_model_or_market_execution_dependencies() -> None:
    source = inspect.getsource(protocol_v4)
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
