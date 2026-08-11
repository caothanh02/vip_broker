"""Regression coverage for the unscoped, fail-closed Protocol V4 draft."""

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


def test_v4_is_a_candidate_free_availability_audit_draft() -> None:
    protocol = protocol_v4.load_protocol_v4(CONFIG_PATH)
    raw = _raw_protocol()

    assert protocol.status == "draft_availability_audit_required"
    assert protocol.executable is False
    assert raw["candidate"] is None
    assert raw["parameters"] is None
    assert raw["development_dataset_range"] is None
    assert raw["required_input_lock"] is None
    assert raw["selection_artifact"] is None
    assert raw["availability_audit"] == {
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
    }


@pytest.mark.parametrize(
    "mutation",
    [
        lambda raw: raw.__setitem__("candidate", {"id": "candidate"}),
        lambda raw: raw.__setitem__("development_dataset_range", {"start": "2019"}),
        lambda raw: raw.__setitem__("required_input_lock", {"sha": "a" * 64}),
        lambda raw: raw.__setitem__("selection_artifact", {"selected": True}),
        lambda raw: raw.__setitem__("status", "ready"),
        lambda raw: raw["strict_oos"].__setitem__("evaluation_authorized", True),
        lambda raw: raw["availability_audit"].__setitem__("selection_basis", "return"),
    ],
)
def test_v4_rejects_candidate_input_lock_oos_or_metric_based_mutation(mutation: object) -> None:
    raw = deepcopy(_raw_protocol())
    assert callable(mutation)
    mutation(raw)
    with pytest.raises(protocol_v4.ProtocolV4Error, match="immutable draft"):
        protocol_v4.validate_protocol_v4(raw)


def test_v4_draft_cannot_freeze_execute_or_authorize_oos() -> None:
    protocol = protocol_v4.load_protocol_v4(CONFIG_PATH)
    with pytest.raises(protocol_v4.ProtocolV4Error, match="cannot freeze"):
        protocol_v4.require_protocol_v4_input_freeze(protocol)
    with pytest.raises(protocol_v4.ProtocolV4Error, match="cannot be executed"):
        protocol_v4.require_protocol_v4_execution(protocol)
    with pytest.raises(protocol_v4.ProtocolV4Error, match="cannot authorize strict OOS"):
        protocol_v4.require_protocol_v4_oos_authorization(protocol)


def test_v4_has_no_data_model_or_market_execution_dependencies() -> None:
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
