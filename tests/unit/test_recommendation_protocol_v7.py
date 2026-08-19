from __future__ import annotations

import inspect
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from trading_bot.recommendations import protocol_v7

ROOT = Path(__file__).resolve().parents[2]


def _raw() -> dict[str, object]:
    value = yaml.safe_load((ROOT / "config/recommendation_protocol_v7.yaml").read_text())
    assert isinstance(value, dict)
    return value


def test_v7_is_non_executable_and_allows_only_availability_audit() -> None:
    protocol = protocol_v7.load_protocol_v7(ROOT / "config/recommendation_protocol_v7.yaml")
    assert protocol.executable is False
    assert protocol.expected_candle_count == 26304
    protocol_v7.require_protocol_v7_availability_audit(protocol)
    for require in (
        protocol_v7.require_protocol_v7_input_freeze,
        protocol_v7.require_protocol_v7_execution,
        protocol_v7.require_protocol_v7_selection,
        protocol_v7.require_protocol_v7_oos_authorization,
    ):
        with pytest.raises(protocol_v7.ProtocolV7Error, match="cannot"):
            require(protocol)


@pytest.mark.parametrize(
    "path",
    [
        ("data_source", "endpoint"),
        ("availability_audit", "expected_candle_count"),
        ("candidate",),
        ("strict_oos", "evaluation_authorized"),
    ],
)
def test_v7_rejects_governance_mutation(path: tuple[str, ...]) -> None:
    value = deepcopy(_raw())
    target: object = value
    for key in path[:-1]:
        assert isinstance(target, dict)
        target = target[key]
    assert isinstance(target, dict)
    target[path[-1]] = "unsafe"
    with pytest.raises(protocol_v7.ProtocolV7Error, match="immutable"):
        protocol_v7.validate_protocol_v7(value)


def test_v7_governance_has_no_network_credential_or_execution_dependency() -> None:
    source = inspect.getsource(protocol_v7)
    for forbidden in (
        "httpx",
        "api_key",
        "Broker",
        "Order",
        "RiskEngine",
        "settings",
        "trading_bot.data",
    ):
        assert forbidden not in source
