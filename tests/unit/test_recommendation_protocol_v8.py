from __future__ import annotations

import inspect
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from trading_bot.recommendations import protocol_v8

ROOT = Path(__file__).resolve().parents[2]


def _raw() -> dict[str, object]:
    value = yaml.safe_load((ROOT / "config/recommendation_protocol_v8.yaml").read_text())
    assert isinstance(value, dict)
    return value


def test_v8_is_non_executable_and_permits_only_availability_audit() -> None:
    protocol = protocol_v8.load_protocol_v8(ROOT / "config/recommendation_protocol_v8.yaml")
    assert protocol.status == protocol_v8.PROTOCOL_V8_STATUS
    assert protocol.executable is False
    protocol_v8.require_protocol_v8_availability_audit(protocol)


@pytest.mark.parametrize(
    "path",
    [
        ("data_source", "historical_endpoint"),
        ("availability_audit", "utc_range", "end"),
        ("availability_audit", "maximum_request_count"),
        ("candidate",),
        ("strict_oos", "evaluation_authorized"),
    ],
)
def test_v8_rejects_governance_mutation(path: tuple[str, ...]) -> None:
    value = deepcopy(_raw())
    target: object = value
    for key in path[:-1]:
        assert isinstance(target, dict)
        target = target[key]
    assert isinstance(target, dict)
    target[path[-1]] = "unsafe"
    with pytest.raises(protocol_v8.ProtocolV8Error, match="immutable"):
        protocol_v8.validate_protocol_v8(value)


def test_v8_blocks_freeze_execution_selection_and_oos() -> None:
    protocol = protocol_v8.load_protocol_v8(ROOT / "config/recommendation_protocol_v8.yaml")
    for function in (
        protocol_v8.require_protocol_v8_input_freeze,
        protocol_v8.require_protocol_v8_execution,
        protocol_v8.require_protocol_v8_selection,
        protocol_v8.require_protocol_v8_oos_authorization,
    ):
        with pytest.raises(protocol_v8.ProtocolV8Error, match="cannot"):
            function(protocol)


def test_v8_governance_has_no_runtime_or_credential_dependency() -> None:
    source = inspect.getsource(protocol_v8)
    for forbidden in (
        "httpx",
        "api_key",
        "api_secret",
        "Broker",
        "Order",
        "RiskEngine",
        "settings",
        "trading_bot.data",
    ):
        assert forbidden not in source
