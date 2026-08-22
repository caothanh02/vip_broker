from __future__ import annotations

import inspect
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from trading_bot.recommendations import protocol_v9

ROOT = Path(__file__).resolve().parents[2]


def _raw() -> dict[str, object]:
    value = yaml.safe_load((ROOT / "config/recommendation_protocol_v9.yaml").read_text())
    assert isinstance(value, dict)
    return value


def test_v9_is_non_executable_and_only_authorizes_its_availability_audit() -> None:
    protocol = protocol_v9.load_protocol_v9(ROOT / "config/recommendation_protocol_v9.yaml")
    assert protocol.status == protocol_v9.PROTOCOL_V9_STATUS
    assert protocol.executable is False
    protocol_v9.require_protocol_v9_availability_audit(protocol)


@pytest.mark.parametrize(
    "path",
    [
        ("data_source", "historical_endpoint"),
        ("entitlement_context", "historical_access_verified"),
        ("availability_audit", "utc_range", "end"),
        ("availability_audit", "maximum_request_count"),
        ("candidate",),
        ("strict_oos", "evaluation_authorized"),
    ],
)
def test_v9_rejects_preregistration_mutation(path: tuple[str, ...]) -> None:
    value = deepcopy(_raw())
    target: object = value
    for key in path[:-1]:
        assert isinstance(target, dict)
        target = target[key]
    assert isinstance(target, dict)
    target[path[-1]] = "unsafe"
    with pytest.raises(protocol_v9.ProtocolV9Error, match="immutable"):
        protocol_v9.validate_protocol_v9(value)


def test_v9_blocks_persistence_research_selection_and_oos() -> None:
    protocol = protocol_v9.load_protocol_v9(ROOT / "config/recommendation_protocol_v9.yaml")
    for action in (
        "persist data",
        "freeze an input",
        "execute research",
        "select a policy",
        "authorize strict OOS",
    ):
        with pytest.raises(protocol_v9.ProtocolV9Error, match="cannot"):
            protocol_v9.require_protocol_v9_blocked(protocol, action)


def test_v9_governance_has_no_runtime_or_credential_dependency() -> None:
    source = inspect.getsource(protocol_v9)
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
