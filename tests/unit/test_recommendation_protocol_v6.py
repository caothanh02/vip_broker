from __future__ import annotations

import inspect
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from trading_bot.recommendations import protocol_v6

ROOT = Path(__file__).resolve().parents[2]


def _raw() -> dict[str, object]:
    value = yaml.safe_load((ROOT / "config/recommendation_protocol_v6.yaml").read_text())
    assert isinstance(value, dict)
    return value


def test_v6_is_non_executable_preregistration() -> None:
    protocol = protocol_v6.load_protocol_v6(ROOT / "config/recommendation_protocol_v6.yaml")
    assert protocol.status == protocol_v6.PROTOCOL_V6_STATUS
    assert protocol.executable is False
    protocol_v6.require_protocol_v6_access_verification(protocol)


@pytest.mark.parametrize(
    "path",
    [
        ("candidate",),
        ("parameters",),
        ("data_source", "provider"),
        ("strict_oos", "evaluation_authorized"),
    ],
)
def test_v6_rejects_mutation(path: tuple[str, ...]) -> None:
    value = deepcopy(_raw())
    target: object = value
    for key in path[:-1]:
        assert isinstance(target, dict)
        target = target[key]
    assert isinstance(target, dict)
    target[path[-1]] = "unsafe"
    with pytest.raises(protocol_v6.ProtocolV6Error, match="immutable"):
        protocol_v6.validate_protocol_v6(value)


def test_v6_blocks_every_enabled_action_and_has_no_runtime_dependency() -> None:
    protocol = protocol_v6.load_protocol_v6(ROOT / "config/recommendation_protocol_v6.yaml")
    for action in ("download data", "freeze input", "execute candidate", "authorize strict OOS"):
        with pytest.raises(protocol_v6.ProtocolV6Error, match="unverified"):
            protocol_v6.require_protocol_v6_access_verified(protocol, action)
    source = inspect.getsource(protocol_v6)
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
