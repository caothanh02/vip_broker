from __future__ import annotations

import inspect
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from trading_bot.recommendations import protocol_v10

ROOT = Path(__file__).resolve().parents[2]


def _raw() -> dict[str, object]:
    value = yaml.safe_load((ROOT / "config/recommendation_protocol_v10.yaml").read_text())
    assert isinstance(value, dict)
    return value


def test_v10_is_non_executable_and_only_authorizes_one_metadata_identity_probe() -> None:
    protocol = protocol_v10.load_protocol_v10(ROOT / "config/recommendation_protocol_v10.yaml")
    assert protocol.status == protocol_v10.PROTOCOL_V10_STATUS
    assert protocol.executable is False
    assert protocol.maximum_request_count == 1
    protocol_v10.require_protocol_v10_metadata_identity_probe(protocol)


@pytest.mark.parametrize(
    "path",
    [
        ("data_source", "provider"),
        ("data_source", "dataset_url_template"),
        ("data_source", "metadata_endpoint"),
        ("data_source", "candle_aggregation"),
        ("entitlement_context", "historical_access_verified"),
        ("entitlement_context", "metadata_probe_proves_historical_access"),
        ("metadata_identity_probe", "response_contract", "expected_type"),
        ("metadata_identity_probe", "response_contract", "available_since_must_be_at_or_before"),
        ("raw_trade_boundary_policy", "source_partition_timestamp"),
        ("raw_trade_boundary_policy", "finite_boundary_partition_set_preregistered"),
        ("proposed_independent_range", "end"),
        ("metadata_identity_probe", "maximum_request_count"),
        ("candidate",),
        ("strict_oos", "evaluation_authorized"),
    ],
)
def test_v10_rejects_preregistration_mutation(path: tuple[str, ...]) -> None:
    value = deepcopy(_raw())
    target: object = value
    for key in path[:-1]:
        assert isinstance(target, dict)
        target = target[key]
    assert isinstance(target, dict)
    target[path[-1]] = "unsafe"
    with pytest.raises(protocol_v10.ProtocolV10Error, match="immutable"):
        protocol_v10.validate_protocol_v10(value)


def test_v10_blocks_data_aggregation_research_selection_and_oos() -> None:
    protocol = protocol_v10.load_protocol_v10(ROOT / "config/recommendation_protocol_v10.yaml")
    for action in (
        "download trades",
        "choose raw-trade partitions",
        "aggregate candles",
        "freeze an input",
        "execute research",
        "select a policy",
        "authorize strict OOS",
    ):
        with pytest.raises(protocol_v10.ProtocolV10Error, match="cannot"):
            protocol_v10.require_protocol_v10_blocked(protocol, action)


def test_v10_governance_has_no_runtime_or_credential_dependency() -> None:
    source = inspect.getsource(protocol_v10)
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
