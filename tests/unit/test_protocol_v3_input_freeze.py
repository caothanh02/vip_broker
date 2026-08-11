"""Tests for the isolated, committed Protocol V3 input-freeze bundle."""

from __future__ import annotations

import inspect
import json
import shutil
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from hashlib import sha256
from pathlib import Path

import pytest

from trading_bot.cli import main
from trading_bot.data.csv_store import csv_sha256, write_candles_atomic
from trading_bot.domain.models import Candle
from trading_bot.recommendations import protocol_v3, v3_input_freeze

START = datetime(2019, 1, 1, tzinfo=UTC)
END = datetime(2022, 1, 1, tzinfo=UTC)
COUNT = int((END - START).total_seconds() // 3600)
GENERATION_ID = "a" * 32
_ORIGINAL_PROTOCOL_SNAPSHOT = v3_input_freeze._protocol_snapshot


def _candle(index: int) -> Candle:
    opened = START + timedelta(hours=index)
    price = Decimal("100") + Decimal(index) / Decimal("1000")
    return Candle(
        open_time=opened,
        close_time=opened + timedelta(hours=1),
        symbol="BTC/USDT",
        timeframe="1h",
        open=price,
        high=price + Decimal("1"),
        low=price - Decimal("1"),
        close=price,
        volume=Decimal("1000"),
        is_closed=True,
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _metadata(csv_path: Path, anomaly_path: Path) -> dict[str, object]:
    return {
        "generation_id": GENERATION_ID,
        "internal_symbol": "BTC/USDT",
        "timeframe": "1h",
        "requested_start": "2019-01-01T00:00:00Z",
        "effective_end": "2022-01-01T00:00:00Z",
        "stored_candle_count": COUNT,
        "requested_range_candle_count": COUNT,
        "missing_candle_count": 0,
        "duplicate_candle_count": 0,
        "conflicting_candle_count": 0,
        "market_interruption_event_count": 0,
        "market_interruption_candle_count": 0,
        "contains_non_tradable_intervals": False,
        "checksum_verification_mode": "official_online",
        "csv_sha256": csv_sha256(csv_path),
        "anomaly_report": anomaly_path.name,
        "anomaly_report_sha256": csv_sha256(anomaly_path),
    }


@pytest.fixture(scope="session")
def v3_input_source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("v3-input-source")
    csv_path = root / "data/raw/btcusdt_1h_v3.csv"
    write_candles_atomic(csv_path, (_candle(index) for index in range(COUNT)))
    anomaly_path = csv_path.with_suffix(".anomalies.json")
    _write_json(
        anomaly_path,
        {
            "generation_id": GENERATION_ID,
            "policy": {"checksum_verification_mode": "official_online"},
            "market_interruptions": [],
        },
    )
    _write_json(
        csv_path.with_name(f"{csv_path.name}.metadata.json"), _metadata(csv_path, anomaly_path)
    )
    return root


@pytest.fixture
def v3_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, v3_input_source: Path) -> Path:
    shutil.copytree(v3_input_source / "data", tmp_path / "data")
    monkeypatch.chdir(tmp_path)
    return Path("data/raw/btcusdt_1h_v3.csv")


@pytest.fixture(autouse=True)
def synthetic_unfrozen_protocol_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise bundle mechanics without making the real closed V3 runnable."""

    config_path = Path(__file__).resolve().parents[2] / "config/recommendation_protocol_v3.yaml"
    content = config_path.read_bytes()
    protocol = protocol_v3.load_protocol_v3(config_path)
    snapshot = v3_input_freeze._ArtifactSnapshot(config_path, content, sha256(content).hexdigest())
    synthetic = replace(protocol, status=protocol_v3.PROTOCOL_V3_UNFROZEN_STATUS)
    monkeypatch.setattr(v3_input_freeze, "_protocol_snapshot", lambda: (synthetic, snapshot))


def _bundle() -> Path:
    return Path("reports/research/manifests/v3-independent-input")


def _metadata_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.name}.metadata.json")


def _refresh_metadata(input_path: Path) -> None:
    metadata_path = _metadata_path(input_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["csv_sha256"] = csv_sha256(input_path)
    _write_json(metadata_path, metadata)


def _paths(bundle: Path) -> tuple[Path, Path, Path]:
    marker = bundle / "commit.json"
    payload = json.loads(marker.read_text(encoding="utf-8"))
    generation = bundle / str(payload["generation_directory"])
    return marker, generation / "manifest.json", generation / "input-lock.json"


def test_closed_v3_rejects_freeze_before_input_access(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(v3_input_freeze, "_protocol_snapshot", _ORIGINAL_PROTOCOL_SNAPSHOT)
    monkeypatch.setattr(
        v3_input_freeze,
        "read_candles_bytes",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not read input")),
    )
    with pytest.raises(v3_input_freeze.ProtocolV3InputFreezeError, match="status is unsafe"):
        v3_input_freeze.freeze_protocol_v3_input(
            Path("missing.csv"), tmp_path / "reports/research/manifests/v3-independent-input"
        )


def test_closed_v3_cli_rejects_before_runtime_or_input_access(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(v3_input_freeze, "_protocol_snapshot", _ORIGINAL_PROTOCOL_SNAPSHOT)
    monkeypatch.setattr(
        "trading_bot.cli._load_non_operational_dependencies",
        lambda: (_ for _ in ()).throw(AssertionError("must not load runtime dependencies")),
    )
    assert (
        main(
            [
                "freeze-protocol-v3-input",
                "--input",
                "missing.csv",
                "--output",
                "reports/research/manifests/v3-independent-input",
            ]
        )
        == 1
    )
    assert "status is unsafe" in capsys.readouterr().err


def test_synthetic_bundle_mechanics_do_not_make_closed_v3_executable(v3_input: Path) -> None:
    bundle = _bundle()
    manifest, lock = v3_input_freeze.freeze_protocol_v3_input(v3_input, bundle)
    marker_path, manifest_path, lock_path = _paths(bundle)

    assert manifest == json.loads(manifest_path.read_text(encoding="utf-8"))
    assert lock == json.loads(lock_path.read_text(encoding="utf-8"))
    loaded_manifest, loaded_lock = v3_input_freeze.load_published_protocol_v3_input_bundle(bundle)
    assert (loaded_manifest, loaded_lock) == (manifest, lock)
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert (
        marker["protocol_config_sha256"]
        == sha256(
            (
                Path(__file__).resolve().parents[2] / "config/recommendation_protocol_v3.yaml"
            ).read_bytes()
        ).hexdigest()
    )
    assert marker["manifest_sha256"] == sha256(manifest_path.read_bytes()).hexdigest()
    assert marker["input_lock_sha256"] == sha256(lock_path.read_bytes()).hexdigest()
    assert marker["dataset"] == {
        "generation_id": GENERATION_ID,
        "csv_sha256": manifest["dataset"]["csv_sha256"],
        "metadata_sha256": manifest["dataset"]["metadata_sha256"],
        "anomaly_sha256": manifest["dataset"]["anomaly_sha256"],
    }
    assert manifest["input_status"] == "frozen_independent_input_not_executable"
    assert manifest["strict_oos_evaluation_history"] is False
    assert manifest["research_claim_eligible"] is False
    protocol = protocol_v3.load_protocol_v3(
        Path(__file__).resolve().parents[2] / "config/recommendation_protocol_v3.yaml"
    )
    assert protocol.status == protocol_v3.PROTOCOL_V3_CLOSED_STATUS
    assert protocol.executable is False


@pytest.mark.parametrize(
    ("field", "value"),
    [("requested_start", "2022-01-01T00:00:00Z"), ("effective_end", "2025-01-01T00:00:00Z")],
)
def test_freeze_rejects_exhausted_or_oos_range_before_reading_candles(
    v3_input: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: str
) -> None:
    metadata_path = _metadata_path(v3_input)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata[field] = value
    _write_json(metadata_path, metadata)
    monkeypatch.setattr(
        v3_input_freeze,
        "read_candles_bytes",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not read candles")),
    )
    with pytest.raises(v3_input_freeze.ProtocolV3InputFreezeError, match="independent target"):
        v3_input_freeze.freeze_protocol_v3_input(v3_input, _bundle())


@pytest.mark.parametrize("mutation", ["open", "gap"])
def test_freeze_rejects_open_or_gapped_candles(v3_input: Path, mutation: str) -> None:
    lines = v3_input.read_text(encoding="utf-8").splitlines()
    if mutation == "open":
        fields = lines[-1].split(",")
        fields[-1] = "false"
        lines[-1] = ",".join(fields)
    else:
        lines.pop(2)
    v3_input.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _refresh_metadata(v3_input)
    with pytest.raises(v3_input_freeze.ProtocolV3InputFreezeError, match="candle validation"):
        v3_input_freeze.freeze_protocol_v3_input(v3_input, _bundle())


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
@pytest.mark.parametrize("sidecar", ["metadata", "anomaly"])
def test_freeze_rejects_nonstandard_json_constants(
    v3_input: Path, constant: str, sidecar: str
) -> None:
    path = (
        _metadata_path(v3_input)
        if sidecar == "metadata"
        else v3_input.with_suffix(".anomalies.json")
    )
    path.write_text('{"unused":' + constant + "}", encoding="utf-8")
    with pytest.raises(v3_input_freeze.ProtocolV3InputFreezeError, match="strict JSON"):
        v3_input_freeze.freeze_protocol_v3_input(v3_input, _bundle())


def test_freeze_rejects_duplicate_key_and_untrusted_anomaly_path(v3_input: Path) -> None:
    metadata_path = _metadata_path(v3_input)
    metadata_path.write_text('{"anomaly_report":"a","anomaly_report":"b"}', encoding="utf-8")
    with pytest.raises(v3_input_freeze.ProtocolV3InputFreezeError, match="strict JSON"):
        v3_input_freeze.freeze_protocol_v3_input(v3_input, _bundle())
    metadata = _metadata(v3_input, v3_input.with_suffix(".anomalies.json"))
    metadata["anomaly_report"] = "../external.anomalies.json"
    _write_json(metadata_path, metadata)
    with pytest.raises(v3_input_freeze.ProtocolV3InputFreezeError, match="canonical anomaly"):
        v3_input_freeze.freeze_protocol_v3_input(v3_input, _bundle())


@pytest.mark.parametrize("target", ["csv", "metadata", "anomaly"])
def test_freeze_rejects_symlinked_input_or_sidecar_before_snapshot(
    v3_input: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    original_is_symlink = Path.is_symlink
    unsafe_path = {
        "csv": Path.cwd() / v3_input,
        "metadata": Path.cwd() / _metadata_path(v3_input),
        "anomaly": Path.cwd() / v3_input.with_suffix(".anomalies.json"),
    }[target].absolute()

    def is_symlink(path: Path) -> bool:
        return path.absolute() == unsafe_path or original_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", is_symlink)
    with pytest.raises(v3_input_freeze.ProtocolV3InputFreezeError, match="symlink or reparse"):
        v3_input_freeze.freeze_protocol_v3_input(v3_input, _bundle())


@pytest.mark.parametrize("target", ["csv", "generation"])
def test_freeze_rejects_checksum_or_generation_mismatch(v3_input: Path, target: str) -> None:
    metadata_path = _metadata_path(v3_input)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if target == "csv":
        metadata["csv_sha256"] = "0" * 64
    else:
        metadata["generation_id"] = "b" * 32
    _write_json(metadata_path, metadata)
    with pytest.raises(v3_input_freeze.ProtocolV3InputFreezeError, match="checksum|generation"):
        v3_input_freeze.freeze_protocol_v3_input(v3_input, _bundle())


def test_freeze_rechecks_config_snapshot_before_publish(
    v3_input: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    protocol, snapshot = v3_input_freeze._protocol_snapshot()
    config_path = tmp_path / "protocol.yaml"
    config_path.write_bytes(snapshot.content)
    copied_snapshot = v3_input_freeze._ArtifactSnapshot(
        config_path, snapshot.content, snapshot.digest
    )
    monkeypatch.setattr(v3_input_freeze, "_protocol_snapshot", lambda: (protocol, copied_snapshot))
    original_recheck = v3_input_freeze._recheck_snapshot

    def mutate_config(candidate: v3_input_freeze._ArtifactSnapshot, label: str) -> None:
        if label == "Protocol V3 configuration":
            candidate.path.write_bytes(candidate.content + b"# changed\n")
        original_recheck(candidate, label)

    monkeypatch.setattr(v3_input_freeze, "_recheck_snapshot", mutate_config)
    bundle = _bundle()
    with pytest.raises(
        v3_input_freeze.ProtocolV3InputFreezeError, match="changed after validation"
    ):
        v3_input_freeze.freeze_protocol_v3_input(v3_input, bundle)
    assert not (bundle / "commit.json").exists()
    with pytest.raises(v3_input_freeze.ProtocolV3InputFreezeError):
        v3_input_freeze.load_published_protocol_v3_input_bundle(bundle)


def test_freeze_rechecks_input_snapshot_before_publish_and_leaves_no_bundle(
    v3_input: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_recheck = v3_input_freeze._recheck_snapshot
    mutated = False

    def mutate_before_recheck(snapshot: v3_input_freeze._ArtifactSnapshot, label: str) -> None:
        nonlocal mutated
        if not mutated and label == "Protocol V3 CSV input":
            snapshot.path.write_bytes(snapshot.content + b"changed")
            mutated = True
        original_recheck(snapshot, label)

    monkeypatch.setattr(v3_input_freeze, "_recheck_snapshot", mutate_before_recheck)
    bundle = _bundle()
    with pytest.raises(
        v3_input_freeze.ProtocolV3InputFreezeError, match="changed after validation"
    ):
        v3_input_freeze.freeze_protocol_v3_input(v3_input, bundle)
    assert not (bundle / "commit.json").exists()
    with pytest.raises(v3_input_freeze.ProtocolV3InputFreezeError):
        v3_input_freeze.load_published_protocol_v3_input_bundle(bundle)


def test_incomplete_generation_is_not_a_published_bundle(
    v3_input: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _bundle()
    original_write = v3_input_freeze._write_exclusive

    def fail_between_payloads(path: Path, content: bytes, label: str) -> None:
        if path.name == "input-lock.json":
            raise v3_input_freeze.ProtocolV3InputFreezeError("interrupted payload publication")
        original_write(path, content, label)

    monkeypatch.setattr(v3_input_freeze, "_write_exclusive", fail_between_payloads)
    with pytest.raises(v3_input_freeze.ProtocolV3InputFreezeError, match="interrupted"):
        v3_input_freeze.freeze_protocol_v3_input(v3_input, bundle)
    assert bundle.exists()
    assert not (bundle / "commit.json").exists()
    with pytest.raises(v3_input_freeze.ProtocolV3InputFreezeError):
        v3_input_freeze.load_published_protocol_v3_input_bundle(bundle)


@pytest.mark.parametrize("tamper", ["missing", "manifest_hash", "generation"])
def test_consumer_rejects_missing_or_tampered_commit_marker(v3_input: Path, tamper: str) -> None:
    bundle = _bundle()
    v3_input_freeze.freeze_protocol_v3_input(v3_input, bundle)
    marker_path, _, _ = _paths(bundle)
    if tamper == "missing":
        marker_path.unlink()
    else:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker["manifest_sha256" if tamper == "manifest_hash" else "bundle_generation_id"] = (
            "0" * 32
        )
        _write_json(marker_path, marker)
    with pytest.raises(v3_input_freeze.ProtocolV3InputFreezeError):
        v3_input_freeze.load_published_protocol_v3_input_bundle(bundle)


@pytest.mark.parametrize("target", ["marker", "manifest", "input_lock"])
@pytest.mark.parametrize("content", ["{", '{"key":"x","key":"y"}', '{"unused":NaN}'])
def test_consumer_rejects_non_strict_json_in_every_bundle_artifact(
    v3_input: Path, target: str, content: str
) -> None:
    bundle = _bundle()
    v3_input_freeze.freeze_protocol_v3_input(v3_input, bundle)
    marker, manifest, input_lock = _paths(bundle)
    {"marker": marker, "manifest": manifest, "input_lock": input_lock}[target].write_text(
        content, encoding="utf-8"
    )
    with pytest.raises(v3_input_freeze.ProtocolV3InputFreezeError, match="strict JSON"):
        v3_input_freeze.load_published_protocol_v3_input_bundle(bundle)


def test_target_appearing_before_commit_marker_is_not_overwritten(
    v3_input: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _bundle()
    original_write = v3_input_freeze._write_exclusive

    def create_competing_marker(path: Path, content: bytes, label: str) -> None:
        if path.name == "commit.json":
            path.write_text('{"external":true}', encoding="utf-8")
        original_write(path, content, label)

    monkeypatch.setattr(v3_input_freeze, "_write_exclusive", create_competing_marker)
    with pytest.raises(v3_input_freeze.ProtocolV3InputFreezeError, match="already exists"):
        v3_input_freeze.freeze_protocol_v3_input(v3_input, bundle)
    assert json.loads((bundle / "commit.json").read_text(encoding="utf-8")) == {"external": True}
    with pytest.raises(v3_input_freeze.ProtocolV3InputFreezeError):
        v3_input_freeze.load_published_protocol_v3_input_bundle(bundle)


def test_freeze_rejects_unsafe_or_existing_bundle(v3_input: Path) -> None:
    with pytest.raises(v3_input_freeze.ProtocolV3InputFreezeError, match="without traversal"):
        v3_input_freeze.freeze_protocol_v3_input(
            v3_input, Path("reports/research/manifests/../escape")
        )
    with pytest.raises(v3_input_freeze.ProtocolV3InputFreezeError, match="without extension"):
        v3_input_freeze.freeze_protocol_v3_input(
            v3_input, Path("reports/research/manifests/v3.json")
        )
    bundle = _bundle()
    bundle.parent.mkdir(parents=True, exist_ok=True)
    bundle.mkdir()
    with pytest.raises(v3_input_freeze.ProtocolV3InputFreezeError, match="already exists"):
        v3_input_freeze.freeze_protocol_v3_input(v3_input, bundle)


def test_freeze_rejects_symlinked_output_before_snapshot(
    v3_input: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _bundle()
    original_is_symlink = Path.is_symlink
    unsafe_path = (Path.cwd() / bundle).absolute()

    def is_symlink(path: Path) -> bool:
        return path.absolute() == unsafe_path or original_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", is_symlink)
    with pytest.raises(v3_input_freeze.ProtocolV3InputFreezeError, match="symlink or reparse"):
        v3_input_freeze.freeze_protocol_v3_input(v3_input, bundle)


def test_freeze_rejects_reparse_input_before_snapshot(
    v3_input: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_reparse = v3_input_freeze._is_reparse_point
    unsafe_path = (Path.cwd() / v3_input).absolute()

    def is_reparse(path: Path) -> bool:
        return path.absolute() == unsafe_path or original_reparse(path)

    monkeypatch.setattr(v3_input_freeze, "_is_reparse_point", is_reparse)
    with pytest.raises(v3_input_freeze.ProtocolV3InputFreezeError, match="symlink or reparse"):
        v3_input_freeze.freeze_protocol_v3_input(v3_input, _bundle())


def test_cli_isolated_freeze_does_not_load_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = (
        {
            "protocol_id": "recommendation_research_v3",
            "input_status": "frozen_independent_input_not_executable",
            "dataset": {"candle_count": 26_304},
            "safety_locks": {
                "live_trading_enabled": False,
                "broker_used": False,
                "orders_submitted": False,
                "ml_used": False,
                "network_used": False,
            },
        },
        {"frozen_manifest_sha256": "a" * 64},
    )
    monkeypatch.setattr(
        "trading_bot.recommendations.v3_input_freeze.freeze_protocol_v3_input",
        lambda *args, **kwargs: result,
    )
    monkeypatch.setattr(
        "trading_bot.cli._load_non_operational_dependencies",
        lambda: (_ for _ in ()).throw(AssertionError("must not load runtime dependencies")),
    )
    assert (
        main(
            [
                "freeze-protocol-v3-input",
                "--input",
                str(tmp_path / "input.csv"),
                "--output",
                "reports/research/manifests/v3-independent-input",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["v3_executable"] is False
    assert payload["safety_locks"]["network_used"] is False


def test_v3_input_freeze_has_no_execution_model_or_network_dependencies() -> None:
    source = inspect.getsource(v3_input_freeze)
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
    ):
        assert forbidden not in source
