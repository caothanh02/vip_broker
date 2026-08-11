"""Freeze a future independent Protocol V3 input without authorizing execution.

This module validates local byte snapshots and publishes a committed bundle only.
It has no strategy, model, broker, order, settings, credential, or network dependency.
"""

from __future__ import annotations

import json
import os
import stat
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml

from trading_bot.data.csv_store import CsvDataError, read_candles_bytes
from trading_bot.data.validation import CandleValidationError, validate_candles
from trading_bot.recommendations.protocol_v3 import (
    PROTOCOL_V3_UNFROZEN_STATUS,
    ProtocolV3,
    ProtocolV3Error,
    validate_protocol_v3,
    validate_protocol_v3_input_lock,
)

_MANIFEST_DIRECTORY = Path("reports/research/manifests")
_MANIFEST_SCHEMA_VERSION = "1.0"
_INPUT_LOCK_SCHEMA_VERSION = "1.0"
_BUNDLE_COMMIT_SCHEMA_VERSION = "1.0"
_COMMIT_MARKER = "commit.json"
_MANIFEST_NAME = "manifest.json"
_INPUT_LOCK_NAME = "input-lock.json"


class ProtocolV3InputFreezeError(ValueError):
    """A local candidate input cannot be frozen as Protocol V3 provenance."""


class _DuplicateJsonKeyError(ValueError):
    pass


class _DuplicateYamlKeyError(ValueError):
    pass


class _NonStandardJsonConstantError(ValueError):
    pass


class _StrictYamlLoader(yaml.SafeLoader):
    pass


def _yaml_mapping(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode, deep: bool = False) -> Any:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise _DuplicateYamlKeyError(str(key))
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictYamlLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _yaml_mapping)


@dataclass(frozen=True, slots=True)
class _ArtifactSnapshot:
    path: Path
    content: bytes
    digest: str


def _workspace() -> Path:
    path = Path.cwd().absolute()
    _assert_safe_existing_ancestors(path, "workspace")
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise ProtocolV3InputFreezeError("could not resolve workspace") from exc


def _is_reparse_point(path: Path) -> bool:
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ProtocolV3InputFreezeError(f"could not inspect {path}") from exc
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _assert_safe_existing_ancestors(path: Path, label: str) -> None:
    """Reject symlinks and Windows reparse points before following a path."""

    absolute = path.absolute()
    for ancestor in (absolute, *absolute.parents):
        try:
            exists = ancestor.exists() or ancestor.is_symlink()
        except OSError as exc:
            raise ProtocolV3InputFreezeError(f"could not inspect {label} ancestor") from exc
        if not exists:
            continue
        if ancestor.is_symlink() or _is_reparse_point(ancestor):
            raise ProtocolV3InputFreezeError(f"{label} must not use a symlink or reparse point")


def _workspace_relative_path(value: Path, label: str, *, file_required: bool) -> Path:
    if value.is_absolute() or value.drive or ".." in value.parts:
        raise ProtocolV3InputFreezeError(
            f"{label} must be a relative workspace path without traversal"
        )
    workspace = _workspace()
    unresolved = workspace / value
    _assert_safe_existing_ancestors(unresolved, label)
    try:
        unresolved.resolve(strict=False).relative_to(workspace)
    except (OSError, ValueError) as exc:
        raise ProtocolV3InputFreezeError(f"{label} must remain inside the workspace") from exc
    if file_required:
        _assert_regular_file(unresolved, label)
    return unresolved


def _assert_regular_file(path: Path, label: str) -> None:
    try:
        if path.is_symlink() or _is_reparse_point(path) or not path.is_file():
            raise ProtocolV3InputFreezeError(f"{label} must be a regular non-symlink file")
    except OSError as exc:
        raise ProtocolV3InputFreezeError(f"could not inspect {label}") from exc


def _assert_regular_directory(path: Path, label: str) -> None:
    try:
        if path.is_symlink() or _is_reparse_point(path) or not path.is_dir():
            raise ProtocolV3InputFreezeError(f"{label} must be a regular non-symlink directory")
    except OSError as exc:
        raise ProtocolV3InputFreezeError(f"could not inspect {label}") from exc


def _canonical_sidecars(input_path: Path) -> tuple[Path, Path]:
    metadata = input_path.with_name(f"{input_path.name}.metadata.json")
    anomaly = input_path.with_suffix(".anomalies.json")
    for path, label in ((metadata, "metadata sidecar"), (anomaly, "anomaly sidecar")):
        _assert_safe_existing_ancestors(path, label)
        _assert_regular_file(path, label)
    return metadata, anomaly


def _snapshot(path: Path, label: str) -> _ArtifactSnapshot:
    _assert_safe_existing_ancestors(path, label)
    _assert_regular_file(path, label)
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise ProtocolV3InputFreezeError(f"could not read {label}") from exc
    return _ArtifactSnapshot(path, content, sha256(content).hexdigest())


def _json_object(snapshot: _ArtifactSnapshot, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise _DuplicateJsonKeyError(key)
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise _NonStandardJsonConstantError(value)

    try:
        value = json.loads(
            snapshot.content.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        _DuplicateJsonKeyError,
        _NonStandardJsonConstantError,
    ) as exc:
        raise ProtocolV3InputFreezeError(
            f"{label} must be strict JSON without duplicate keys or non-finite constants"
        ) from exc
    if not isinstance(value, dict):
        raise ProtocolV3InputFreezeError(f"{label} must be a JSON object")
    return value


def _protocol_snapshot() -> tuple[ProtocolV3, _ArtifactSnapshot]:
    config = Path(__file__).resolve().parents[3] / "config/recommendation_protocol_v3.yaml"
    snapshot = _snapshot(config, "Protocol V3 configuration")
    try:
        raw = yaml.load(snapshot.content.decode("utf-8"), Loader=_StrictYamlLoader)
        protocol = validate_protocol_v3(raw)
    except (UnicodeDecodeError, yaml.YAMLError, _DuplicateYamlKeyError, ProtocolV3Error) as exc:
        raise ProtocolV3InputFreezeError("Protocol V3 configuration is invalid") from exc
    if protocol.status != PROTOCOL_V3_UNFROZEN_STATUS or protocol.executable:
        raise ProtocolV3InputFreezeError("Protocol V3 preregistration status is unsafe")
    return protocol, snapshot


def _iso_utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_utc_hour(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise ProtocolV3InputFreezeError(f"{label} must be a UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProtocolV3InputFreezeError(f"{label} is invalid") from exc
    if (
        parsed.tzinfo is None
        or parsed.utcoffset() != timedelta(0)
        or parsed.minute
        or parsed.second
        or parsed.microsecond
    ):
        raise ProtocolV3InputFreezeError(f"{label} must be a UTC hour")
    return parsed.astimezone(UTC)


def _nonnegative_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProtocolV3InputFreezeError(f"metadata {key} is invalid")
    return value


def _expected_candle_count(protocol: ProtocolV3) -> int:
    seconds = (protocol.development_end - protocol.development_start).total_seconds()
    if seconds <= 0 or seconds % 3600:
        raise ProtocolV3InputFreezeError("Protocol V3 target range is invalid")
    return int(seconds // 3600)


def _validate_metadata_before_candle_read(
    metadata: dict[str, Any],
    anomaly: dict[str, Any],
    snapshots: tuple[_ArtifactSnapshot, ...],
    protocol: ProtocolV3,
) -> str:
    csv_snapshot, _, anomaly_snapshot = snapshots
    if metadata.get("anomaly_report") != anomaly_snapshot.path.name:
        raise ProtocolV3InputFreezeError(
            "canonical anomaly sidecar identity does not match metadata"
        )
    if metadata.get("csv_sha256") != csv_snapshot.digest:
        raise ProtocolV3InputFreezeError("metadata CSV checksum does not match snapshot")
    if metadata.get("anomaly_report_sha256") != anomaly_snapshot.digest:
        raise ProtocolV3InputFreezeError("metadata anomaly checksum does not match snapshot")
    generation = metadata.get("generation_id")
    if not isinstance(generation, str) or generation != anomaly.get("generation_id"):
        raise ProtocolV3InputFreezeError("metadata and anomaly generation identity does not match")
    if (
        metadata.get("internal_symbol") != "BTC/USDT"
        or metadata.get("timeframe") != "1h"
        or metadata.get("checksum_verification_mode") != "official_online"
    ):
        raise ProtocolV3InputFreezeError("metadata market identity or checksum mode is invalid")
    policy = anomaly.get("policy")
    if (
        not isinstance(policy, dict)
        or policy.get("checksum_verification_mode") != "official_online"
    ):
        raise ProtocolV3InputFreezeError("anomaly checksum mode is invalid")
    if (
        _parse_utc_hour(metadata.get("requested_start"), "metadata requested_start")
        != protocol.development_start
    ):
        raise ProtocolV3InputFreezeError("input range is not the Protocol V3 independent target")
    if (
        _parse_utc_hour(metadata.get("effective_end"), "metadata effective_end")
        != protocol.development_end
    ):
        raise ProtocolV3InputFreezeError("input range is not the Protocol V3 independent target")
    interruptions = anomaly.get("market_interruptions")
    if not isinstance(interruptions, list) or interruptions:
        raise ProtocolV3InputFreezeError(
            "Protocol V3 independent input must not contain interruptions"
        )
    if (
        metadata.get("contains_non_tradable_intervals") is not False
        or _nonnegative_int(metadata, "market_interruption_event_count") != 0
        or _nonnegative_int(metadata, "market_interruption_candle_count") != 0
        or _nonnegative_int(metadata, "missing_candle_count") != 0
    ):
        raise ProtocolV3InputFreezeError("Protocol V3 independent input must not contain gaps")
    if (
        _nonnegative_int(metadata, "duplicate_candle_count") != 0
        or _nonnegative_int(metadata, "conflicting_candle_count") != 0
    ):
        raise ProtocolV3InputFreezeError("metadata records duplicate or conflicting candles")
    expected_count = _expected_candle_count(protocol)
    if _nonnegative_int(metadata, "stored_candle_count") != expected_count:
        raise ProtocolV3InputFreezeError("metadata candle count does not match Protocol V3 target")
    if _nonnegative_int(metadata, "requested_range_candle_count") != expected_count:
        raise ProtocolV3InputFreezeError(
            "metadata requested count does not match Protocol V3 target"
        )
    return generation


def _recheck_snapshot(snapshot: _ArtifactSnapshot, label: str) -> None:
    current = _snapshot(snapshot.path, label)
    if current.digest != snapshot.digest:
        raise ProtocolV3InputFreezeError(f"{label} changed after validation")


def _namespace() -> Path:
    workspace = _workspace()
    namespace = workspace / _MANIFEST_DIRECTORY
    _assert_safe_existing_ancestors(namespace, "Protocol V3 manifest namespace")
    return namespace


def _bundle_path(value: Path, *, must_exist: bool) -> Path:
    path = _workspace_relative_path(value, "Protocol V3 bundle output", file_required=False)
    if path.suffix:
        raise ProtocolV3InputFreezeError(
            "Protocol V3 bundle output must be a directory without extension"
        )
    namespace = _namespace()
    try:
        if path.parent.resolve(strict=False) != namespace.resolve(strict=False):
            raise ProtocolV3InputFreezeError(
                "Protocol V3 bundle output must be a direct child of reports/research/manifests"
            )
    except OSError as exc:
        raise ProtocolV3InputFreezeError("could not inspect Protocol V3 bundle output") from exc
    if must_exist:
        _assert_safe_existing_ancestors(path, "Protocol V3 bundle output")
        _assert_regular_directory(path, "Protocol V3 bundle output")
    elif path.exists() or path.is_symlink() or _is_reparse_point(path):
        raise ProtocolV3InputFreezeError("Protocol V3 bundle output already exists or is unsafe")
    return path


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _write_exclusive(path: Path, content: bytes, label: str) -> None:
    try:
        descriptor = os.open(
            path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600
        )
    except FileExistsError as exc:
        raise ProtocolV3InputFreezeError(f"{label} already exists") from exc
    except OSError as exc:
        raise ProtocolV3InputFreezeError(f"could not create {label}") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise ProtocolV3InputFreezeError(f"could not write {label}") from exc


def _prepare_bundle_root(bundle: Path) -> None:
    namespace = _namespace()
    try:
        namespace.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ProtocolV3InputFreezeError("could not create Protocol V3 manifest namespace") from exc
    _assert_safe_existing_ancestors(namespace, "Protocol V3 manifest namespace")
    _assert_regular_directory(namespace, "Protocol V3 manifest namespace")
    try:
        bundle.mkdir()
    except FileExistsError as exc:
        raise ProtocolV3InputFreezeError("Protocol V3 bundle output already exists") from exc
    except OSError as exc:
        raise ProtocolV3InputFreezeError("could not reserve Protocol V3 bundle output") from exc
    _assert_safe_existing_ancestors(bundle, "Protocol V3 bundle output")
    _assert_regular_directory(bundle, "Protocol V3 bundle output")


def _generation_directory(bundle: Path, token: str) -> Path:
    stage = bundle / f".staging-{token}"
    try:
        stage.mkdir()
    except OSError as exc:
        raise ProtocolV3InputFreezeError("could not create Protocol V3 staging generation") from exc
    _assert_safe_existing_ancestors(stage, "Protocol V3 staging generation")
    _assert_regular_directory(stage, "Protocol V3 staging generation")
    return stage


def _marker_payload(
    protocol: ProtocolV3,
    config_digest: str,
    generation_name: str,
    manifest_content: bytes,
    input_lock_content: bytes,
    dataset: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": _BUNDLE_COMMIT_SCHEMA_VERSION,
        "protocol_id": protocol.protocol_id,
        "bundle_generation_id": generation_name.removeprefix("generation-"),
        "generation_directory": generation_name,
        "protocol_config_sha256": config_digest,
        "manifest_sha256": sha256(manifest_content).hexdigest(),
        "input_lock_sha256": sha256(input_lock_content).hexdigest(),
        "dataset": dataset,
    }


def _publish_bundle(
    bundle: Path,
    protocol: ProtocolV3,
    config_snapshot: _ArtifactSnapshot,
    input_snapshots: tuple[_ArtifactSnapshot, ...],
    manifest_content: bytes,
    input_lock_content: bytes,
    dataset: dict[str, Any],
) -> None:
    _prepare_bundle_root(bundle)
    token = uuid.uuid4().hex
    stage = _generation_directory(bundle, token)
    _write_exclusive(stage / _MANIFEST_NAME, manifest_content, "Protocol V3 staged manifest")
    _write_exclusive(stage / _INPUT_LOCK_NAME, input_lock_content, "Protocol V3 staged input lock")
    _recheck_snapshot(config_snapshot, "Protocol V3 configuration")
    for snapshot, label in zip(
        input_snapshots,
        ("Protocol V3 CSV input", "metadata sidecar", "anomaly sidecar"),
        strict=True,
    ):
        _recheck_snapshot(snapshot, label)
    generation_name = f"generation-{token}"
    generation = bundle / generation_name
    try:
        os.replace(stage, generation)
    except OSError as exc:
        raise ProtocolV3InputFreezeError("could not finalize Protocol V3 generation") from exc
    _assert_safe_existing_ancestors(generation, "Protocol V3 generation")
    _assert_regular_directory(generation, "Protocol V3 generation")
    marker = _marker_payload(
        protocol,
        config_snapshot.digest,
        generation_name,
        manifest_content,
        input_lock_content,
        dataset,
    )
    _write_exclusive(bundle / _COMMIT_MARKER, _json_bytes(marker), "Protocol V3 commit marker")


def _require_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolV3InputFreezeError(f"{label} must be an object")
    return value


def _require_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ProtocolV3InputFreezeError(f"{label} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ProtocolV3InputFreezeError(f"{label} must be a SHA-256 digest") from exc
    return value


def _validate_committed_bundle(
    marker: dict[str, Any],
    manifest: dict[str, Any],
    input_lock: dict[str, Any],
    protocol: ProtocolV3,
    config_digest: str,
    manifest_snapshot: _ArtifactSnapshot,
    lock_snapshot: _ArtifactSnapshot,
) -> None:
    expected_marker = {
        "schema_version",
        "protocol_id",
        "bundle_generation_id",
        "generation_directory",
        "protocol_config_sha256",
        "manifest_sha256",
        "input_lock_sha256",
        "dataset",
    }
    if set(marker) != expected_marker or marker["schema_version"] != _BUNDLE_COMMIT_SCHEMA_VERSION:
        raise ProtocolV3InputFreezeError("Protocol V3 commit marker schema is invalid")
    if marker["protocol_id"] != protocol.protocol_id:
        raise ProtocolV3InputFreezeError("Protocol V3 commit marker protocol is invalid")
    if (
        _require_sha(marker["protocol_config_sha256"], "commit marker config digest")
        != config_digest
    ):
        raise ProtocolV3InputFreezeError("Protocol V3 commit marker config digest is invalid")
    if (
        _require_sha(marker["manifest_sha256"], "commit marker manifest digest")
        != manifest_snapshot.digest
    ):
        raise ProtocolV3InputFreezeError("Protocol V3 commit marker manifest digest is invalid")
    if (
        _require_sha(marker["input_lock_sha256"], "commit marker input lock digest")
        != lock_snapshot.digest
    ):
        raise ProtocolV3InputFreezeError("Protocol V3 commit marker input lock digest is invalid")
    directory = marker["generation_directory"]
    generation_id = marker["bundle_generation_id"]
    if (
        not isinstance(directory, str)
        or not isinstance(generation_id, str)
        or directory != f"generation-{generation_id}"
        or len(generation_id) != 32
    ):
        raise ProtocolV3InputFreezeError("Protocol V3 commit marker generation is invalid")
    try:
        int(generation_id, 16)
    except ValueError as exc:
        raise ProtocolV3InputFreezeError("Protocol V3 commit marker generation is invalid") from exc
    dataset = _require_mapping(marker["dataset"], "commit marker dataset")
    manifest_dataset = _require_mapping(manifest.get("dataset"), "manifest dataset")
    if (
        set(manifest)
        != {
            "schema_version",
            "protocol_id",
            "research_role",
            "strict_oos_evaluation_history",
            "research_claim_eligible",
            "input_status",
            "dataset",
            "market_interruptions",
            "safety_locks",
        }
        or manifest.get("schema_version") != _MANIFEST_SCHEMA_VERSION
    ):
        raise ProtocolV3InputFreezeError("Protocol V3 manifest schema is invalid")
    if set(manifest_dataset) != {
        "path",
        "symbol",
        "timeframe",
        "range",
        "candle_count",
        "generation_id",
        "csv_sha256",
        "metadata_sha256",
        "anomaly_sha256",
        "validation_status",
        "checksum_verification_mode",
    }:
        raise ProtocolV3InputFreezeError("Protocol V3 manifest dataset schema is invalid")
    if dataset != {
        "generation_id": manifest_dataset.get("generation_id"),
        "csv_sha256": manifest_dataset.get("csv_sha256"),
        "metadata_sha256": manifest_dataset.get("metadata_sha256"),
        "anomaly_sha256": manifest_dataset.get("anomaly_sha256"),
    }:
        raise ProtocolV3InputFreezeError("Protocol V3 commit marker dataset is invalid")
    if (
        manifest.get("protocol_id") != protocol.protocol_id
        or manifest.get("research_role") != "development"
        or manifest.get("strict_oos_evaluation_history") is not False
        or manifest.get("research_claim_eligible") is not False
        or manifest.get("input_status") != "frozen_independent_input_not_executable"
        or manifest.get("market_interruptions") != []
        or manifest_dataset.get("symbol") != "BTC/USDT"
        or manifest_dataset.get("timeframe") != "1h"
        or manifest_dataset.get("validation_status") != "valid"
        or manifest_dataset.get("checksum_verification_mode") != "official_online"
        or manifest.get("safety_locks")
        != {
            "live_trading_enabled": False,
            "broker_used": False,
            "orders_submitted": False,
            "ml_used": False,
            "network_used": False,
        }
    ):
        raise ProtocolV3InputFreezeError("Protocol V3 manifest governance fields are invalid")
    if manifest_dataset.get("range") != {
        "start": _iso_utc(protocol.development_start),
        "end": _iso_utc(protocol.development_end),
    } or manifest_dataset.get("candle_count") != _expected_candle_count(protocol):
        raise ProtocolV3InputFreezeError("Protocol V3 manifest range is invalid")
    try:
        validate_protocol_v3_input_lock(input_lock, protocol, config_digest)
    except ProtocolV3Error as exc:
        raise ProtocolV3InputFreezeError("Protocol V3 input lock is invalid") from exc
    lock_dataset = _require_mapping(input_lock.get("dataset"), "input lock dataset")
    expected_lock_dataset: dict[str, Any] = {
        "generation_id": manifest_dataset.get("generation_id"),
        "csv_sha256": manifest_dataset.get("csv_sha256"),
        "metadata_sha256": manifest_dataset.get("metadata_sha256"),
        "anomaly_sha256": manifest_dataset.get("anomaly_sha256"),
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "utc_range": manifest_dataset.get("range"),
        "audited_interruption_ids": [],
    }
    if lock_dataset != expected_lock_dataset:
        raise ProtocolV3InputFreezeError("Protocol V3 input lock dataset does not match manifest")


def load_published_protocol_v3_input_bundle(
    bundle_output: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read only a complete committed V3 input bundle; orphan generations are invalid."""

    protocol, config_snapshot = _protocol_snapshot()
    bundle = _bundle_path(bundle_output, must_exist=True)
    marker_snapshot = _snapshot(bundle / _COMMIT_MARKER, "Protocol V3 commit marker")
    marker = _json_object(marker_snapshot, "Protocol V3 commit marker")
    directory = marker.get("generation_directory")
    if not isinstance(directory, str) or Path(directory).name != directory:
        raise ProtocolV3InputFreezeError("Protocol V3 commit marker generation is invalid")
    generation = bundle / directory
    _assert_safe_existing_ancestors(generation, "Protocol V3 generation")
    _assert_regular_directory(generation, "Protocol V3 generation")
    entries = {entry.name for entry in bundle.iterdir()}
    if entries != {_COMMIT_MARKER, directory}:
        raise ProtocolV3InputFreezeError("Protocol V3 bundle contains uncommitted artifacts")
    generation_entries = {entry.name for entry in generation.iterdir()}
    if generation_entries != {_MANIFEST_NAME, _INPUT_LOCK_NAME}:
        raise ProtocolV3InputFreezeError("Protocol V3 generation file set is invalid")
    manifest_snapshot = _snapshot(generation / _MANIFEST_NAME, "Protocol V3 manifest")
    lock_snapshot = _snapshot(generation / _INPUT_LOCK_NAME, "Protocol V3 input lock")
    manifest = _json_object(manifest_snapshot, "Protocol V3 manifest")
    input_lock = _json_object(lock_snapshot, "Protocol V3 input lock")
    _validate_committed_bundle(
        marker,
        manifest,
        input_lock,
        protocol,
        config_snapshot.digest,
        manifest_snapshot,
        lock_snapshot,
    )
    _recheck_snapshot(config_snapshot, "Protocol V3 configuration")
    return manifest, input_lock


def freeze_protocol_v3_input(
    input_path: Path, bundle_output: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Freeze a verified independent input as an unexecutable committed V3 bundle."""

    protocol, config_snapshot = _protocol_snapshot()
    csv_path = _workspace_relative_path(input_path, "Protocol V3 CSV input", file_required=True)
    metadata_path, anomaly_path = _canonical_sidecars(csv_path)
    bundle = _bundle_path(bundle_output, must_exist=False)
    csv_snapshot = _snapshot(csv_path, "Protocol V3 CSV input")
    metadata_snapshot = _snapshot(metadata_path, "metadata sidecar")
    anomaly_snapshot = _snapshot(anomaly_path, "anomaly sidecar")
    metadata = _json_object(metadata_snapshot, "metadata sidecar")
    anomaly = _json_object(anomaly_snapshot, "anomaly sidecar")
    generation_id = _validate_metadata_before_candle_read(
        metadata, anomaly, (csv_snapshot, metadata_snapshot, anomaly_snapshot), protocol
    )
    try:
        candles = read_candles_bytes(csv_snapshot.content)
        validate_candles(candles)
    except (CsvDataError, CandleValidationError) as exc:
        raise ProtocolV3InputFreezeError("Protocol V3 input candle validation failed") from exc
    if (
        len(candles) != _expected_candle_count(protocol)
        or candles[0].open_time != protocol.development_start
        or candles[-1].close_time != protocol.development_end
    ):
        raise ProtocolV3InputFreezeError(
            "Protocol V3 input candles do not match the independent target"
        )
    workspace = _workspace()
    dataset = {
        "generation_id": generation_id,
        "csv_sha256": csv_snapshot.digest,
        "metadata_sha256": metadata_snapshot.digest,
        "anomaly_sha256": anomaly_snapshot.digest,
    }
    manifest: dict[str, Any] = {
        "schema_version": _MANIFEST_SCHEMA_VERSION,
        "protocol_id": protocol.protocol_id,
        "research_role": "development",
        "strict_oos_evaluation_history": False,
        "research_claim_eligible": False,
        "input_status": "frozen_independent_input_not_executable",
        "dataset": {
            "path": csv_path.relative_to(workspace).as_posix(),
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "range": {
                "start": _iso_utc(protocol.development_start),
                "end": _iso_utc(protocol.development_end),
            },
            "candle_count": len(candles),
            **dataset,
            "validation_status": "valid",
            "checksum_verification_mode": "official_online",
        },
        "market_interruptions": [],
        "safety_locks": {
            "live_trading_enabled": False,
            "broker_used": False,
            "orders_submitted": False,
            "ml_used": False,
            "network_used": False,
        },
    }
    manifest_content = _json_bytes(manifest)
    input_lock: dict[str, Any] = {
        "schema_version": _INPUT_LOCK_SCHEMA_VERSION,
        "protocol_id": protocol.protocol_id,
        "protocol_config_sha256": config_snapshot.digest,
        "frozen_manifest_sha256": sha256(manifest_content).hexdigest(),
        "dataset": {
            **dataset,
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "utc_range": manifest["dataset"]["range"],
            "audited_interruption_ids": [],
        },
    }
    try:
        validate_protocol_v3_input_lock(input_lock, protocol, config_snapshot.digest)
    except ProtocolV3Error as exc:
        raise ProtocolV3InputFreezeError("Protocol V3 input lock is invalid") from exc
    _publish_bundle(
        bundle,
        protocol,
        config_snapshot,
        (csv_snapshot, metadata_snapshot, anomaly_snapshot),
        manifest_content,
        _json_bytes(input_lock),
        dataset,
    )
    return manifest, input_lock
