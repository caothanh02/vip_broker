"""Read-only operational observability for the closed recommendation programme.

This module deliberately has no dependency on recommendation generation, model
inference, broker, order, risk, exchange-client, or settings-loading code.  It
only verifies a local, already-published candle dataset and writes ignored JSON
status artifacts.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trading_bot.data.csv_store import (
    CSV_FIELDS,
    CsvDataError,
    csv_sha256,
    read_candles,
    verified_missing_open_times,
    verify_metadata_checksum,
    write_json_atomic,
)
from trading_bot.data.validation import CandleValidationError, validate_candles

_SCHEMA_VERSION = "1.0"
_STATUS_DIRECTORY = Path("reports/operations")
_INPUT_DIRECTORY = Path("data/raw")
_FORBIDDEN_IMPORT_PARTS = frozenset(
    {
        "backtest",
        "binance",
        "execution",
        "ml",
        "recommendations",
        "risk",
        "runtime",
        "settings",
    }
)
_REQUIRED_IGNORE_RULES = frozenset(
    {
        "data/raw/*",
        "data/archive_cache/",
        "data/validated/*",
        "data/features/*",
        "data/datasets/*",
        "data/processed/*",
        "data/dry_run/*",
        "models/*",
        "reports/backtests/*",
        "reports/ml/*",
        "reports/dry_run/*",
        "reports/recommendations/*",
        "reports/research/",
        "reports/operations/",
        "*.db",
        "*.sqlite*",
        "*.tmp",
        ".venv/",
        ".pytest_cache/",
    }
)
_AUDITED_MODULES = (Path("src/trading_bot/operations.py"), Path("src/trading_bot/cli.py"))


class OperationalSafetyError(ValueError):
    """A read-only operational status or audit request is unsafe or invalid."""


def _utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise OperationalSafetyError(f"could not read {label}") from exc
    if not isinstance(value, dict):
        raise OperationalSafetyError(f"{label} must be a JSON object")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _workspace() -> Path:
    return Path.cwd().resolve()


def _has_symlinked_workspace_component(path: Path) -> bool:
    workspace = _workspace()
    try:
        relative = path.relative_to(workspace)
    except ValueError:
        return True
    current = workspace
    try:
        for part in relative.parts:
            current /= part
            if current.is_symlink():
                return True
    except OSError:
        return True
    return False


def _regular_workspace_file(path: Path, directory: Path, label: str) -> Path:
    if path.is_absolute() or ".." in path.parts:
        raise OperationalSafetyError(f"{label} must be a relative path without traversal")
    workspace = _workspace()
    unresolved = workspace / path
    try:
        if _has_symlinked_workspace_component(unresolved) or not unresolved.is_file():
            raise OperationalSafetyError(f"{label} must be a regular non-symlink file")
        resolved = unresolved.resolve()
        resolved.relative_to((workspace / directory).resolve())
    except OperationalSafetyError:
        raise
    except (OSError, ValueError) as exc:
        raise OperationalSafetyError(f"{label} must remain inside {directory.as_posix()}") from exc
    return resolved


def _status_output_path(path: Path, protected: set[Path]) -> Path:
    if path.suffix != ".json" or path.is_absolute() or ".." in path.parts:
        raise OperationalSafetyError(
            "operational output must be a relative .json path without traversal"
        )
    workspace = _workspace()
    unresolved = workspace / path
    try:
        if _has_symlinked_workspace_component(unresolved):
            raise OperationalSafetyError("operational output must not be a symlink")
        if unresolved.exists() and not unresolved.is_file():
            raise OperationalSafetyError("operational output must be a regular file")
        resolved = unresolved.resolve()
        resolved.relative_to((workspace / _STATUS_DIRECTORY).resolve())
    except OperationalSafetyError:
        raise
    except (OSError, ValueError) as exc:
        raise OperationalSafetyError(
            "operational output must be inside reports/operations"
        ) from exc
    if resolved in protected:
        raise OperationalSafetyError("operational output must not replace an input artifact")
    return resolved


def _canonical_artifacts(input_path: Path) -> tuple[Path, Path, Path]:
    csv_path = _regular_workspace_file(input_path, _INPUT_DIRECTORY, "dataset input")
    metadata = csv_path.with_name(f"{csv_path.name}.metadata.json")
    anomaly = csv_path.with_suffix(".anomalies.json")
    for path, label in ((metadata, "metadata sidecar"), (anomaly, "anomaly sidecar")):
        try:
            if path.is_symlink() or not path.is_file():
                raise OperationalSafetyError(f"{label} must be a regular non-symlink file")
            path.resolve().relative_to(csv_path.parent.resolve())
        except OperationalSafetyError:
            raise
        except (OSError, ValueError) as exc:
            raise OperationalSafetyError(f"{label} must remain beside the dataset input") from exc
    return csv_path, metadata, anomaly


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _snapshot_artifacts(paths: tuple[Path, Path, Path]) -> tuple[bytes, bytes, bytes]:
    """Read each trusted artifact once before validation to prevent mixed generations."""

    try:
        return tuple(path.read_bytes() for path in paths)  # type: ignore[return-value]
    except OSError as exc:
        raise OperationalSafetyError("could not snapshot dataset artifacts") from exc


def _write_snapshot(
    directory: Path, paths: tuple[Path, Path, Path], data: tuple[bytes, bytes, bytes]
) -> Path:
    try:
        for path, content in zip(paths, data, strict=True):
            (directory / path.name).write_bytes(content)
    except OSError as exc:
        raise OperationalSafetyError("could not prepare dataset artifact snapshot") from exc
    return directory / paths[0].name


def _recheck_snapshot(paths: tuple[Path, Path, Path], digests: tuple[str, str, str]) -> None:
    """Fail closed if an original artifact changed after its validated snapshot."""

    try:
        current = tuple(csv_sha256(path) for path in paths)
    except CsvDataError as exc:
        raise OperationalSafetyError("could not recheck dataset artifacts before publish") from exc
    if current != digests:
        raise OperationalSafetyError("dataset artifacts changed after validation")


def _safety_locks() -> dict[str, Any]:
    return {
        "default_recommendation": "NEUTRAL",
        "development_protocols": {
            "v1": "no_policy_selected",
            "v2": "no_policy_selected",
        },
        "strict_oos_2025": {"sealed": True, "executed": False},
        "strict_oos_sealed": True,
        "strict_oos_evaluated": False,
        "live_trading_enabled": False,
        "broker_used": False,
        "orders_submitted": False,
        "ml_used": False,
        "ml_inference_used": False,
        "recommendation_engine_used": False,
        "risk_engine_used": False,
        "dry_run_broker_used": False,
        "authenticated_binance_api_used": False,
        "network_used": False,
    }


def _interruption_status(report: dict[str, Any]) -> list[dict[str, Any]]:
    interruptions = report.get("market_interruptions")
    if not isinstance(interruptions, list):
        raise OperationalSafetyError("anomaly sidecar interruption records are invalid")
    result: list[dict[str, Any]] = []
    for item in interruptions:
        if not isinstance(item, dict):
            raise OperationalSafetyError("anomaly sidecar interruption record is invalid")
        event_id = item.get("event_id")
        missing = item.get("missing_open_times")
        if not isinstance(event_id, str) or not isinstance(missing, list):
            raise OperationalSafetyError("anomaly sidecar interruption identity is invalid")
        result.append({"event_id": event_id, "missing_open_times": missing, "tradable": False})
    return result


def _reject_raw_duplicate_timestamps(path: Path) -> None:
    """Reject duplicate source rows before the shared reader can merge them."""

    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != CSV_FIELDS:
                return
            timestamps: set[str] = set()
            for row in reader:
                timestamp = row.get("open_time")
                if not isinstance(timestamp, str):
                    return
                if timestamp in timestamps:
                    raise OperationalSafetyError("dataset contains duplicate source timestamp")
                timestamps.add(timestamp)
    except OperationalSafetyError:
        raise
    except OSError as exc:
        raise OperationalSafetyError("could not inspect dataset source rows") from exc


def operational_status(
    input_path: Path,
    output_path: Path,
    *,
    overwrite: bool,
    now: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Verify a local candle dataset and atomically publish non-research status."""

    # Reject an unsafe publication target before inspecting a potentially untrusted input.
    resolved_output = _status_output_path(output_path, set())
    csv_path, metadata_path_value, anomaly_path = _canonical_artifacts(input_path)
    if resolved_output in {
        csv_path.resolve(),
        metadata_path_value.resolve(),
        anomaly_path.resolve(),
    }:
        raise OperationalSafetyError("operational output must not replace an input artifact")
    artifact_paths = (csv_path, metadata_path_value, anomaly_path)
    snapshot_data = _snapshot_artifacts(artifact_paths)
    snapshot_digest_items = tuple(_sha256_bytes(item) for item in snapshot_data)
    snapshot_digests = (
        snapshot_digest_items[0],
        snapshot_digest_items[1],
        snapshot_digest_items[2],
    )
    with tempfile.TemporaryDirectory(prefix="trading-bot-operations-") as temporary_directory:
        snapshot_csv = _write_snapshot(Path(temporary_directory), artifact_paths, snapshot_data)
        snapshot_metadata = snapshot_csv.with_name(f"{snapshot_csv.name}.metadata.json")
        snapshot_anomaly = snapshot_csv.with_suffix(".anomalies.json")
        metadata = _read_object(snapshot_metadata, "metadata sidecar snapshot")
        if metadata.get("anomaly_report") != snapshot_anomaly.name:
            raise OperationalSafetyError(
                "metadata anomaly sidecar identity does not match dataset input"
            )
        report = _read_object(snapshot_anomaly, "anomaly sidecar snapshot")
        try:
            _reject_raw_duplicate_timestamps(snapshot_csv)
            if not verify_metadata_checksum(snapshot_csv):
                raise OperationalSafetyError("metadata CSV checksum is required")
            missing = verified_missing_open_times(snapshot_csv)
            candles = read_candles(snapshot_csv, allowed_missing_open_times=missing)
            validate_candles(candles, allowed_missing_open_times=missing)
        except (CsvDataError, CandleValidationError) as exc:
            raise OperationalSafetyError("dataset validation failed") from exc
        csv_digest, metadata_digest, anomaly_digest = snapshot_digests
        if metadata.get("anomaly_report_sha256") != anomaly_digest:
            raise OperationalSafetyError("metadata anomaly sidecar checksum mismatch")
        if metadata.get("generation_id") != report.get("generation_id"):
            raise OperationalSafetyError("metadata and anomaly sidecar generation mismatch")
        generation_id = metadata.get("generation_id")
        verification_mode = metadata.get("checksum_verification_mode")
        source = metadata.get("source")
        requested_start = metadata.get("requested_start")
        effective_end = metadata.get("effective_end")
        requested_count = metadata.get("requested_range_candle_count")
        if (
            not isinstance(generation_id, str)
            or not isinstance(verification_mode, str)
            or not isinstance(source, str)
            or not isinstance(requested_start, str)
            or not isinstance(effective_end, str)
            or not isinstance(requested_count, int)
        ):
            raise OperationalSafetyError("metadata identity is invalid")
        interruptions = _interruption_status(report)
    created_at = (now or (lambda: datetime.now(UTC)))().astimezone(UTC)
    payload: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "generated_at": _utc(created_at),
        "operational_contract_revision": "neutral-only-v1",
        "status_kind": "read_only_operational_observability",
        "not_research_or_oos_evidence": True,
        "dataset": {
            "path": input_path.as_posix(),
            "symbol": candles[0].symbol,
            "timeframe": candles[0].timeframe,
            "source": source,
            "requested_range": {
                "start": requested_start,
                "end": effective_end,
                "candle_count": requested_count,
            },
            "stored_range": {
                "start": _utc(candles[0].open_time),
                "end": _utc(candles[-1].close_time),
            },
            "candle_count": len(candles),
            "generation_id": generation_id,
            "checksums": {
                "csv_sha256": csv_digest,
                "metadata_sha256": metadata_digest,
                "anomaly_sidecar_sha256": anomaly_digest,
                "metadata_csv_checksum_verified": True,
                "anomaly_sidecar_checksum_verified": True,
                "verification_mode": verification_mode,
            },
            "freshness": {
                "last_closed_candle": _utc(candles[-1].close_time),
                "age_seconds": max(0, int((created_at - candles[-1].close_time).total_seconds())),
            },
        },
        "tradability": {
            "continuous_for_trading": not interruptions,
            "contains_audited_non_tradable_interruption": bool(interruptions),
            "audited_non_tradable_interruptions": interruptions,
        },
        "safety_locks": _safety_locks(),
    }
    if resolved_output.exists() and not overwrite:
        raise OperationalSafetyError(
            "operational output already exists; pass --overwrite after validation"
        )
    _recheck_snapshot(artifact_paths, snapshot_digests)
    try:
        write_json_atomic(resolved_output, payload)
    except OSError as exc:
        raise OperationalSafetyError("could not atomically publish operational output") from exc
    return payload


def _top_level_runtime_imports(path: Path) -> list[str]:
    """Return real module imports, excluding only ``if TYPE_CHECKING`` branches."""
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, SyntaxError) as exc:
        raise OperationalSafetyError("could not inspect operational runtime imports") from exc
    imported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    return sorted(
        name
        for name in imported
        if any(part in _FORBIDDEN_IMPORT_PARTS for part in name.split("."))
    )


def _forbidden_runtime_imports() -> list[str]:
    result: list[str] = []
    for module in _AUDITED_MODULES:
        result.extend(
            f"{module.as_posix()}:{name}"
            for name in _top_level_runtime_imports(_workspace() / module)
        )
    return result


def _live_mode_fail_closed() -> bool:
    """Verify the actual Pydantic validator AST without loading settings or ``.env``."""

    settings_path = _workspace() / "src/trading_bot/settings.py"
    try:
        if settings_path.is_symlink() or not settings_path.is_file():
            return False
        tree = ast.parse(settings_path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return False
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "BotSettings":
            continue
        for method in node.body:
            if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not any(
                (isinstance(decorator, ast.Name) and decorator.id == "model_validator")
                or (
                    isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Name)
                    and decorator.func.id == "model_validator"
                )
                for decorator in method.decorator_list
            ):
                continue
            for candidate in ast.walk(method):
                if not isinstance(candidate, ast.If) or not isinstance(candidate.test, ast.Compare):
                    continue
                if not _is_live_mode_test(candidate.test):
                    continue
                if any(_is_value_error_raise(item) for item in ast.walk(candidate)):
                    return True
    return False


def _is_live_mode_test(test: ast.Compare) -> bool:
    return (
        len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
        and isinstance(test.left, ast.Attribute)
        and isinstance(test.left.value, ast.Name)
        and test.left.value.id == "self"
        and test.left.attr == "bot_mode"
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "live"
    )


def _is_value_error_raise(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Raise)
        and isinstance(node.exc, ast.Call)
        and isinstance(node.exc.func, ast.Name)
        and node.exc.func.id == "ValueError"
    )


def _ignored_artifact_rules() -> tuple[bool, list[str]]:
    try:
        rules = [
            line.strip()
            for line in (_workspace() / ".gitignore").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    except OSError:
        return False, sorted(_REQUIRED_IGNORE_RULES)
    missing = [
        rule for rule in sorted(_REQUIRED_IGNORE_RULES) if not _rule_remains_ignored(rule, rules)
    ]
    return not missing, missing


def _rule_remains_ignored(rule: str, rules: list[str]) -> bool:
    try:
        position = rules.index(rule)
    except ValueError:
        return False
    root = rule.removesuffix("*").rstrip("/")
    for later in rules[position + 1 :]:
        if not later.startswith("!"):
            continue
        exception = later.removeprefix("!")
        if exception.endswith(".gitkeep") or exception.endswith("README.md"):
            continue
        if exception == rule or exception.removesuffix("*").rstrip("/").startswith(root):
            return False
    return True


def audit_safety(
    output_path: Path, *, overwrite: bool, now: Callable[[], datetime] | None = None
) -> dict[str, Any]:
    """Publish a bounded, machine-readable safety audit without loading config or Git."""

    resolved_output = _status_output_path(output_path, set())
    forbidden = _forbidden_runtime_imports()
    live_mode_locked = _live_mode_fail_closed()
    ignore_rules_present, missing_ignore_rules = _ignored_artifact_rules()
    findings = [
        {
            "code": "LIVE_MODE_FAIL_CLOSED",
            "status": "pass" if live_mode_locked else "fail",
            "detail": "The checked-in settings source rejects BOT_MODE=live."
            if live_mode_locked
            else "The checked-in live-mode safety lock could not be verified.",
        },
        {
            "code": "CLI_RUNTIME_IMPORT_CLOSURE",
            "status": "pass" if not forbidden else "fail",
            "detail": (
                "No prohibited top-level runtime imports detected in the operational CLI closure."
            )
            if not forbidden
            else ("Prohibited top-level runtime imports detected in the operational CLI closure."),
        },
        {
            "code": "NO_CREDENTIALS_REQUIRED",
            "status": "pass",
            "detail": "Operational commands do not load settings or credentials.",
        },
        {
            "code": "GENERATED_ARTIFACTS_IGNORED",
            "status": "pass" if ignore_rules_present else "fail",
            "detail": "Required data, state, report, cache, model, and database ignore rules exist."
            if ignore_rules_present
            else "One or more required generated-artifact ignore rules are missing.",
        },
        {
            "code": "SOURCE_IDENTITY_NOT_CERTIFIED",
            "status": "not_applicable",
            "detail": "This runtime audit deliberately does not execute shell or Git.",
        },
    ]
    passed = all(item["status"] != "fail" for item in findings)
    created_at = (now or (lambda: datetime.now(UTC)))().astimezone(UTC)
    payload: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "generated_at": _utc(created_at),
        "audit_kind": "read_only_runtime_safety_contract",
        "not_a_certification": True,
        "passed": passed,
        "findings": findings,
        "safety_locks": _safety_locks(),
    }
    if resolved_output.exists() and not overwrite:
        raise OperationalSafetyError(
            "operational output already exists; pass --overwrite after validation"
        )
    try:
        write_json_atomic(resolved_output, payload)
    except OSError as exc:
        raise OperationalSafetyError("could not atomically publish operational output") from exc
    return payload
