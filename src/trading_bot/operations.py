"""Read-only operational observability for the closed recommendation programme.

This module deliberately has no dependency on recommendation generation, model
inference, broker, order, risk, exchange-client, or settings-loading code.  It
only verifies a local, already-published candle dataset and writes ignored JSON
status artifacts.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trading_bot.data.csv_store import (
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
        "data/dry_run/*",
        "models/*",
        "reports/recommendations/*",
        "reports/research/",
        "reports/operations/",
        "*.db",
        "*.sqlite*",
    }
)


class OperationalSafetyError(ValueError):
    """A read-only operational status or audit request is unsafe or invalid."""


def _utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OperationalSafetyError(f"could not read {label}") from exc
    if not isinstance(value, dict):
        raise OperationalSafetyError(f"{label} must be a JSON object")
    return value


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


def _safety_locks() -> dict[str, Any]:
    return {
        "default_recommendation": "NEUTRAL",
        "development_protocols": {
            "v1": "no_policy_selected",
            "v2": "no_policy_selected",
        },
        "strict_oos_2025": {"sealed": True, "executed": False},
        "live_trading_enabled": False,
        "broker_used": False,
        "orders_submitted": False,
        "ml_used": False,
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
    metadata = _read_object(metadata_path_value, "metadata sidecar")
    if metadata.get("anomaly_report") != anomaly_path.name:
        raise OperationalSafetyError(
            "metadata anomaly sidecar identity does not match dataset input"
        )
    report = _read_object(anomaly_path, "anomaly sidecar")
    try:
        if not verify_metadata_checksum(csv_path):
            raise OperationalSafetyError("metadata CSV checksum is required")
        missing = verified_missing_open_times(csv_path)
        candles = read_candles(csv_path, allowed_missing_open_times=missing)
        validate_candles(candles, allowed_missing_open_times=missing)
        csv_digest = csv_sha256(csv_path)
        anomaly_digest = csv_sha256(anomaly_path)
        metadata_digest = csv_sha256(metadata_path_value)
    except (CsvDataError, CandleValidationError) as exc:
        raise OperationalSafetyError("dataset validation failed") from exc
    if metadata.get("anomaly_report_sha256") != anomaly_digest:
        raise OperationalSafetyError("metadata anomaly sidecar checksum mismatch")
    if metadata.get("generation_id") != report.get("generation_id"):
        raise OperationalSafetyError("metadata and anomaly sidecar generation mismatch")
    generation_id = metadata.get("generation_id")
    verification_mode = metadata.get("checksum_verification_mode")
    if not isinstance(generation_id, str) or not isinstance(verification_mode, str):
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
            "range": {"start": _utc(candles[0].open_time), "end": _utc(candles[-1].close_time)},
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
    try:
        write_json_atomic(resolved_output, payload)
    except OSError as exc:
        raise OperationalSafetyError("could not atomically publish operational output") from exc
    return payload


def _forbidden_imports() -> list[str]:
    """Inspect this bounded module only; this is a safety check, not certification."""

    try:
        source = Path(__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, SyntaxError) as exc:
        raise OperationalSafetyError("could not inspect operational module imports") from exc
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    return sorted(
        name
        for name in imported
        if any(part in _FORBIDDEN_IMPORT_PARTS for part in name.split("."))
    )


def _live_mode_fail_closed() -> bool:
    """Boundedly inspect the checked-in settings lock without loading any settings."""

    settings_path = _workspace() / "src/trading_bot/settings.py"
    try:
        if settings_path.is_symlink() or not settings_path.is_file():
            return False
        source = settings_path.read_text(encoding="utf-8")
    except OSError:
        return False
    return (
        'if self.bot_mode == "live":' in source
        and 'raise ValueError("live mode is deliberately disabled in this build")' in source
    )


def _ignored_artifact_rules() -> tuple[bool, list[str]]:
    try:
        rules = {
            line.strip()
            for line in (_workspace() / ".gitignore").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
    except OSError:
        return False, sorted(_REQUIRED_IGNORE_RULES)
    missing = sorted(_REQUIRED_IGNORE_RULES - rules)
    return not missing, missing


def audit_safety(
    output_path: Path, *, overwrite: bool, now: Callable[[], datetime] | None = None
) -> dict[str, Any]:
    """Publish a bounded, machine-readable safety audit without loading config or Git."""

    resolved_output = _status_output_path(output_path, set())
    forbidden = _forbidden_imports()
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
            "code": "OPERATIONAL_PATH_PROHIBITED_DEPENDENCIES",
            "status": "pass" if not forbidden else "fail",
            "detail": "No prohibited operational-module imports detected."
            if not forbidden
            else "Prohibited operational-module imports detected.",
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
