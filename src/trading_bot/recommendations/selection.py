"""Immutable development-selection artifacts required before strict OOS access."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trading_bot.data.csv_store import csv_sha256, write_json_atomic
from trading_bot.recommendations import experiments

_SCHEMA_VERSION = "1.0"
_WALK_FORWARD_DIRECTORY = Path("reports/research/walk-forward")
_MANIFEST_DIRECTORY = Path("reports/research/manifests")
_SELECTION_DIRECTORY = Path("reports/research/selections")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")


class DevelopmentSelectionError(ValueError):
    """A development selection artifact is absent, malformed, or no longer valid."""


def source_revision() -> str:
    """Return the exact checked-out commit used to build an artifact, or fail closed."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            # Callers and tests may run from a temporary working directory.  The
            # source revision is a property of this installed source tree, not of
            # the report/output directory.
            cwd=Path(__file__).resolve().parents[3],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise DevelopmentSelectionError("could not determine source revision") from exc
    revision = result.stdout.strip()
    if not _REVISION.fullmatch(revision):
        raise DevelopmentSelectionError("source revision is invalid")
    return revision


def _path(path: Path, directory: Path, label: str, *, exists: bool) -> Path:
    if path.suffix != ".json" or path.is_absolute() or ".." in path.parts:
        raise DevelopmentSelectionError(f"{label} must be a relative .json path without traversal")
    try:
        unresolved = Path.cwd().resolve() / path
        if unresolved.is_symlink():
            raise DevelopmentSelectionError(f"{label} must not be a symlink")
        if exists and (not unresolved.is_file() or unresolved.is_symlink()):
            raise DevelopmentSelectionError(f"{label} must be a regular file")
        resolved = unresolved.resolve()
        resolved.relative_to((Path.cwd().resolve() / directory).resolve())
    except DevelopmentSelectionError:
        raise
    except (OSError, ValueError) as exc:
        raise DevelopmentSelectionError(f"{label} must be inside {directory.as_posix()}") from exc
    return resolved


def _object(path: Path, label: str) -> dict[str, Any]:
    try:
        return experiments._read_object(path, label)
    except experiments.RecommendationExperimentError as exc:
        raise DevelopmentSelectionError(str(exc)) from exc


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise DevelopmentSelectionError(f"{label} must be a SHA-256 digest")
    return value


def _candidate(candidate_id: object, contract: object) -> tuple[str, dict[str, Any]]:
    if not isinstance(candidate_id, str) or candidate_id not in experiments._CANDIDATES:
        raise DevelopmentSelectionError("selection candidate is unknown or unregistered")
    expected = experiments._CANDIDATES[candidate_id]
    if not isinstance(contract, dict) or contract != expected:
        raise DevelopmentSelectionError("selection candidate contract does not match registry")
    return candidate_id, expected


def create_development_selection(
    report_path: Path,
    output_path: Path,
    *,
    overwrite: bool,
    now: Callable[[], datetime] | None = None,
    revision: Callable[[], str] = source_revision,
) -> dict[str, Any]:
    """Seal one selected development policy; reject ``no_policy_selected`` results."""

    report_file = _path(report_path, _WALK_FORWARD_DIRECTORY, "development report", exists=True)
    output_file = _path(
        output_path, _SELECTION_DIRECTORY, "selection artifact output", exists=False
    )
    report = _object(report_file, "development report")
    decision = report.get("selection_decision")
    if (
        report.get("protocol_version") != "development_walk_forward_v1"
        or report.get("research_role") != "development"
        or report.get("strict_oos_evaluation_history") is not False
        or report.get("research_claim_eligible") is not False
        or not isinstance(decision, dict)
        or decision.get("decision") != "selected"
    ):
        raise DevelopmentSelectionError("development report did not select a policy")
    candidate_id, contract = _candidate(report.get("candidate_id"), report.get("candidate"))
    if decision.get("selected_candidate_id") != candidate_id:
        raise DevelopmentSelectionError("development report selected candidate is inconsistent")
    source_manifest = report.get("source_manifest")
    if not isinstance(source_manifest, dict):
        raise DevelopmentSelectionError("development report manifest provenance is invalid")
    source_manifest_path = source_manifest.get("path")
    if not isinstance(source_manifest_path, str):
        raise DevelopmentSelectionError("development report manifest path is invalid")
    manifest_file = _path(
        Path(source_manifest_path),
        _MANIFEST_DIRECTORY,
        "development manifest",
        exists=True,
    )
    if csv_sha256(manifest_file) != _sha(source_manifest.get("sha256"), "development manifest SHA"):
        raise DevelopmentSelectionError("development manifest checksum does not match report")
    report_revision = report.get("code_revision")
    current_revision = revision()
    if not isinstance(report_revision, str) or not _REVISION.fullmatch(report_revision):
        raise DevelopmentSelectionError("development report code revision is invalid")
    if report_revision != current_revision:
        raise DevelopmentSelectionError(
            "development report code revision does not match current source"
        )
    if output_file.exists() and not overwrite:
        raise DevelopmentSelectionError(
            "selection artifact already exists; pass --overwrite after validation"
        )
    created = (
        (now or (lambda: datetime.now(UTC)))().astimezone(UTC).isoformat().replace("+00:00", "Z")
    )
    artifact: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "created_at": created,
        "code_revision": current_revision,
        "protocol_version": report["protocol_version"],
        "candidate_id": candidate_id,
        "candidate": contract,
        "development_report": {"path": report_path.as_posix(), "sha256": csv_sha256(report_file)},
        "development_manifest": {
            "path": source_manifest["path"],
            "sha256": source_manifest["sha256"],
        },
        "selection_decision": "selected",
        "strict_oos_authorized": True,
    }
    write_json_atomic(output_file, artifact)
    return artifact


def validate_development_selection(
    artifact_path: Path,
    candidate_id: str,
    *,
    revision: Callable[[], str] = source_revision,
) -> tuple[dict[str, Any], str]:
    """Validate the selection artifact before any strict-OOS artifact is opened."""

    artifact_file = _path(artifact_path, _SELECTION_DIRECTORY, "selection artifact", exists=True)
    artifact = _object(artifact_file, "selection artifact")
    required = {
        "schema_version",
        "created_at",
        "code_revision",
        "protocol_version",
        "candidate_id",
        "candidate",
        "development_report",
        "development_manifest",
        "selection_decision",
        "strict_oos_authorized",
    }
    if set(artifact) != required or artifact.get("schema_version") != _SCHEMA_VERSION:
        raise DevelopmentSelectionError("selection artifact schema is invalid")
    selected_id, _ = _candidate(artifact.get("candidate_id"), artifact.get("candidate"))
    if (
        selected_id != candidate_id
        or artifact.get("selection_decision") != "selected"
        or artifact.get("strict_oos_authorized") is not True
    ):
        raise DevelopmentSelectionError(
            "selection artifact does not authorize this strict OOS candidate"
        )
    if artifact.get("protocol_version") != "development_walk_forward_v1":
        raise DevelopmentSelectionError("selection artifact protocol is invalid")
    artifact_revision = artifact.get("code_revision")
    if not isinstance(artifact_revision, str) or not _REVISION.fullmatch(artifact_revision):
        raise DevelopmentSelectionError("selection artifact code revision is invalid")
    if artifact_revision != revision():
        raise DevelopmentSelectionError(
            "selection artifact code revision does not match current source"
        )
    resolved_references: dict[str, tuple[dict[str, Any], Path]] = {}
    for key, directory, label in (
        ("development_report", _WALK_FORWARD_DIRECTORY, "development report"),
        ("development_manifest", _MANIFEST_DIRECTORY, "development manifest"),
    ):
        value = artifact.get(key)
        if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
            raise DevelopmentSelectionError(f"selection artifact {label} is invalid")
        path = value.get("path")
        if not isinstance(path, str):
            raise DevelopmentSelectionError(f"selection artifact {label} path is invalid")
        file = _path(Path(path), directory, label, exists=True)
        if csv_sha256(file) != _sha(value.get("sha256"), f"{label} SHA"):
            raise DevelopmentSelectionError(f"selection artifact {label} checksum does not match")
        resolved_references[key] = (value, file)

    report = _object(resolved_references["development_report"][1], "development report")
    decision = report.get("selection_decision")
    if (
        report.get("protocol_version") != artifact["protocol_version"]
        or report.get("code_revision") != artifact_revision
        or report.get("research_role") != "development"
        or report.get("strict_oos_evaluation_history") is not False
        or report.get("research_claim_eligible") is not False
        or report.get("candidate_id") != selected_id
        or report.get("candidate") != artifact["candidate"]
        or not isinstance(decision, dict)
        or decision.get("decision") != "selected"
        or decision.get("selected_candidate_id") != selected_id
        or report.get("source_manifest") != artifact["development_manifest"]
    ):
        raise DevelopmentSelectionError("selection artifact development report is inconsistent")
    return artifact, csv_sha256(artifact_file)
