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

_SCHEMA_VERSION = "1.1"
_SOURCE_IDENTITY_SCHEMA_VERSION = "1.0"
_WALK_FORWARD_DIRECTORY = Path("reports/research/walk-forward")
_MANIFEST_DIRECTORY = Path("reports/research/manifests")
_SELECTION_DIRECTORY = Path("reports/research/selections")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_SOURCE_INPUTS = ("src/trading_bot", "pyproject.toml", "uv.lock")


class DevelopmentSelectionError(ValueError):
    """A development selection artifact is absent, malformed, or no longer valid."""


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _git_output(*arguments: str) -> str:
    """Run a bounded Git query against this source checkout."""

    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=_repository_root(),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise DevelopmentSelectionError("could not determine source identity") from exc
    return result.stdout.strip()


def source_revision() -> str:
    """Return the exact checked-out commit used to build a report, or fail closed."""

    revision = _git_output("rev-parse", "HEAD")
    if not _REVISION.fullmatch(revision):
        raise DevelopmentSelectionError("source revision is invalid")
    return revision


def _source_worktree_entries() -> list[str]:
    output = _git_output("status", "--porcelain=v1", "--untracked-files=all", "--", *_SOURCE_INPUTS)
    return [line for line in output.splitlines() if line]


def _source_identity_value(value: object) -> dict[str, Any]:
    expected = {"schema_version", "revision", "tracked_objects"}
    if not isinstance(value, dict) or set(value) != expected:
        raise DevelopmentSelectionError("source identity schema is invalid")
    revision = value.get("revision")
    objects = value.get("tracked_objects")
    if (
        value.get("schema_version") != _SOURCE_IDENTITY_SCHEMA_VERSION
        or not isinstance(revision, str)
        or not _REVISION.fullmatch(revision)
        or not isinstance(objects, dict)
        or set(objects) != set(_SOURCE_INPUTS)
        or not all(isinstance(item, str) and _REVISION.fullmatch(item) for item in objects.values())
    ):
        raise DevelopmentSelectionError("source identity is invalid")
    return value


def source_identity() -> dict[str, Any]:
    """Return a clean, executable source identity without inspecting ignored artifacts."""

    entries = _source_worktree_entries()
    if entries:
        raise DevelopmentSelectionError("tracked executable source tree is not clean")
    identity = {
        "schema_version": _SOURCE_IDENTITY_SCHEMA_VERSION,
        "revision": source_revision(),
        "tracked_objects": {
            item: _git_output("rev-parse", f"HEAD:{item}") for item in _SOURCE_INPUTS
        },
    }
    return _source_identity_value(identity)


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


def _selection_evidence(report: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    """Recompute v1 selection from the report's fold evidence, never its declaration."""

    # Avoid a module cycle: walk_forward writes a revision through this module,
    # while sealing consumes its deterministic gate implementation.
    from trading_bot.recommendations import walk_forward

    folds = report.get("folds")
    pooled = report.get("pooled_metrics")
    declared_gate = report.get("selection_gate")
    declared_decision = report.get("selection_decision")
    if (
        not isinstance(folds, list)
        or len(folds) != len(walk_forward.FOLDS)
        or not isinstance(pooled, dict)
        or not isinstance(declared_gate, dict)
        or not isinstance(declared_decision, dict)
    ):
        raise DevelopmentSelectionError("development report selection evidence is invalid")
    try:
        fold_metrics = [
            fold["metrics"]
            for fold, expected_fold in zip(folds, walk_forward.FOLDS, strict=True)
            if isinstance(fold, dict) and fold.get("fold_id") == expected_fold.identifier
        ]
        if len(fold_metrics) != len(walk_forward.FOLDS) or not all(
            isinstance(metrics, dict) for metrics in fold_metrics
        ):
            raise KeyError("fold evidence")
        computed_gate = walk_forward.development_selection_gate(fold_metrics, pooled)
        computed_decision = walk_forward.select_candidate(
            {
                candidate_id: {
                    "fold_metrics": folds,
                    "pooled_metrics": pooled,
                    "selection_gate": computed_gate,
                }
            }
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise DevelopmentSelectionError("development report selection evidence is invalid") from exc
    if (
        computed_gate.get("passed") is not True
        or declared_gate != computed_gate
        or declared_decision != computed_decision
        or computed_decision.get("decision") != "selected"
        or computed_decision.get("selected_candidate_id") != candidate_id
    ):
        raise DevelopmentSelectionError("development report did not select a policy")
    return computed_gate


def _development_report_contract(
    report: dict[str, Any], current_identity: dict[str, Any]
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Validate every provenance-bearing part of a v1 walk-forward report."""

    expected = {
        "schema_version",
        "protocol_version",
        "code_revision",
        "run_at",
        "candidate_id",
        "candidate",
        "research_role",
        "strict_oos_evaluation_history",
        "research_claim_eligible",
        "research_claim_eligibility_reason",
        "source_manifest",
        "dataset",
        "folds",
        "pooled_metrics",
        "selection_gate",
        "selection_decision",
        "disclaimer",
        "safety_locks",
    }
    if set(report) != expected:
        raise DevelopmentSelectionError("development report schema is invalid")
    try:
        experiments._parse_utc(report["run_at"], "development report run_at")
    except experiments.RecommendationExperimentError as exc:
        raise DevelopmentSelectionError("development report run_at is invalid") from exc
    if (
        report.get("schema_version") != "1.0"
        or report.get("protocol_version") != "development_walk_forward_v1"
        or report.get("code_revision") != current_identity["revision"]
        or report.get("research_role") != "development"
        or report.get("strict_oos_evaluation_history") is not False
        or report.get("research_claim_eligible") is not False
        or report.get("research_claim_eligibility_reason") != "development_dataset_not_strict_oos"
    ):
        raise DevelopmentSelectionError("development report provenance is invalid")
    candidate_id, contract = _candidate(report.get("candidate_id"), report.get("candidate"))
    source_manifest = report.get("source_manifest")
    if not isinstance(source_manifest, dict) or set(source_manifest) != {"path", "sha256"}:
        raise DevelopmentSelectionError("development report manifest provenance is invalid")
    source_manifest_path = source_manifest.get("path")
    if not isinstance(source_manifest_path, str):
        raise DevelopmentSelectionError("development report manifest path is invalid")
    dataset = report.get("dataset")
    if not isinstance(dataset, dict) or set(dataset) != {
        "path",
        "csv_sha256",
        "generation_id",
        "range",
        "candle_count",
    }:
        raise DevelopmentSelectionError("development report dataset provenance is invalid")
    locks = report.get("safety_locks")
    if (
        not isinstance(locks, dict)
        or set(locks)
        != {
            "live_trading_enabled",
            "broker_used",
            "orders_submitted",
            "ml_inference_used",
        }
        or any(value is not False for value in locks.values())
    ):
        raise DevelopmentSelectionError("development report safety locks are invalid")
    _selection_evidence(report, candidate_id)
    return candidate_id, contract, source_manifest


def create_development_selection(
    report_path: Path,
    output_path: Path,
    *,
    overwrite: bool,
    now: Callable[[], datetime] | None = None,
    identity: Callable[[], dict[str, Any]] = source_identity,
) -> dict[str, Any]:
    """Seal one selected development policy; reject ``no_policy_selected`` results."""

    current_identity = _source_identity_value(identity())
    report_file = _path(report_path, _WALK_FORWARD_DIRECTORY, "development report", exists=True)
    output_file = _path(
        output_path, _SELECTION_DIRECTORY, "selection artifact output", exists=False
    )
    report = _object(report_file, "development report")
    candidate_id, contract, source_manifest = _development_report_contract(report, current_identity)
    source_manifest_path = source_manifest["path"]
    manifest_file = _path(
        Path(source_manifest_path),
        _MANIFEST_DIRECTORY,
        "development manifest",
        exists=True,
    )
    if csv_sha256(manifest_file) != _sha(source_manifest.get("sha256"), "development manifest SHA"):
        raise DevelopmentSelectionError("development manifest checksum does not match report")
    current_revision = current_identity["revision"]
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
        "source_identity": current_identity,
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
    identity: Callable[[], dict[str, Any]] = source_identity,
) -> tuple[dict[str, Any], str]:
    """Validate the selection artifact before any strict-OOS artifact is opened."""

    current_identity = _source_identity_value(identity())
    artifact_file = _path(artifact_path, _SELECTION_DIRECTORY, "selection artifact", exists=True)
    artifact = _object(artifact_file, "selection artifact")
    required = {
        "schema_version",
        "created_at",
        "code_revision",
        "source_identity",
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
    artifact_identity = _source_identity_value(artifact.get("source_identity"))
    if artifact_revision != current_identity["revision"] or artifact_identity != current_identity:
        raise DevelopmentSelectionError(
            "selection artifact source identity does not match current source"
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
    report_candidate_id, report_candidate, report_manifest = _development_report_contract(
        report, current_identity
    )
    if (
        report.get("protocol_version") != artifact["protocol_version"]
        or report_candidate_id != selected_id
        or report_candidate != artifact["candidate"]
        or report_manifest != artifact["development_manifest"]
    ):
        raise DevelopmentSelectionError("selection artifact development report is inconsistent")
    return artifact, csv_sha256(artifact_file)
