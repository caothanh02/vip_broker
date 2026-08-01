"""Reproducible, development-only recommendation experiments.

This module intentionally consumes a frozen local manifest and never contacts an
exchange, loads an inference model, or creates executable trading objects.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from trading_bot.data.csv_store import (
    CsvDataError,
    csv_sha256,
    metadata_path,
    read_candles,
    verified_missing_open_times,
    verify_metadata_checksum,
    write_json_atomic,
)
from trading_bot.data.validation import CandleValidationError, validate_candles
from trading_bot.domain.models import Candle
from trading_bot.recommendations.engine import (
    RecommendationEngine,
    RecommendationHistoryProvenance,
    accuracy_report,
    backfill_recommendations,
    evaluate_outcomes,
)
from trading_bot.settings import BotSettings

_EXPERIMENT_SCHEMA_VERSION = "1.0"
_MANIFEST_SCHEMA_VERSION = "1.0"
_MANIFEST_DIRECTORY = Path("reports/research/manifests")
_EXPERIMENT_DIRECTORY = Path("reports/research/experiments")
_HORIZONS = ("1h", "4h", "24h")

_CANDIDATES: dict[str, dict[str, Any]] = {
    "baseline_ema_volume_atr_v1": {
        "hypothesis": (
            "baseline rule-only EMA/volume/ATR creates a measurable BUY_BIAS / AVOID / "
            "NEUTRAL recommendation baseline"
        ),
        "parameters": {
            "ema_fast": 20,
            "ema_slow": 50,
            "ema_trend": 200,
            "volume_window": 20,
            "volume_multiplier": 1.2,
            "atr_window": 14,
        },
        "cost_model": {
            "entry_fee_rate": "0.001",
            "exit_fee_rate": "0.001",
            "entry_slippage_rate": "0.0005",
            "exit_slippage_rate": "0.0005",
        },
    }
}


class RecommendationExperimentError(ValueError):
    """A requested development experiment is not reproducible or safe."""


@dataclass(frozen=True, slots=True)
class _VerifiedExperimentInput:
    manifest_path: Path
    dataset_path: Path
    candles: list[Candle]
    missing_open_times: set[datetime]
    manifest: dict[str, Any]
    manifest_sha256: str


def _utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_utc(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise RecommendationExperimentError(f"{label} must be an ISO UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RecommendationExperimentError(f"{label} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise RecommendationExperimentError(f"{label} must be UTC")
    return parsed.astimezone(UTC)


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RecommendationExperimentError(f"could not read {label}") from exc
    if not isinstance(value, dict):
        raise RecommendationExperimentError(f"{label} must be a JSON object")
    return value


def _require_regular_non_symlink(path: Path, label: str) -> None:
    try:
        if path.is_symlink():
            raise RecommendationExperimentError(f"{label} must not be a symlink")
        if not path.is_file():
            raise RecommendationExperimentError(f"{label} must be a regular file")
    except OSError as exc:
        raise RecommendationExperimentError(f"could not inspect {label}") from exc


def _unresolved_workspace_path(value: object, label: str) -> Path:
    if not isinstance(value, str):
        raise RecommendationExperimentError(f"{label} must be a relative workspace path")
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise RecommendationExperimentError(f"{label} must remain inside the workspace")
    workspace = Path.cwd().resolve()
    unresolved = workspace / candidate
    try:
        unresolved.relative_to(workspace)
    except (OSError, ValueError) as exc:
        raise RecommendationExperimentError(f"{label} must remain inside the workspace") from exc
    return unresolved


def _workspace_path(value: object, label: str) -> Path:
    unresolved = _unresolved_workspace_path(value, label)
    _require_regular_non_symlink(unresolved, label)
    try:
        resolved = unresolved.resolve()
        resolved.relative_to(Path.cwd().resolve())
    except (OSError, ValueError) as exc:
        raise RecommendationExperimentError(f"{label} must remain inside the workspace") from exc
    return resolved


def _experiment_output_path(path: Path) -> Path:
    if path.suffix != ".json" or ".." in path.parts:
        raise RecommendationExperimentError(
            "experiment output must be a .json path without traversal"
        )
    workspace = Path.cwd().resolve()
    experiment_directory = (workspace / _EXPERIMENT_DIRECTORY).resolve()
    unresolved = _unresolved_workspace_path(str(path), "experiment output")
    try:
        if unresolved.is_symlink():
            raise RecommendationExperimentError("experiment output must not be a symlink")
    except OSError as exc:
        raise RecommendationExperimentError("could not inspect experiment output") from exc
    try:
        experiment_directory.relative_to(workspace)
        resolved = unresolved.resolve()
        resolved.relative_to(experiment_directory)
    except (OSError, ValueError) as exc:
        raise RecommendationExperimentError(
            "experiment output must be inside reports/research/experiments"
        ) from exc
    return resolved


def _manifest_input_path(path: Path) -> Path:
    if path.suffix != ".json" or ".." in path.parts:
        raise RecommendationExperimentError("research manifest path is invalid")
    workspace = Path.cwd().resolve()
    manifest_directory = (workspace / _MANIFEST_DIRECTORY).resolve()
    unresolved = _unresolved_workspace_path(str(path), "research manifest path")
    _require_regular_non_symlink(unresolved, "research manifest")
    try:
        manifest_directory.relative_to(workspace)
        resolved = unresolved.resolve()
        resolved.relative_to(manifest_directory)
    except (OSError, ValueError) as exc:
        raise RecommendationExperimentError(
            "research manifest must be inside reports/research/manifests"
        ) from exc
    return resolved


def _require_exact_keys(value: object, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise RecommendationExperimentError(f"{label} schema is invalid")
    return value


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RecommendationExperimentError(f"{label} must be a SHA-256 digest")
    return value


def _validate_manifest(manifest: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    _require_exact_keys(
        manifest,
        {
            "schema_version",
            "created_at",
            "research_role",
            "strict_oos_evaluation_history",
            "strict_oos_start",
            "dataset",
            "market_interruptions",
            "safety_locks",
        },
        "research manifest",
    )
    if manifest["schema_version"] != _MANIFEST_SCHEMA_VERSION:
        raise RecommendationExperimentError("research manifest schema is unsupported")
    _parse_utc(manifest["created_at"], "research manifest created_at")
    if manifest["research_role"] != "development":
        raise RecommendationExperimentError("research manifest must have development role")
    if manifest["strict_oos_evaluation_history"] is not False:
        raise RecommendationExperimentError(
            "strict OOS manifests cannot run development experiments"
        )
    strict_oos_start = _parse_utc(
        manifest["strict_oos_start"], "research manifest strict_oos_start"
    )
    dataset = _require_exact_keys(
        manifest["dataset"],
        {
            "path",
            "symbol",
            "timeframe",
            "range",
            "candle_count",
            "csv_sha256",
            "metadata_sha256",
            "anomaly_sidecar_sha256",
            "generation_id",
            "validation_status",
            "checksum_verification_mode",
        },
        "research manifest dataset",
    )
    if dataset["symbol"] != "BTC/USDT" or dataset["timeframe"] != "1h":
        raise RecommendationExperimentError("research manifest market must be BTC/USDT 1h")
    if not isinstance(dataset["candle_count"], int) or dataset["candle_count"] <= 0:
        raise RecommendationExperimentError("research manifest candle count is invalid")
    if not isinstance(dataset["generation_id"], str) or not dataset["generation_id"]:
        raise RecommendationExperimentError("research manifest generation ID is invalid")
    if dataset["validation_status"] not in {"valid", "valid_with_market_interruptions"}:
        raise RecommendationExperimentError("research manifest validation status is invalid")
    if dataset["checksum_verification_mode"] != "official_online":
        raise RecommendationExperimentError(
            "research manifest checksum verification mode is invalid"
        )
    for key in ("csv_sha256", "metadata_sha256", "anomaly_sidecar_sha256"):
        _require_sha256(dataset[key], f"research manifest {key}")
    range_value = _require_exact_keys(dataset["range"], {"start", "end"}, "research manifest range")
    range_start = _parse_utc(range_value["start"], "research manifest range start")
    range_end = _parse_utc(range_value["end"], "research manifest range end")
    if range_end <= range_start or strict_oos_start < range_end:
        raise RecommendationExperimentError("research manifest development range is inconsistent")
    interruptions = manifest["market_interruptions"]
    if not isinstance(interruptions, list):
        raise RecommendationExperimentError("research manifest market interruptions are invalid")
    for interruption in interruptions:
        _require_exact_keys(
            interruption,
            {"event_id", "missing_open_times", "official_source_urls", "tradable"},
            "research manifest market interruption",
        )
        if (
            not isinstance(interruption["event_id"], str)
            or interruption["tradable"] is not False
            or not isinstance(interruption["missing_open_times"], list)
            or not isinstance(interruption["official_source_urls"], list)
        ):
            raise RecommendationExperimentError("research manifest market interruption is invalid")
        for missing in interruption["missing_open_times"]:
            _parse_utc(missing, "research manifest missing candle time")
        if not all(isinstance(url, str) for url in interruption["official_source_urls"]):
            raise RecommendationExperimentError(
                "research manifest interruption source URLs are invalid"
            )
    safety_locks = _require_exact_keys(
        manifest["safety_locks"],
        {"live_trading_enabled", "broker_used", "orders_submitted", "ml_used"},
        "research manifest safety locks",
    )
    if any(value is not False for value in safety_locks.values()):
        raise RecommendationExperimentError("research manifest safety locks are invalid")
    return _workspace_path(dataset["path"], "research manifest dataset path"), dataset


def _verified_experiment_input(manifest_path: Path) -> _VerifiedExperimentInput:
    resolved_manifest = _manifest_input_path(manifest_path)
    manifest = _read_object(resolved_manifest, "research manifest")
    dataset_path, dataset = _validate_manifest(manifest)
    _require_regular_non_symlink(dataset_path, "development dataset CSV")
    metadata_file = metadata_path(dataset_path)
    anomaly_file = dataset_path.with_suffix(".anomalies.json")
    _require_regular_non_symlink(metadata_file, "development dataset metadata sidecar")
    _require_regular_non_symlink(anomaly_file, "development dataset anomaly sidecar")
    if (
        csv_sha256(dataset_path) != dataset["csv_sha256"]
        or csv_sha256(metadata_file) != dataset["metadata_sha256"]
        or csv_sha256(anomaly_file) != dataset["anomaly_sidecar_sha256"]
    ):
        raise RecommendationExperimentError(
            "research manifest sidecar checksum does not match input"
        )
    metadata = _read_object(metadata_file, "metadata sidecar")
    report = _read_object(anomaly_file, "anomaly sidecar")
    if metadata.get("anomaly_report") != anomaly_file.name:
        raise RecommendationExperimentError("metadata anomaly sidecar identity does not match CSV")
    if (
        metadata.get("generation_id") != dataset["generation_id"]
        or report.get("generation_id") != dataset["generation_id"]
        or metadata.get("checksum_verification_mode") != dataset["checksum_verification_mode"]
        or not isinstance(report.get("policy"), dict)
        or report["policy"].get("checksum_verification_mode")
        != dataset["checksum_verification_mode"]
    ):
        raise RecommendationExperimentError("research manifest provenance does not match sidecars")
    raw_interruptions = report.get("market_interruptions")
    if not isinstance(raw_interruptions, list) or not all(
        isinstance(item, dict) for item in raw_interruptions
    ):
        raise RecommendationExperimentError(
            "development anomaly interruption provenance is invalid"
        )
    actual_interruptions = [
        {
            "event_id": item.get("event_id"),
            "missing_open_times": item.get("missing_open_times"),
            "official_source_urls": item.get("official_source_urls"),
            "tradable": item.get("tradable"),
        }
        for item in raw_interruptions
    ]
    if actual_interruptions != manifest["market_interruptions"]:
        raise RecommendationExperimentError(
            "research manifest interruption provenance does not match"
        )
    try:
        if not verify_metadata_checksum(dataset_path):
            raise RecommendationExperimentError("development metadata checksum is invalid")
        missing = verified_missing_open_times(dataset_path)
        candles = read_candles(dataset_path, allowed_missing_open_times=missing)
        validate_candles(candles, allowed_missing_open_times=missing)
    except (CsvDataError, CandleValidationError) as exc:
        raise RecommendationExperimentError("development dataset validation failed") from exc
    if (
        len(candles) != dataset["candle_count"]
        or candles[0].open_time
        != _parse_utc(dataset["range"]["start"], "research manifest range start")
        or candles[-1].close_time
        != _parse_utc(dataset["range"]["end"], "research manifest range end")
        or any(candle.symbol != "BTC/USDT" or candle.timeframe != "1h" for candle in candles)
    ):
        raise RecommendationExperimentError(
            "research manifest dataset identity does not match input"
        )
    return _VerifiedExperimentInput(
        resolved_manifest,
        dataset_path,
        candles,
        missing,
        manifest,
        csv_sha256(resolved_manifest),
    )


def _development_metrics(recommendations: list[Any], outcomes: list[Any]) -> dict[str, Any]:
    metrics = accuracy_report(
        recommendations, outcomes, RecommendationHistoryProvenance(strict_oos=False)
    )
    metrics["strict_oos"] = False
    metrics["strict_oos_validation"] = "development_dataset_not_strict_oos"
    metrics["research_claim_eligible"] = False
    metrics["research_claim_eligibility_reason"] = "development_dataset_not_strict_oos"
    horizons = metrics["horizons"]
    for horizon in _HORIZONS:
        horizon_metrics = horizons[horizon]
        horizon_metrics["research_claim_eligible"] = False
        horizon_metrics["research_claim_eligibility_reason"] = "development_dataset_not_strict_oos"
    return metrics


def _validate_candidate_settings(candidate_id: str, settings: BotSettings) -> None:
    candidate = _CANDIDATES[candidate_id]
    parameters = candidate["parameters"]
    for field, expected in parameters.items():
        if getattr(settings, field) != expected:
            raise RecommendationExperimentError(
                f"experiment candidate parameter {field} does not match its predeclared value"
            )
    cost_model = candidate["cost_model"]
    for field, expected in cost_model.items():
        if getattr(settings, field) != Decimal(expected):
            raise RecommendationExperimentError(
                f"experiment candidate cost rate {field} does not match its predeclared value"
            )


def run_development_experiment(
    manifest_path: Path,
    candidate_id: str,
    output_path: Path,
    settings: BotSettings,
    *,
    overwrite: bool,
    now: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Run the single predeclared rule-only baseline against frozen development data."""

    if candidate_id not in _CANDIDATES:
        raise RecommendationExperimentError("unknown or unregistered experiment candidate")
    if settings.bot_mode == "live" or settings.ml_filter_enabled:
        raise RecommendationExperimentError(
            "development experiments require rule-only non-live settings"
        )
    _validate_candidate_settings(candidate_id, settings)
    resolved_output = _experiment_output_path(output_path)
    snapshot = _verified_experiment_input(manifest_path)
    if resolved_output.exists() and not overwrite:
        raise RecommendationExperimentError("experiment output already exists; pass --overwrite")
    recommendations = backfill_recommendations(
        RecommendationEngine(settings), snapshot.candles, snapshot.missing_open_times
    )
    outcomes = evaluate_outcomes(
        recommendations, snapshot.candles, settings, snapshot.missing_open_times
    )
    metrics = _development_metrics(recommendations, outcomes)
    dataset = snapshot.manifest["dataset"]
    run_time = (now or (lambda: datetime.now(UTC)))().astimezone(UTC)
    result: dict[str, Any] = {
        "schema_version": _EXPERIMENT_SCHEMA_VERSION,
        "run_at": _utc(run_time),
        "candidate_id": candidate_id,
        "candidate": _CANDIDATES[candidate_id],
        "research_role": "development",
        "strict_oos_evaluation_history": False,
        "research_claim_eligible": False,
        "research_claim_eligibility_reason": "development_dataset_not_strict_oos",
        "source_manifest": {
            "path": manifest_path.resolve().relative_to(Path.cwd().resolve()).as_posix(),
            "sha256": snapshot.manifest_sha256,
        },
        "dataset": {
            "path": dataset["path"],
            "csv_sha256": dataset["csv_sha256"],
            "generation_id": dataset["generation_id"],
            "range": dataset["range"],
            "candle_count": dataset["candle_count"],
        },
        "recommendation_count": len(recommendations),
        "outcome_count": len(outcomes),
        "metrics": metrics,
        "cost_model": _CANDIDATES[candidate_id]["cost_model"],
        "disclaimer": (
            "Development-only research result. It is not strict OOS evidence, an accuracy claim, "
            "investment advice, or an instruction to trade."
        ),
        "safety_locks": {
            "live_trading_enabled": False,
            "broker_used": False,
            "orders_submitted": False,
            "ml_inference_used": False,
        },
    }
    write_json_atomic(resolved_output, result)
    return result
