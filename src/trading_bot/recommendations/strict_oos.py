"""Frozen strict-OOS evaluation for the pre-registered rule-only baseline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
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
from trading_bot.domain.models import Candle, Recommendation, RecommendationOutcome
from trading_bot.recommendations import experiments
from trading_bot.recommendations.engine import (
    RecommendationEngine,
    RecommendationHistoryProvenance,
    accuracy_report,
    backfill_recommendations,
    evaluate_outcomes,
)
from trading_bot.settings import BotSettings

_MANIFEST_SCHEMA_VERSION = "1.0"
_REPORT_SCHEMA_VERSION = "1.0"
_MANIFEST_DIRECTORY = Path("reports/research/manifests")
_REPORT_DIRECTORY = Path("reports/research/strict-oos")
_OOS_START = datetime(2025, 1, 1, tzinfo=UTC)
_OOS_END = datetime(2026, 1, 1, tzinfo=UTC)


class StrictOosError(ValueError):
    """A strict-OOS dataset, manifest, or evaluation is not safe to use."""


@dataclass(frozen=True, slots=True)
class _VerifiedOosInput:
    input_path: Path
    candles: list[Candle]
    missing_open_times: set[datetime]
    dataset: dict[str, Any]
    interruptions: list[dict[str, Any]]


def _utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_utc(value: object, label: str) -> datetime:
    try:
        return experiments._parse_utc(value, label)
    except experiments.RecommendationExperimentError as exc:
        raise StrictOosError(str(exc)) from exc


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        return experiments._read_object(path, label)
    except experiments.RecommendationExperimentError as exc:
        raise StrictOosError(str(exc)) from exc


def _regular(path: Path, label: str) -> None:
    try:
        experiments._require_regular_non_symlink(path, label)
    except experiments.RecommendationExperimentError as exc:
        raise StrictOosError(str(exc)) from exc


def _inside_workspace(value: object, label: str) -> Path:
    try:
        return experiments._workspace_path(value, label)
    except experiments.RecommendationExperimentError as exc:
        raise StrictOosError(str(exc)) from exc


def _manifest_path(path: Path) -> Path:
    if path.suffix != ".json" or ".." in path.parts:
        raise StrictOosError("strict OOS manifest path is invalid")
    try:
        unresolved = experiments._unresolved_workspace_path(str(path), "strict OOS manifest path")
        _regular(unresolved, "strict OOS manifest")
        resolved = unresolved.resolve()
        resolved.relative_to((Path.cwd().resolve() / _MANIFEST_DIRECTORY).resolve())
    except StrictOosError:
        raise
    except (OSError, ValueError) as exc:
        raise StrictOosError(
            "strict OOS manifest must be inside reports/research/manifests"
        ) from exc
    return resolved


def _output_path(path: Path, directory: Path, label: str) -> Path:
    if path.suffix != ".json" or ".." in path.parts:
        raise StrictOosError(f"{label} must be a .json path without traversal")
    try:
        unresolved = experiments._unresolved_workspace_path(str(path), label)
        if unresolved.is_symlink():
            raise StrictOosError(f"{label} must not be a symlink")
        resolved = unresolved.resolve()
        resolved.relative_to((Path.cwd().resolve() / directory).resolve())
    except StrictOosError:
        raise
    except (OSError, ValueError) as exc:
        raise StrictOosError(f"{label} must be inside {directory.as_posix()}") from exc
    return resolved


def _normalized_interruptions(
    report: dict[str, Any], missing: set[datetime]
) -> list[dict[str, Any]]:
    raw = report.get("market_interruptions")
    if not isinstance(raw, list) or not all(isinstance(item, dict) for item in raw):
        raise StrictOosError("strict OOS anomaly interruption records are invalid")
    result: list[dict[str, Any]] = []
    reported_missing: set[datetime] = set()
    for item in raw:
        event_id = item.get("event_id")
        times = item.get("missing_open_times")
        sources = item.get("official_source_urls")
        if (
            not isinstance(event_id, str)
            or not isinstance(times, list)
            or not isinstance(sources, list)
            or not all(isinstance(source, str) for source in sources)
            or item.get("tradable") is not False
        ):
            raise StrictOosError("strict OOS market interruption is invalid")
        parsed_times = [_parse_utc(value, "strict OOS missing candle time") for value in times]
        reported_missing.update(parsed_times)
        result.append(
            {
                "event_id": event_id,
                "missing_open_times": [_utc(value) for value in parsed_times],
                "official_source_urls": sources,
                "tradable": False,
            }
        )
    if reported_missing != missing:
        raise StrictOosError("strict OOS audited gaps do not match anomaly sidecar")
    return result


def _verify_oos_dataset(input_path: Path) -> _VerifiedOosInput:
    _regular(input_path, "strict OOS dataset CSV")
    sidecar = metadata_path(input_path)
    anomaly = input_path.with_suffix(".anomalies.json")
    _regular(sidecar, "strict OOS dataset metadata sidecar")
    _regular(anomaly, "strict OOS dataset anomaly sidecar")
    metadata = _read_object(sidecar, "strict OOS metadata sidecar")
    if metadata.get("anomaly_report") != anomaly.name:
        raise StrictOosError("strict OOS anomaly sidecar identity does not match CSV")
    report = _read_object(anomaly, "strict OOS anomaly sidecar")
    try:
        if not verify_metadata_checksum(input_path):
            raise StrictOosError("strict OOS metadata checksum is invalid")
        missing = verified_missing_open_times(input_path)
        candles = read_candles(input_path, allowed_missing_open_times=missing)
        validate_candles(candles, allowed_missing_open_times=missing)
    except (CsvDataError, CandleValidationError) as exc:
        raise StrictOosError("strict OOS dataset validation failed") from exc
    policy = report.get("policy")
    if (
        metadata.get("checksum_verification_mode") != "official_online"
        or not isinstance(policy, dict)
        or policy.get("checksum_verification_mode") != "official_online"
        or metadata.get("generation_id") != report.get("generation_id")
        or not isinstance(metadata.get("generation_id"), str)
    ):
        raise StrictOosError("strict OOS dataset provenance is invalid")
    if (
        not candles
        or candles[0].open_time != _OOS_START
        or candles[-1].close_time != _OOS_END
        or any(candle.symbol != "BTC/USDT" or candle.timeframe != "1h" for candle in candles)
    ):
        raise StrictOosError("strict OOS dataset identity or range is invalid")
    interruptions = _normalized_interruptions(report, missing)
    dataset = {
        "path": input_path.resolve().relative_to(Path.cwd().resolve()).as_posix(),
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "range": {"start": _utc(_OOS_START), "end": _utc(_OOS_END)},
        "candle_count": len(candles),
        "csv_sha256": csv_sha256(input_path),
        "metadata_sha256": csv_sha256(sidecar),
        "anomaly_sidecar_sha256": csv_sha256(anomaly),
        "generation_id": metadata["generation_id"],
        "validation_status": "valid_with_market_interruptions" if interruptions else "valid",
        "checksum_verification_mode": "official_online",
    }
    return _VerifiedOosInput(input_path.resolve(), candles, missing, dataset, interruptions)


def freeze_strict_oos_dataset(
    input_path: Path,
    output_path: Path,
    *,
    overwrite: bool,
    now: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Freeze the fully separated 2025 strict-OOS dataset after sidecar verification."""

    resolved_input = _inside_workspace(str(input_path), "strict OOS dataset path")
    resolved_output = _output_path(output_path, _MANIFEST_DIRECTORY, "strict OOS manifest output")
    verified = _verify_oos_dataset(resolved_input)
    if resolved_output.exists() and not overwrite:
        raise StrictOosError(
            "strict OOS manifest already exists; pass --overwrite after validation"
        )
    manifest: dict[str, Any] = {
        "schema_version": _MANIFEST_SCHEMA_VERSION,
        "created_at": _utc((now or (lambda: datetime.now(UTC)))().astimezone(UTC)),
        "research_role": "strict_oos",
        "strict_oos_evaluation_history": True,
        "strict_oos_start": _utc(_OOS_START),
        "dataset": verified.dataset,
        "market_interruptions": verified.interruptions,
        "safety_locks": {
            "live_trading_enabled": False,
            "broker_used": False,
            "orders_submitted": False,
            "ml_used": False,
        },
    }
    write_json_atomic(resolved_output, manifest)
    return manifest


def _manifest_snapshot(path: Path) -> tuple[dict[str, Any], _VerifiedOosInput, str]:
    resolved = _manifest_path(path)
    manifest = _read_object(resolved, "strict OOS manifest")
    expected = {
        "schema_version",
        "created_at",
        "research_role",
        "strict_oos_evaluation_history",
        "strict_oos_start",
        "dataset",
        "market_interruptions",
        "safety_locks",
    }
    if set(manifest) != expected or manifest.get("schema_version") != _MANIFEST_SCHEMA_VERSION:
        raise StrictOosError("strict OOS manifest schema is invalid")
    if (
        manifest.get("research_role") != "strict_oos"
        or manifest.get("strict_oos_evaluation_history") is not True
        or _parse_utc(manifest.get("strict_oos_start"), "strict OOS start") != _OOS_START
    ):
        raise StrictOosError("strict OOS manifest provenance is invalid")
    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict):
        raise StrictOosError("strict OOS manifest dataset is invalid")
    verified = _verify_oos_dataset(
        _inside_workspace(dataset.get("path"), "strict OOS dataset path")
    )
    if (
        dataset != verified.dataset
        or manifest.get("market_interruptions") != verified.interruptions
    ):
        raise StrictOosError("strict OOS manifest does not match verified input")
    locks = manifest.get("safety_locks")
    if not isinstance(locks, dict) or any(value is not False for value in locks.values()):
        raise StrictOosError("strict OOS manifest safety locks are invalid")
    return manifest, verified, csv_sha256(resolved)


def _strict_metrics(
    recommendations: list[Recommendation],
    outcomes: list[RecommendationOutcome],
    provenance: RecommendationHistoryProvenance,
) -> dict[str, Any]:
    metrics = accuracy_report(recommendations, outcomes, provenance)
    for horizon, horizon_metrics in metrics["horizons"].items():
        horizon_outcomes = [outcome for outcome in outcomes if outcome.horizon == horizon]
        horizon_metrics["applicable_resolved_count"] = horizon_metrics["sample_size"]
        horizon_metrics["outcome_count"] = len(horizon_outcomes)
        horizon_metrics["insufficient_future_data_count"] = sum(
            outcome.status.value == "insufficient_future_data" for outcome in horizon_outcomes
        )
    return metrics


def run_strict_oos_evaluation(
    manifest_path: Path,
    candidate_id: str,
    output_path: Path,
    settings: BotSettings,
    *,
    overwrite: bool,
    now: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Evaluate an already registered candidate without tuning it from OOS results."""

    if candidate_id not in experiments._CANDIDATES:
        raise StrictOosError("unknown or unregistered strict OOS candidate")
    if settings.bot_mode == "live" or settings.ml_filter_enabled:
        raise StrictOosError("strict OOS evaluation requires rule-only non-live settings")
    try:
        experiments._validate_candidate_settings(candidate_id, settings)
    except experiments.RecommendationExperimentError as exc:
        raise StrictOosError(str(exc)) from exc
    resolved_output = _output_path(output_path, _REPORT_DIRECTORY, "strict OOS report output")
    manifest, verified, manifest_sha256 = _manifest_snapshot(manifest_path)
    if resolved_output.exists() and not overwrite:
        raise StrictOosError("strict OOS report already exists; pass --overwrite after validation")
    recommendations = backfill_recommendations(
        RecommendationEngine(settings), verified.candles, verified.missing_open_times
    )
    outcomes: list[RecommendationOutcome] = evaluate_outcomes(
        recommendations, verified.candles, settings, verified.missing_open_times
    )
    metrics = _strict_metrics(
        recommendations,
        outcomes,
        RecommendationHistoryProvenance(
            True,
            # The strict dataset begins at the opening boundary; recommendations are
            # timestamped on completed candles, so provenance starts at first close.
            verified.candles[0].close_time,
            verified.dataset["csv_sha256"],
            verified.candles[0].close_time,
            verified.candles[-1].close_time,
        ),
    )
    result: dict[str, Any] = {
        "schema_version": _REPORT_SCHEMA_VERSION,
        "run_at": _utc((now or (lambda: datetime.now(UTC)))().astimezone(UTC)),
        "research_role": "strict_oos",
        "strict_oos_evaluation_history": True,
        "strict_oos_start": _utc(_OOS_START),
        "research_claim_eligible": metrics["research_claim_eligible"],
        "research_claim_eligibility_reason": metrics["research_claim_eligibility_reason"],
        "source_manifest": {
            "path": _manifest_path(manifest_path).relative_to(Path.cwd().resolve()).as_posix(),
            "sha256": manifest_sha256,
        },
        "dataset": verified.dataset,
        "candidate_id": candidate_id,
        "candidate": experiments._CANDIDATES[candidate_id],
        "recommendation_count": len(recommendations),
        "outcome_count": len(outcomes),
        "metrics": metrics,
        "disclaimer": (
            "Strict OOS research evaluation. It is not investment advice, an instruction to trade, "
            "or live trading. A statistical result does not authorize order submission."
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
