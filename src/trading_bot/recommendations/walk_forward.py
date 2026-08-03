"""Executable, development-only walk-forward protocol v1.

The protocol is deliberately separate from strict-OOS reporting.  It consumes the
same checksum-verified frozen input and immutable candidate contract as the
development experiment runner, but it never opens a 2025 candle or creates an
OOS claim.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trading_bot.data.csv_store import write_json_atomic
from trading_bot.domain.models import (
    Candle,
    Recommendation,
    RecommendationOutcome,
    RecommendationType,
)
from trading_bot.recommendations import experiments
from trading_bot.recommendations.candidate_rules import candidate_protocol
from trading_bot.recommendations.engine import (
    HORIZONS,
    RecommendationEngine,
    backfill_recommendations,
    evaluate_outcomes,
    outcome_json,
    recommendation_json,
)
from trading_bot.recommendations.selection import DevelopmentSelectionError, source_identity
from trading_bot.settings import BotSettings

_WALK_FORWARD_SCHEMA_VERSION = "1.1"
PROTOCOL_VERSION = "development_walk_forward_v1"
_WALK_FORWARD_DIRECTORY = Path("reports/research/walk-forward")
_DEVELOPMENT_START = datetime(2022, 1, 1, tzinfo=UTC)
_DEVELOPMENT_END = datetime(2025, 1, 1, tzinfo=UTC)
_MIN_FOLD_APPLICABLE = 30
_MIN_POOLED_APPLICABLE = 100
_MIN_NON_NEUTRAL_COVERAGE = 0.01
_MAX_NON_NEUTRAL_COVERAGE = 0.50
_CHANCE_ACCURACY = 0.50


class RecommendationWalkForwardError(ValueError):
    """The requested development walk-forward run is unsafe or non-reproducible."""


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    """One expanding causal context and its disjoint future validation interval."""

    identifier: str
    calibration_start: datetime
    calibration_end: datetime
    validation_start: datetime
    validation_end: datetime


FOLDS: tuple[WalkForwardFold, ...] = (
    WalkForwardFold(
        "fold_1",
        _DEVELOPMENT_START,
        datetime(2023, 1, 1, tzinfo=UTC),
        datetime(2023, 1, 1, tzinfo=UTC),
        datetime(2023, 9, 1, tzinfo=UTC),
    ),
    WalkForwardFold(
        "fold_2",
        _DEVELOPMENT_START,
        datetime(2023, 9, 1, tzinfo=UTC),
        datetime(2023, 9, 1, tzinfo=UTC),
        datetime(2024, 5, 1, tzinfo=UTC),
    ),
    WalkForwardFold(
        "fold_3",
        _DEVELOPMENT_START,
        datetime(2024, 5, 1, tzinfo=UTC),
        datetime(2024, 5, 1, tzinfo=UTC),
        _DEVELOPMENT_END,
    ),
)


def _utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _walk_forward_output_path(path: Path) -> Path:
    if path.suffix != ".json" or ".." in path.parts:
        raise RecommendationWalkForwardError(
            "walk-forward output must be a .json path without traversal"
        )
    try:
        unresolved = experiments._unresolved_workspace_path(str(path), "walk-forward output")
        if unresolved.is_symlink():
            raise RecommendationWalkForwardError("walk-forward output must not be a symlink")
        workspace = Path.cwd().resolve()
        output_directory = (workspace / _WALK_FORWARD_DIRECTORY).resolve()
        resolved = unresolved.resolve()
        output_directory.relative_to(workspace)
        resolved.relative_to(output_directory)
    except RecommendationWalkForwardError:
        raise
    except (OSError, ValueError) as exc:
        raise RecommendationWalkForwardError(
            "walk-forward output must be inside reports/research/walk-forward"
        ) from exc
    return resolved


def _validate_protocol_input(snapshot: experiments._VerifiedExperimentInput) -> None:
    """Require the exact, fully development-only v1 input identity."""

    dataset = snapshot.manifest["dataset"]
    try:
        range_start = experiments._parse_utc(dataset["range"]["start"], "dataset range start")
        range_end = experiments._parse_utc(dataset["range"]["end"], "dataset range end")
    except experiments.RecommendationExperimentError as exc:
        raise RecommendationWalkForwardError("walk-forward dataset range is invalid") from exc
    strict_oos_start = experiments._parse_utc(
        snapshot.manifest["strict_oos_start"], "strict OOS start"
    )
    if (
        range_start != _DEVELOPMENT_START
        or range_end != _DEVELOPMENT_END
        or strict_oos_start != _DEVELOPMENT_END
    ):
        raise RecommendationWalkForwardError(
            "walk-forward protocol v1 requires the exact 2022-2024 development range"
        )
    if not snapshot.candles or any(
        candle.close_time > _DEVELOPMENT_END for candle in snapshot.candles
    ):
        raise RecommendationWalkForwardError("walk-forward input must not contain 2025 candles")


def _fold_candles(candles: Sequence[Candle], fold: WalkForwardFold) -> list[Candle]:
    """Return only causal context ending at this fold's validation boundary."""

    return [
        candle
        for candle in candles
        if fold.calibration_start <= candle.open_time and candle.close_time <= fold.validation_end
    ]


def _validation_recommendations(
    recommendations: Sequence[Recommendation], fold: WalkForwardFold
) -> list[Recommendation]:
    return [
        recommendation
        for recommendation in recommendations
        if fold.validation_start <= recommendation.signal_candle_time < fold.validation_end
    ]


def _metrics(
    recommendations: Sequence[Recommendation], outcomes: Sequence[RecommendationOutcome]
) -> dict[str, Any]:
    report = experiments._development_metrics(list(recommendations), list(outcomes))
    horizons = report["horizons"]
    non_neutral = sum(
        recommendation.recommendation != RecommendationType.NEUTRAL
        for recommendation in recommendations
    )
    non_neutral_coverage = non_neutral / len(recommendations) if recommendations else 0.0
    for horizon in HORIZONS:
        metrics = horizons[horizon]
        metrics["applicable_resolved_count"] = metrics["sample_size"]
        metrics["non_neutral_coverage"] = non_neutral_coverage
        metrics["after_cost_directional_accuracy"] = metrics["directional_accuracy"]
        metrics["insufficient_future_data_count"] = sum(
            outcome.horizon == horizon and outcome.status.value == "insufficient_future_data"
            for outcome in outcomes
        )
    return report


def _fold_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    horizon_results: dict[str, dict[str, Any]] = {}
    for horizon in HORIZONS:
        value = metrics["horizons"][horizon]
        sample = value["applicable_resolved_count"]
        coverage = value["non_neutral_coverage"]
        accuracy = value["after_cost_directional_accuracy"]
        passed = (
            isinstance(sample, int)
            and sample >= _MIN_FOLD_APPLICABLE
            and isinstance(coverage, float)
            and _MIN_NON_NEUTRAL_COVERAGE <= coverage <= _MAX_NON_NEUTRAL_COVERAGE
            and isinstance(accuracy, float)
            and accuracy > _CHANCE_ACCURACY
        )
        horizon_results[horizon] = {
            "passed": passed,
            "applicable_resolved_count": sample,
            "non_neutral_coverage": coverage,
            "after_cost_directional_accuracy": accuracy,
        }
    return {
        "passed": all(item["passed"] for item in horizon_results.values()),
        "horizons": horizon_results,
    }


def _pooled_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    horizon_results: dict[str, dict[str, Any]] = {}
    for horizon in HORIZONS:
        value = metrics["horizons"][horizon]
        statistic = value["statistical_result"]
        sample = value["applicable_resolved_count"]
        lower_bound = statistic["two_sided_95_percent_exact_lower_bound"]
        passed = (
            isinstance(sample, int)
            and sample >= _MIN_POOLED_APPLICABLE
            and isinstance(lower_bound, float)
            and lower_bound > _CHANCE_ACCURACY
        )
        horizon_results[horizon] = {
            "passed": passed,
            "applicable_resolved_count": sample,
            "two_sided_95_percent_exact_lower_bound": lower_bound,
        }
    return {
        "passed": all(item["passed"] for item in horizon_results.values()),
        "horizons": horizon_results,
    }


def development_selection_gate(
    fold_metrics: Sequence[Mapping[str, Any]], pooled_metrics: Mapping[str, Any]
) -> dict[str, Any]:
    """Execute the fixed v1 development gate without making an OOS claim."""

    fold_results = [_fold_gate(metrics) for metrics in fold_metrics]
    pooled_result = _pooled_gate(pooled_metrics)
    passed = (
        bool(fold_results)
        and all(item["passed"] for item in fold_results)
        and pooled_result["passed"]
    )
    return {
        "passed": passed,
        "fold_requirements": {
            "minimum_applicable_resolved_count": _MIN_FOLD_APPLICABLE,
            "minimum_non_neutral_coverage": _MIN_NON_NEUTRAL_COVERAGE,
            "maximum_non_neutral_coverage": _MAX_NON_NEUTRAL_COVERAGE,
            "after_cost_directional_accuracy_must_exceed": _CHANCE_ACCURACY,
        },
        "pooled_requirements": {
            "minimum_applicable_resolved_count": _MIN_POOLED_APPLICABLE,
            "two_sided_95_percent_exact_lower_bound_must_exceed": _CHANCE_ACCURACY,
        },
        "fold_results": fold_results,
        "pooled_result": pooled_result,
        "reason": "all_development_gates_passed"
        if passed
        else "one_or_more_development_gates_failed",
    }


def select_candidate(results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Choose a registered candidate using v1's deterministic tie-break order."""

    registry_order = {identifier: index for index, identifier in enumerate(experiments._CANDIDATES)}
    eligible = [
        (identifier, result)
        for identifier, result in results.items()
        if identifier in registry_order and result.get("selection_gate", {}).get("passed") is True
    ]
    if not eligible:
        return {"decision": "no_policy_selected", "selected_candidate_id": None}

    def score(item: tuple[str, Mapping[str, Any]]) -> tuple[float, int, float, int]:
        identifier, result = item
        fold_metrics = result["fold_metrics"]
        all_horizons = [
            fold["metrics"]["horizons"][horizon] for fold in fold_metrics for horizon in HORIZONS
        ]
        accuracies = [value["after_cost_directional_accuracy"] for value in all_horizons]
        samples = [value["applicable_resolved_count"] for value in all_horizons]
        neutral_rates = [value["neutral_rate"] for value in all_horizons]
        return (
            -min(float(value) for value in accuracies if value is not None),
            -min(int(value) for value in samples),
            max(float(value) for value in neutral_rates),
            registry_order[identifier],
        )

    identifier, _ = min(eligible, key=score)
    return {"decision": "selected", "selected_candidate_id": identifier}


def build_development_walk_forward_report(
    manifest_path: Path,
    candidate_id: str,
    settings: BotSettings,
    *,
    now: Callable[[], datetime] | None = None,
    identity: Callable[[], dict[str, Any]] = source_identity,
) -> dict[str, Any]:
    """Deterministically compute v1 evidence without publishing any artifact."""

    if candidate_id not in experiments._CANDIDATES:
        raise RecommendationWalkForwardError("unknown or unregistered walk-forward candidate")
    if settings.bot_mode == "live" or settings.ml_filter_enabled:
        raise RecommendationWalkForwardError("walk-forward requires rule-only non-live settings")
    try:
        current_identity = identity()
        experiments._validate_candidate_settings(candidate_id, settings)
        snapshot = experiments._verified_experiment_input(manifest_path)
    except (DevelopmentSelectionError, experiments.RecommendationExperimentError) as exc:
        raise RecommendationWalkForwardError(str(exc)) from exc
    _validate_protocol_input(snapshot)

    fold_results: list[dict[str, Any]] = []
    all_recommendations: list[Recommendation] = []
    all_outcomes: list[RecommendationOutcome] = []
    engine = RecommendationEngine(settings, candidate_id=candidate_id)
    for fold in FOLDS:
        context = _fold_candles(snapshot.candles, fold)
        generated = backfill_recommendations(engine, context, snapshot.missing_open_times)
        recommendations = _validation_recommendations(generated, fold)
        outcomes = evaluate_outcomes(
            recommendations, context, settings, snapshot.missing_open_times
        )
        metrics = _metrics(recommendations, outcomes)
        all_recommendations.extend(recommendations)
        all_outcomes.extend(outcomes)
        fold_results.append(
            {
                "fold_id": fold.identifier,
                "calibration_window": {
                    "start": _utc(fold.calibration_start),
                    "end": _utc(fold.calibration_end),
                },
                "validation_window": {
                    "start": _utc(fold.validation_start),
                    "end": _utc(fold.validation_end),
                },
                "recommendation_count": len(recommendations),
                "outcome_count": len(outcomes),
                "recommendations": [recommendation_json(item) for item in recommendations],
                "outcomes": [outcome_json(item) for item in outcomes],
                "metrics": metrics,
            }
        )
    pooled_metrics = _metrics(all_recommendations, all_outcomes)
    selection_gate = development_selection_gate(
        [item["metrics"] for item in fold_results], pooled_metrics
    )
    candidate_result: dict[str, Any] = {
        "fold_metrics": fold_results,
        "pooled_metrics": pooled_metrics,
        "selection_gate": selection_gate,
    }
    decision = select_candidate({candidate_id: candidate_result})
    dataset = snapshot.manifest["dataset"]
    run_time = (now or (lambda: datetime.now(UTC)))().astimezone(UTC)
    result: dict[str, Any] = {
        "schema_version": _WALK_FORWARD_SCHEMA_VERSION,
        "protocol_version": candidate_protocol(candidate_id),
        "code_revision": current_identity["revision"],
        "source_identity": current_identity,
        "run_at": _utc(run_time),
        "candidate_id": candidate_id,
        "candidate": experiments._CANDIDATES[candidate_id],
        "research_role": "development",
        "strict_oos_evaluation_history": False,
        "research_claim_eligible": False,
        "research_claim_eligibility_reason": "development_dataset_not_strict_oos",
        "source_manifest": {
            "path": snapshot.manifest_path.relative_to(Path.cwd().resolve()).as_posix(),
            "sha256": snapshot.manifest_sha256,
        },
        "dataset": {
            "path": dataset["path"],
            "csv_sha256": dataset["csv_sha256"],
            "generation_id": dataset["generation_id"],
            "range": dataset["range"],
            "candle_count": dataset["candle_count"],
        },
        "folds": fold_results,
        "pooled_metrics": pooled_metrics,
        "selection_gate": selection_gate,
        "selection_decision": decision,
        "disclaimer": (
            "Development-only walk-forward research. It is not strict OOS evidence, a public "
            "accuracy claim, investment advice, or an instruction to trade."
        ),
        "safety_locks": {
            "live_trading_enabled": False,
            "broker_used": False,
            "orders_submitted": False,
            "ml_inference_used": False,
        },
    }
    return result


def run_development_walk_forward(
    manifest_path: Path,
    candidate_id: str,
    output_path: Path,
    settings: BotSettings,
    *,
    overwrite: bool,
    now: Callable[[], datetime] | None = None,
    identity: Callable[[], dict[str, Any]] = source_identity,
) -> dict[str, Any]:
    """Run and atomically publish the immutable v1 development report."""

    resolved_output = _walk_forward_output_path(output_path)
    result = build_development_walk_forward_report(
        manifest_path, candidate_id, settings, now=now, identity=identity
    )
    if resolved_output.exists() and not overwrite:
        raise RecommendationWalkForwardError("walk-forward output already exists; pass --overwrite")
    write_json_atomic(resolved_output, result)
    return result
