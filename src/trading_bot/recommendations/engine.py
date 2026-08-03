"""Recommendation-only research path; it deliberately has no execution dependencies."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from trading_bot.data.csv_store import write_json_atomic
from trading_bot.data.validation import CandleValidationError, validate_candles
from trading_bot.domain.models import (
    Candle,
    Recommendation,
    RecommendationOutcome,
    RecommendationOutcomeStatus,
    RecommendationType,
)
from trading_bot.features.pipeline import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION, build_features
from trading_bot.settings import BotSettings
from trading_bot.strategy.ema_volume_atr import EmaVolumeAtrStrategy

HORIZONS: tuple[str, ...] = ("1h", "4h", "24h")
_HORIZON_CANDLES = {"1h": 1, "4h": 4, "24h": 24}
_RULE_CONFIDENCE = 0.55
_MIN_CONCLUSIVE_SAMPLES = 30
_MIN_RESEARCH_CLAIM_SAMPLES = 100
_RESEARCH_CLAIM_CONFIDENCE = 0.95
_RESEARCH_CLAIM_CHANCE_THRESHOLD = 0.5
_BETA_FRACTION_MAX_ITERATIONS = 200
_BETA_FRACTION_EPSILON = 3.0e-14
_BETA_FRACTION_MINIMUM = 1.0e-300


class RecommendationError(ValueError):
    """A persisted recommendation history is malformed or incompatible."""


class ProbabilityModel(Protocol):
    """Optional inference-only model contract; no model is loaded by the CLI."""

    model_version: str
    feature_schema_version: str
    production_eligible: bool
    live_trading_enabled: bool

    def probability_up(self, values: Mapping[str, float]) -> float: ...


@dataclass(frozen=True, slots=True)
class RecommendationReport:
    recommendation: Recommendation
    feature_count: int


@dataclass(frozen=True, slots=True)
class RecommendationHistoryProvenance:
    """Audit metadata that prevents strict OOS evidence from being mixed."""

    strict_oos: bool
    evaluation_start: datetime | None = None
    input_sha256: str | None = None
    input_first_close: datetime | None = None
    input_last_close: datetime | None = None


def _utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_utc(value: object, field: str) -> datetime:
    if not isinstance(value, str):
        raise RecommendationError(f"{field} must be an ISO UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RecommendationError(f"{field} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise RecommendationError(f"{field} must be UTC")
    return parsed.astimezone(UTC)


def _decimal(value: object, field: str) -> Decimal:
    if not isinstance(value, str):
        raise RecommendationError(f"{field} must be a decimal string")
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise RecommendationError(f"{field} is invalid") from exc
    if not result.is_finite():
        raise RecommendationError(f"{field} must be finite")
    return result


class RecommendationEngine:
    """Generate one causal, non-executable recommendation from closed BTC/USDT candles."""

    def __init__(
        self,
        settings: BotSettings,
        model: ProbabilityModel | None = None,
        require_model: bool = False,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        if settings.bot_mode == "live":
            raise RecommendationError("live mode is disabled")
        self.settings = settings
        self.strategy = EmaVolumeAtrStrategy(settings)
        self.model = model
        self.require_model = require_model
        self.now = now or (lambda: datetime.now(UTC))

    def recommend(
        self,
        candles: Sequence[Candle],
        allowed_missing_open_times: set[datetime] | None = None,
        *,
        created_at: datetime | None = None,
    ) -> RecommendationReport:
        items = list(candles)
        if not items:
            raise RecommendationError("recommendations require at least one candle")
        try:
            segments = continuous_candle_segments(items, allowed_missing_open_times)
        except CandleValidationError:
            return RecommendationReport(
                self._neutral(
                    items[-1],
                    "invalid_or_gapped_candle_data",
                    "invalid_or_gapped_candle_data",
                ),
                0,
            )
        latest_segment = segments[-1]
        features = build_features(latest_segment)
        return self._recommend_from_causal_feature_window(
            latest_segment[-1], features, created_at=created_at
        )

    def _recommend_from_causal_feature_window(
        self, candle: Candle, features: pd.DataFrame, *, created_at: datetime | None = None
    ) -> RecommendationReport:
        """Recommend from a causal feature window ending at ``candle``."""

        current = features.iloc[-1]
        needed = ["ema20", "ema50", "ema200", "volume_sma20", "atr14"]
        if len(features) < 2 or current[needed].isna().any():
            return RecommendationReport(
                self._neutral(
                    candle,
                    "insufficient_feature_history",
                    "insufficient_feature_history",
                    created_at=created_at,
                ),
                0,
            )
        signal = self.strategy.entry(features, False, False, False)
        if signal is None:
            return RecommendationReport(
                self._neutral(candle, "no_rule_candidate", created_at=created_at),
                len(FEATURE_COLUMNS),
            )
        values = {column: float(current[column]) for column in FEATURE_COLUMNS}
        if not all(math.isfinite(value) for value in values.values()):
            return RecommendationReport(
                self._neutral(candle, "non_finite_features", created_at=created_at), 0
            )
        model_result = self._model_result(candle, values, created_at=created_at)
        if model_result is None:
            return RecommendationReport(
                self._rule_buy(candle, Decimal(str(signal.atr)), created_at=created_at), len(values)
            )
        if isinstance(model_result, Recommendation):
            return RecommendationReport(model_result, len(values))
        probability, version = model_result
        recommendation = (
            RecommendationType.BUY_BIAS
            if probability >= 0.55
            else RecommendationType.AVOID
            if probability <= 0.45
            else RecommendationType.NEUTRAL
        )
        reason = (
            "rule_candidate_ml_filter_avoid_buy"
            if recommendation == RecommendationType.AVOID
            else "rule_candidate_ml_filter"
        )
        return RecommendationReport(
            self._create(
                candle,
                recommendation,
                probability,
                abs(probability - 0.5) * 2,
                version,
                reason,
                Decimal(str(signal.atr)),
                created_at=created_at,
            ),
            len(values),
        )

    def _model_result(
        self, candle: Candle, values: Mapping[str, float], *, created_at: datetime | None = None
    ) -> tuple[float, str] | Recommendation | None:
        if self.model is None:
            return (
                self._neutral(candle, "model_required_but_missing", created_at=created_at)
                if self.require_model
                else None
            )
        if self.model.feature_schema_version != FEATURE_SCHEMA_VERSION:
            return self._neutral(candle, "model_feature_schema_mismatch", created_at=created_at)
        if not self.model.production_eligible or self.model.live_trading_enabled:
            return self._neutral(
                candle, "model_not_eligible_for_recommendations", created_at=created_at
            )
        try:
            probability = float(self.model.probability_up(values))
        except (TypeError, ValueError) as exc:
            raise RecommendationError("recommendation model inference failed") from exc
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            return self._neutral(candle, "model_probability_invalid", created_at=created_at)
        return probability, self.model.model_version

    def _neutral(
        self,
        candle: Candle,
        reason: str,
        data_quality: str = "validated_closed_contiguous",
        *,
        created_at: datetime | None = None,
    ) -> Recommendation:
        return self._create(
            candle,
            RecommendationType.NEUTRAL,
            None,
            0.0,
            None,
            reason,
            None,
            data_quality,
            created_at=created_at,
        )

    def _rule_buy(
        self, candle: Candle, atr: Decimal, *, created_at: datetime | None = None
    ) -> Recommendation:
        return self._create(
            candle,
            RecommendationType.BUY_BIAS,
            None,
            _RULE_CONFIDENCE,
            None,
            "ema_volume_atr_rule_candidate_rule_only",
            atr,
            created_at=created_at,
        )

    def _create(
        self,
        candle: Candle,
        kind: RecommendationType,
        probability_up: float | None,
        confidence: float,
        model_version: str | None,
        reason: str,
        atr: Decimal | None,
        data_quality: str = "validated_closed_contiguous",
        *,
        created_at: datetime | None = None,
    ) -> Recommendation:
        identifier = hashlib.sha256(
            f"BTC/USDT|1h|{_utc(candle.close_time)}|{FEATURE_SCHEMA_VERSION}".encode()
        ).hexdigest()[:32]
        entry = candle.close if kind != RecommendationType.AVOID else None
        return Recommendation(
            identifier,
            (created_at or self.now()).astimezone(UTC),
            candle.close_time,
            "BTC/USDT",
            "1h",
            HORIZONS,
            kind,
            probability_up,
            confidence,
            model_version,
            FEATURE_SCHEMA_VERSION,
            reason,
            data_quality,
            entry,
            entry - Decimal("2") * atr if entry is not None and atr is not None else None,
            entry + Decimal("4") * atr if entry is not None and atr is not None else None,
        )


def continuous_candle_segments(
    candles: Sequence[Candle],
    allowed_missing_open_times: set[datetime] | None = None,
) -> list[list[Candle]]:
    """Split only sidecar-verified interruptions into independent causal segments."""

    items = list(candles)
    validate_candles(items, allowed_missing_open_times=allowed_missing_open_times)
    segments: list[list[Candle]] = [[items[0]]]
    interval = timedelta(hours=1)
    for candle in items[1:]:
        if candle.open_time != segments[-1][-1].open_time + interval:
            segments.append([])
        segments[-1].append(candle)
    return segments


def evaluate_outcomes(
    recommendations: Iterable[Recommendation],
    candles: Sequence[Candle],
    settings: BotSettings,
    allowed_missing_open_times: set[datetime] | None = None,
) -> list[RecommendationOutcome]:
    items = list(candles)
    segments = continuous_candle_segments(items, allowed_missing_open_times)
    by_close = {
        candle.close_time: (segment, index)
        for segment in segments
        for index, candle in enumerate(segment)
    }
    costs = (
        settings.entry_fee_rate
        + settings.exit_fee_rate
        + settings.entry_slippage_rate
        + settings.exit_slippage_rate
    )
    outcomes: list[RecommendationOutcome] = []
    for recommendation in recommendations:
        signal = by_close.get(recommendation.signal_candle_time)
        for horizon in recommendation.horizons:
            steps = _HORIZON_CANDLES.get(horizon)
            if signal is None or steps is None:
                outcomes.append(
                    RecommendationOutcome(
                        recommendation.id,
                        horizon,
                        None,
                        None,
                        None,
                        None,
                        None,
                        RecommendationOutcomeStatus.INSUFFICIENT_FUTURE_DATA,
                    )
                )
                continue
            segment, signal_index = signal
            if signal_index + steps >= len(segment):
                outcomes.append(
                    RecommendationOutcome(
                        recommendation.id,
                        horizon,
                        None,
                        None,
                        None,
                        None,
                        None,
                        RecommendationOutcomeStatus.INSUFFICIENT_FUTURE_DATA,
                    )
                )
                continue
            future = segment[signal_index + steps]
            reference = (
                recommendation.entry_reference
                if recommendation.entry_reference is not None
                else segment[signal_index].close
            )
            raw_return = future.close / reference - Decimal("1")
            realized = raw_return - costs
            path = segment[signal_index + 1 : signal_index + steps + 1]
            invalidated = (
                bool(
                    recommendation.invalidation_price is not None
                    and any(candle.low <= recommendation.invalidation_price for candle in path)
                )
                if recommendation.recommendation != RecommendationType.AVOID
                else None
            )
            target = (
                bool(
                    not invalidated
                    and recommendation.target_price is not None
                    and any(candle.high >= recommendation.target_price for candle in path)
                )
                if recommendation.recommendation != RecommendationType.AVOID
                else None
            )
            correct = (
                realized > 0
                if recommendation.recommendation == RecommendationType.BUY_BIAS
                else realized <= 0
                if recommendation.recommendation == RecommendationType.AVOID
                else None
            )
            outcomes.append(
                RecommendationOutcome(
                    recommendation.id,
                    horizon,
                    future.close_time,
                    realized,
                    correct,
                    target,
                    invalidated,
                    RecommendationOutcomeStatus.RESOLVED,
                )
            )
    return outcomes


def merge_recommendations(
    existing: Iterable[Recommendation], updates: Iterable[Recommendation]
) -> list[Recommendation]:
    """Merge deterministic recommendations without duplicating a signal identity."""

    by_id = {item.id: item for item in existing}
    for item in updates:
        by_id.setdefault(item.id, item)
    return sorted(by_id.values(), key=lambda item: (item.signal_candle_time, item.id))


def merge_outcomes(
    existing: Iterable[RecommendationOutcome], updates: Iterable[RecommendationOutcome]
) -> list[RecommendationOutcome]:
    """Resolved outcomes are immutable; incomplete outcomes may be promoted."""

    by_key = {(item.recommendation_id, item.horizon): item for item in existing}
    for item in updates:
        key = (item.recommendation_id, item.horizon)
        current = by_key.get(key)
        if current is None or current.status != RecommendationOutcomeStatus.RESOLVED:
            by_key[key] = item
    return sorted(
        by_key.values(),
        key=lambda item: (item.recommendation_id, _HORIZON_CANDLES[item.horizon]),
    )


def validate_strict_oos_history(
    recommendations: Iterable[Recommendation],
    provenance: RecommendationHistoryProvenance | None,
) -> None:
    """Reject a strict history whose records cannot be OOS evidence."""

    if provenance is None or not provenance.strict_oos:
        return
    boundary = provenance.evaluation_start
    first_close = provenance.input_first_close
    last_close = provenance.input_last_close
    checksum = provenance.input_sha256
    if (
        not isinstance(checksum, str)
        or len(checksum) != 64
        or any(character not in "0123456789abcdef" for character in checksum)
        or first_close is None
        or last_close is None
        or boundary is None
    ):
        raise RecommendationError("strict OOS history provenance is incomplete")
    if (
        any(
            timestamp.tzinfo is None or timestamp.utcoffset() != timedelta(0)
            for timestamp in (boundary, first_close, last_close)
        )
        or not first_close <= boundary <= last_close
    ):
        raise RecommendationError("strict OOS history provenance is inconsistent")
    for recommendation in recommendations:
        if recommendation.symbol != "BTC/USDT" or recommendation.timeframe != "1h":
            raise RecommendationError("strict OOS history recommendation market is invalid")
        if recommendation.signal_candle_time < boundary:
            raise RecommendationError("strict OOS history contains pre-boundary recommendations")


def _beta_continued_fraction(alpha: float, beta: float, value: float) -> float:
    """Evaluate the continued fraction used by the regularized beta function."""

    alpha_plus_beta = alpha + beta
    alpha_plus_one = alpha + 1.0
    alpha_minus_one = alpha - 1.0
    denominator = 1.0 - alpha_plus_beta * value / alpha_plus_one
    if abs(denominator) < _BETA_FRACTION_MINIMUM:
        denominator = _BETA_FRACTION_MINIMUM
    denominator = 1.0 / denominator
    numerator_factor = 1.0
    fraction = denominator
    for iteration in range(1, _BETA_FRACTION_MAX_ITERATIONS + 1):
        doubled = 2 * iteration
        numerator = iteration * (beta - iteration) * value
        denominator_factor = (alpha_minus_one + doubled) * (alpha + doubled)
        denominator = 1.0 + numerator * denominator / denominator_factor
        if abs(denominator) < _BETA_FRACTION_MINIMUM:
            denominator = _BETA_FRACTION_MINIMUM
        numerator_factor = 1.0 + numerator / denominator_factor / numerator_factor
        if abs(numerator_factor) < _BETA_FRACTION_MINIMUM:
            numerator_factor = _BETA_FRACTION_MINIMUM
        denominator = 1.0 / denominator
        fraction *= denominator * numerator_factor

        numerator = -(alpha + iteration) * (alpha_plus_beta + iteration) * value
        denominator_factor = (alpha + doubled) * (alpha_plus_one + doubled)
        denominator = 1.0 + numerator * denominator / denominator_factor
        if abs(denominator) < _BETA_FRACTION_MINIMUM:
            denominator = _BETA_FRACTION_MINIMUM
        numerator_factor = 1.0 + numerator / denominator_factor / numerator_factor
        if abs(numerator_factor) < _BETA_FRACTION_MINIMUM:
            numerator_factor = _BETA_FRACTION_MINIMUM
        denominator = 1.0 / denominator
        delta = denominator * numerator_factor
        fraction *= delta
        if abs(delta - 1.0) <= _BETA_FRACTION_EPSILON:
            return fraction
    raise RecommendationError("exact binomial confidence interval did not converge")


def _regularized_incomplete_beta(alpha: float, beta: float, value: float) -> float:
    """Return I_x(alpha, beta) without a new scientific-computing dependency."""

    if not alpha > 0.0 or not beta > 0.0 or not 0.0 <= value <= 1.0:
        raise RecommendationError("exact binomial confidence interval inputs are invalid")
    if value == 0.0:
        return 0.0
    if value == 1.0:
        return 1.0
    log_front = (
        alpha * math.log(value)
        + beta * math.log1p(-value)
        - math.lgamma(alpha)
        - math.lgamma(beta)
        + math.lgamma(alpha + beta)
    )
    front = math.exp(log_front)
    pivot = (alpha + 1.0) / (alpha + beta + 2.0)
    if value < pivot:
        return front * _beta_continued_fraction(alpha, beta, value) / alpha
    return 1.0 - front * _beta_continued_fraction(beta, alpha, 1.0 - value) / beta


def _inverse_regularized_incomplete_beta(probability: float, alpha: float, beta: float) -> float:
    """Invert I_x(alpha, beta) with a deterministic bounded bisection search."""

    if not 0.0 <= probability <= 1.0:
        raise RecommendationError("exact binomial confidence probability is invalid")
    if probability == 0.0:
        return 0.0
    if probability == 1.0:
        return 1.0
    lower = 0.0
    upper = 1.0
    for _ in range(100):
        midpoint = (lower + upper) / 2.0
        if _regularized_incomplete_beta(alpha, beta, midpoint) < probability:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def _clopper_pearson_lower_bound(correct: int, sample_size: int) -> float | None:
    """Return the two-sided 95% exact binomial lower bound, or none with no sample."""

    if sample_size == 0:
        return None
    if correct < 0 or correct > sample_size:
        raise RecommendationError("exact binomial confidence interval counts are invalid")
    if correct == 0:
        return 0.0
    lower_tail = (1.0 - _RESEARCH_CLAIM_CONFIDENCE) / 2.0
    bound = _inverse_regularized_incomplete_beta(
        lower_tail, float(correct), float(sample_size - correct + 1)
    )
    return min(1.0, max(0.0, bound))


def backfill_recommendations(
    engine: RecommendationEngine,
    candles: Sequence[Candle],
    allowed_missing_open_times: set[datetime] | None = None,
) -> list[Recommendation]:
    """Create causal recommendations without carrying state across an interruption."""

    items = list(candles)
    segments = continuous_candle_segments(items, allowed_missing_open_times)
    recommendations: list[Recommendation] = []
    for segment in segments:
        features = build_features(segment)
        recommendations.extend(
            engine._recommend_from_causal_feature_window(
                segment[index],
                features.iloc[max(0, index - 1) : index + 1].copy(),
                created_at=segment[index].close_time,
            ).recommendation
            for index in range(len(segment))
        )
    return recommendations


def accuracy_report(
    recommendations: Iterable[Recommendation],
    outcomes: Iterable[RecommendationOutcome],
    provenance: RecommendationHistoryProvenance | None = None,
) -> dict[str, Any]:
    """Calculate metrics and keep statistical evidence separate from claim eligibility."""

    records = {item.id: item for item in recommendations}
    restored_outcomes = list(outcomes)
    strict_oos_valid = provenance is not None and provenance.strict_oos
    if strict_oos_valid:
        validate_strict_oos_history(records.values(), provenance)
    strict_oos_reason = (
        "strict_oos_provenance_valid" if strict_oos_valid else "strict_oos_provenance_required"
    )
    result: dict[str, Any] = {
        "horizons": {},
        "inconclusive": True,
        "strict_oos": strict_oos_valid,
        "strict_oos_validation": strict_oos_reason,
    }
    for horizon in HORIZONS:
        resolved = [
            item
            for item in restored_outcomes
            if item.horizon == horizon and item.status == RecommendationOutcomeStatus.RESOLVED
        ]
        applicable = [
            item
            for item in resolved
            if records.get(item.recommendation_id) is not None
            and records[item.recommendation_id].recommendation != RecommendationType.NEUTRAL
        ]
        buy = [
            item
            for item in applicable
            if records[item.recommendation_id].recommendation == RecommendationType.BUY_BIAS
        ]
        avoid = [
            item
            for item in applicable
            if records[item.recommendation_id].recommendation == RecommendationType.AVOID
        ]
        neutral_count = sum(
            record.recommendation == RecommendationType.NEUTRAL for record in records.values()
        )
        correct = sum(item.direction_correct is True for item in applicable)
        probabilities: list[tuple[float, bool]] = []
        for item in resolved:
            record = records.get(item.recommendation_id)
            if (
                record is not None
                and record.probability_up is not None
                and item.realized_return is not None
            ):
                probabilities.append((record.probability_up, item.realized_return > 0))
        brier = (
            sum((probability - float(up)) ** 2 for probability, up in probabilities)
            / len(probabilities)
            if probabilities
            else None
        )
        calibration = (
            {
                "sample_size": len(probabilities),
                "mean_predicted_probability_up": sum(
                    probability for probability, _ in probabilities
                )
                / len(probabilities),
                "observed_up_rate": sum(up for _, up in probabilities) / len(probabilities),
            }
            if probabilities
            else None
        )
        sample = len(applicable)
        accuracy = correct / sample if sample else None
        exact_lower_bound = _clopper_pearson_lower_bound(correct, sample)
        statistical_gate_passed = (
            sample >= _MIN_RESEARCH_CLAIM_SAMPLES
            and exact_lower_bound is not None
            and exact_lower_bound > _RESEARCH_CLAIM_CHANCE_THRESHOLD
        )
        research_claim_eligible = strict_oos_valid and statistical_gate_passed
        eligibility_reason = (
            "eligible"
            if research_claim_eligible
            else strict_oos_reason
            if not strict_oos_valid
            else "fewer_than_100_applicable_resolved_samples"
            if sample < _MIN_RESEARCH_CLAIM_SAMPLES
            else "exact_95_percent_lower_bound_not_above_50_percent"
        )
        result["horizons"][horizon] = {
            "total_recommendations": len(records),
            "resolved_recommendations": len(resolved),
            "coverage": len(resolved) / len(records) if records else 0.0,
            "directional_accuracy": accuracy,
            "buy_bias_precision": sum(item.direction_correct is True for item in buy) / len(buy)
            if buy
            else None,
            "avoid_precision": sum(item.direction_correct is True for item in avoid) / len(avoid)
            if avoid
            else None,
            "neutral_rate": neutral_count / len(records) if records else 0.0,
            "brier_score": brier,
            "calibration": calibration,
            "sample_size": sample,
            "statistical_gate_passed": statistical_gate_passed,
            "statistical_result": {
                "minimum_applicable_resolved_samples": _MIN_RESEARCH_CLAIM_SAMPLES,
                "applicable_resolved_samples": sample,
                "two_sided_95_percent_exact_lower_bound": exact_lower_bound,
                "lower_bound_must_exceed": _RESEARCH_CLAIM_CHANCE_THRESHOLD,
                "passed": statistical_gate_passed,
            },
            "inconclusive": sample < _MIN_CONCLUSIVE_SAMPLES,
            "research_claim_eligible": research_claim_eligible,
            "research_claim_eligibility_reason": eligibility_reason,
        }
    result["inconclusive"] = all(item["inconclusive"] for item in result["horizons"].values())
    result["research_claim_eligible"] = all(
        item["research_claim_eligible"] for item in result["horizons"].values()
    )
    result["research_claim_eligibility_reason"] = (
        "eligible"
        if result["research_claim_eligible"]
        else strict_oos_reason
        if not strict_oos_valid
        else "one_or_more_horizons_failed_the_statistical_gate"
    )
    return result


def recommendation_json(item: Recommendation) -> dict[str, Any]:
    payload = asdict(item)
    # Keep the serialized evidence stable across JSON write/read/replay.  The
    # domain model intentionally uses an immutable tuple, whereas JSON arrays
    # always deserialize as lists.
    payload["horizons"] = list(item.horizons)
    payload["created_at"] = _utc(item.created_at)
    payload["signal_candle_time"] = _utc(item.signal_candle_time)
    payload["recommendation"] = item.recommendation.value
    payload["entry_reference"] = (
        str(item.entry_reference) if item.entry_reference is not None else None
    )
    payload["invalidation_price"] = (
        str(item.invalidation_price) if item.invalidation_price is not None else None
    )
    payload["target_price"] = str(item.target_price) if item.target_price is not None else None
    return payload


def outcome_json(item: RecommendationOutcome) -> dict[str, Any]:
    payload = asdict(item)
    payload["resolved_at"] = _utc(item.resolved_at) if item.resolved_at else None
    payload["realized_return"] = (
        str(item.realized_return) if item.realized_return is not None else None
    )
    payload["status"] = item.status.value
    return payload


def _recommendation_from_json(value: object) -> Recommendation:
    if not isinstance(value, dict):
        raise RecommendationError("recommendation record is invalid")
    try:
        kind = RecommendationType(value["recommendation"])
        horizons = tuple(value["horizons"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RecommendationError("recommendation record is invalid") from exc
    if horizons != HORIZONS or not isinstance(value.get("id"), str):
        raise RecommendationError("recommendation record has unsupported schema")
    probability = value.get("probability_up")
    confidence = value.get("confidence")
    if probability is not None and (
        not isinstance(probability, (int, float)) or not 0 <= probability <= 1
    ):
        raise RecommendationError("recommendation probability is invalid")
    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise RecommendationError("recommendation confidence is invalid")
    for field in ("symbol", "timeframe", "feature_schema_version", "rule_reason", "data_quality"):
        if not isinstance(value.get(field), str):
            raise RecommendationError(f"recommendation {field} is invalid")
    if value["symbol"] != "BTC/USDT" or value["timeframe"] != "1h":
        raise RecommendationError("recommendation market is invalid")
    model_version = value.get("model_version")
    if model_version is not None and not isinstance(model_version, str):
        raise RecommendationError("recommendation model version is invalid")
    invalidation = value.get("invalidation_price")
    target = value.get("target_price")
    entry = value.get("entry_reference")
    if kind == RecommendationType.AVOID:
        # Legacy histories could contain long levels for AVOID. Discard them so an
        # avoid-buy recommendation can never be presented as a short trade.
        if entry is not None:
            _decimal(entry, "entry_reference")
        if invalidation is not None:
            _decimal(invalidation, "invalidation_price")
        if target is not None:
            _decimal(target, "target_price")
        parsed_entry = None
    else:
        parsed_entry = _decimal(entry, "entry_reference")
    return Recommendation(
        value["id"],
        _parse_utc(value.get("created_at"), "created_at"),
        _parse_utc(value.get("signal_candle_time"), "signal_candle_time"),
        value["symbol"],
        value["timeframe"],
        horizons,
        kind,
        float(probability) if probability is not None else None,
        float(confidence),
        model_version,
        value["feature_schema_version"],
        value["rule_reason"],
        value["data_quality"],
        parsed_entry,
        (
            _decimal(invalidation, "invalidation_price")
            if kind != RecommendationType.AVOID and invalidation is not None
            else None
        ),
        (
            _decimal(target, "target_price")
            if kind != RecommendationType.AVOID and target is not None
            else None
        ),
    )


def _outcome_from_json(value: object) -> RecommendationOutcome:
    if not isinstance(value, dict):
        raise RecommendationError("outcome record is invalid")
    try:
        status = RecommendationOutcomeStatus(value["status"])
    except (KeyError, ValueError) as exc:
        raise RecommendationError("outcome status is invalid") from exc
    if not isinstance(value.get("recommendation_id"), str) or value.get("horizon") not in HORIZONS:
        raise RecommendationError("outcome record is invalid")
    return RecommendationOutcome(
        value["recommendation_id"],
        value["horizon"],
        _parse_utc(value["resolved_at"], "resolved_at") if value.get("resolved_at") else None,
        _decimal(value["realized_return"], "realized_return")
        if value.get("realized_return")
        else None,
        value.get("direction_correct")
        if isinstance(value.get("direction_correct"), bool)
        else None,
        value.get("target_hit") if isinstance(value.get("target_hit"), bool) else None,
        value.get("invalidation_hit") if isinstance(value.get("invalidation_hit"), bool) else None,
        status,
    )


class RecommendationHistoryStore:
    """Atomic, secret-free local recommendation and outcome history."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> tuple[list[Recommendation], list[RecommendationOutcome]]:
        recommendations, outcomes, _, _ = self.load_with_provenance()
        return recommendations, outcomes

    def load_with_provenance(
        self,
    ) -> tuple[
        list[Recommendation],
        list[RecommendationOutcome],
        RecommendationHistoryProvenance | None,
        bool,
    ]:
        """Load records plus provenance; ``legacy`` is true for schema 1.0 files."""

        if not self.path.exists():
            return [], [], None, False
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RecommendationError("could not read recommendation history") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") not in {"1.0", "1.1"}:
            raise RecommendationError("unsupported recommendation history schema")
        recommendations = payload.get("recommendations")
        outcomes = payload.get("outcomes")
        if not isinstance(recommendations, list) or not isinstance(outcomes, list):
            raise RecommendationError("recommendation history records are invalid")
        records = [_recommendation_from_json(item) for item in recommendations]
        restored_outcomes = [_outcome_from_json(item) for item in outcomes]
        if payload["schema_version"] == "1.0":
            return records, restored_outcomes, None, True
        provenance = _provenance_from_json(payload.get("provenance"))
        return records, restored_outcomes, provenance, False

    def save(
        self,
        recommendations: Iterable[Recommendation],
        outcomes: Iterable[RecommendationOutcome],
        provenance: RecommendationHistoryProvenance | None = None,
    ) -> None:
        current_provenance = provenance or RecommendationHistoryProvenance(False)
        write_json_atomic(
            self.path,
            {
                "schema_version": "1.1",
                "provenance": _provenance_json(current_provenance),
                "recommendations": [recommendation_json(item) for item in recommendations],
                "outcomes": [outcome_json(item) for item in outcomes],
            },
        )


def _provenance_json(provenance: RecommendationHistoryProvenance) -> dict[str, Any]:
    return {
        "strict_oos": provenance.strict_oos,
        "evaluation_start": _utc(provenance.evaluation_start)
        if provenance.evaluation_start is not None
        else None,
        "input_sha256": provenance.input_sha256,
        "input_first_close": _utc(provenance.input_first_close)
        if provenance.input_first_close is not None
        else None,
        "input_last_close": _utc(provenance.input_last_close)
        if provenance.input_last_close is not None
        else None,
    }


def _provenance_from_json(value: object) -> RecommendationHistoryProvenance:
    expected_fields = {
        "strict_oos",
        "evaluation_start",
        "input_sha256",
        "input_first_close",
        "input_last_close",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected_fields
        or not isinstance(value.get("strict_oos"), bool)
    ):
        raise RecommendationError("recommendation history provenance is invalid")
    strict_oos = value["strict_oos"]
    evaluation_start = value.get("evaluation_start")
    input_sha256 = value.get("input_sha256")
    first_close = value.get("input_first_close")
    last_close = value.get("input_last_close")
    if strict_oos:
        if (
            not isinstance(input_sha256, str)
            or len(input_sha256) != 64
            or any(character not in "0123456789abcdef" for character in input_sha256)
        ):
            raise RecommendationError("strict OOS history input checksum is invalid")
        if evaluation_start is None or first_close is None or last_close is None:
            raise RecommendationError("strict OOS history provenance is incomplete")
    for field, item in (
        ("input_sha256", input_sha256),
        ("evaluation_start", evaluation_start),
        ("input_first_close", first_close),
        ("input_last_close", last_close),
    ):
        if item is not None and not isinstance(item, str):
            raise RecommendationError(f"history provenance {field} is invalid")
    parsed_boundary = (
        _parse_utc(evaluation_start, "evaluation_start") if evaluation_start is not None else None
    )
    parsed_first_close = (
        _parse_utc(first_close, "input_first_close") if first_close is not None else None
    )
    parsed_last_close = (
        _parse_utc(last_close, "input_last_close") if last_close is not None else None
    )
    if strict_oos and (
        parsed_boundary is None
        or parsed_first_close is None
        or parsed_last_close is None
        or parsed_first_close > parsed_last_close
        or not parsed_first_close <= parsed_boundary <= parsed_last_close
    ):
        raise RecommendationError("strict OOS history provenance is inconsistent")
    return RecommendationHistoryProvenance(
        strict_oos,
        parsed_boundary,
        input_sha256,
        parsed_first_close,
        parsed_last_close,
    )
