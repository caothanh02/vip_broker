from __future__ import annotations

import inspect
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from trading_bot.cli import main
from trading_bot.domain.models import (
    Candle,
    Recommendation,
    RecommendationOutcomeStatus,
    RecommendationType,
)
from trading_bot.recommendations import experiments, walk_forward
from trading_bot.recommendations.engine import (
    RecommendationEngine,
    backfill_recommendations,
    evaluate_outcomes,
)
from trading_bot.settings import BotSettings


def _candle(
    open_time: datetime,
    close: Decimal = Decimal("100"),
    volume: Decimal = Decimal("1000"),
) -> Candle:
    return Candle(
        open_time,
        open_time + timedelta(hours=1),
        "BTC/USDT",
        "1h",
        close,
        close + Decimal("1"),
        close - Decimal("1"),
        close,
        volume,
        True,
    )


def _recommendation(signal_time: datetime, identifier: str = "recommendation") -> Recommendation:
    return Recommendation(
        identifier,
        signal_time,
        signal_time,
        "BTC/USDT",
        "1h",
        ("1h", "4h", "24h"),
        RecommendationType.BUY_BIAS,
        None,
        0.55,
        None,
        "test_feature_schema",
        "test_rule_candidate",
        "validated_closed_contiguous",
        Decimal("100"),
        Decimal("98"),
        Decimal("104"),
    )


def _metric(
    *, sample: int, coverage: float, accuracy: float | None, lower: float | None
) -> dict[str, Any]:
    return {
        "horizons": {
            horizon: {
                "applicable_resolved_count": sample,
                "non_neutral_coverage": coverage,
                "after_cost_directional_accuracy": accuracy,
                "neutral_rate": 1.0 - coverage,
                "statistical_result": {"two_sided_95_percent_exact_lower_bound": lower},
            }
            for horizon in ("1h", "4h", "24h")
        }
    }


def _passing_fold_metrics() -> list[dict[str, Any]]:
    return [_metric(sample=30, coverage=0.10, accuracy=0.60, lower=0.40) for _ in range(3)]


def test_protocol_v1_folds_are_fixed_and_leave_2025_sealed() -> None:
    assert [fold.identifier for fold in walk_forward.FOLDS] == ["fold_1", "fold_2", "fold_3"]
    assert walk_forward.FOLDS[0].calibration_start == datetime(2022, 1, 1, tzinfo=UTC)
    assert walk_forward.FOLDS[0].validation_end == datetime(2023, 9, 1, tzinfo=UTC)
    assert walk_forward.FOLDS[1].validation_end == datetime(2024, 5, 1, tzinfo=UTC)
    assert walk_forward.FOLDS[2].validation_end == datetime(2025, 1, 1, tzinfo=UTC)
    assert all(
        fold.validation_end <= datetime(2025, 1, 1, tzinfo=UTC) for fold in walk_forward.FOLDS
    )


def test_validation_filter_scores_only_decisions_inside_fold() -> None:
    fold = walk_forward.FOLDS[1]
    before = _recommendation(fold.validation_start - timedelta(hours=1), "before")
    inside = _recommendation(fold.validation_start, "inside")
    after = _recommendation(fold.validation_end, "after")

    assert walk_forward._validation_recommendations([before, inside, after], fold) == [inside]


def test_fold_end_uses_insufficient_outcome_instead_of_next_fold_candle() -> None:
    fold = walk_forward.FOLDS[0]
    context = [
        _candle(fold.validation_end - timedelta(hours=30) + timedelta(hours=index))
        for index in range(30)
    ]
    recommendation = _recommendation(fold.validation_end - timedelta(hours=1))

    outcomes = evaluate_outcomes([recommendation], context, BotSettings())

    by_horizon = {outcome.horizon: outcome for outcome in outcomes}
    assert by_horizon["24h"].status == RecommendationOutcomeStatus.INSUFFICIENT_FUTURE_DATA


def test_audited_gap_is_a_segment_boundary_for_outcomes() -> None:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    candles = [
        _candle(base),
        _candle(base + timedelta(hours=1)),
        _candle(base + timedelta(hours=3)),
    ]
    recommendation = _recommendation(base + timedelta(hours=2))

    outcomes = evaluate_outcomes(
        [recommendation], candles, BotSettings(), {base + timedelta(hours=2)}
    )

    assert all(
        outcome.status == RecommendationOutcomeStatus.INSUFFICIENT_FUTURE_DATA
        for outcome in outcomes
    )


def test_real_strategy_recommendation_is_unchanged_when_future_candles_mutate() -> None:
    base = datetime(2022, 1, 1, tzinfo=UTC)
    candles = []
    for index in range(270):
        close = (
            Decimal("120") - Decimal(index) * Decimal("0.1") + Decimal(index % 2) * Decimal("0.2")
            if index < 220
            else Decimal("150") + Decimal(index - 220) * Decimal("3")
        )
        candles.append(
            _candle(
                base + timedelta(hours=index),
                close,
                Decimal("5000") if index >= 220 else Decimal("1000"),
            )
        )
    engine = RecommendationEngine(BotSettings(), now=lambda: datetime(2026, 1, 1, tzinfo=UTC))
    baseline = backfill_recommendations(engine, candles)
    candidate_index = next(
        index for index, item in enumerate(baseline) if item.rule_reason.startswith("ema_volume")
    )
    mutated = list(candles)
    for index in range(candidate_index + 1, len(mutated)):
        candle = mutated[index]
        mutated[index] = Candle(
            candle.open_time,
            candle.close_time,
            candle.symbol,
            candle.timeframe,
            candle.open + Decimal("100"),
            candle.high + Decimal("100"),
            candle.low + Decimal("100"),
            candle.close + Decimal("100"),
            candle.volume * Decimal("2"),
            True,
        )

    assert backfill_recommendations(engine, mutated)[candidate_index] == baseline[candidate_index]


@pytest.mark.parametrize(
    ("metrics", "message"),
    [
        (_metric(sample=29, coverage=0.10, accuracy=0.60, lower=0.40), "sample"),
        (_metric(sample=30, coverage=0.51, accuracy=0.60, lower=0.40), "coverage"),
        (_metric(sample=30, coverage=0.10, accuracy=0.50, lower=0.40), "accuracy"),
    ],
)
def test_selection_gate_rejects_each_fold_requirement(
    metrics: dict[str, Any], message: str
) -> None:
    result = walk_forward.development_selection_gate(
        [metrics, *_passing_fold_metrics()[1:]],
        _metric(sample=100, coverage=0.10, accuracy=0.60, lower=0.51),
    )

    assert result["passed"] is False, message


def test_selection_gate_requires_pooled_exact_lower_bound() -> None:
    result = walk_forward.development_selection_gate(
        _passing_fold_metrics(), _metric(sample=100, coverage=0.10, accuracy=0.60, lower=0.50)
    )

    assert result["passed"] is False


def test_selection_gate_passes_only_when_all_fold_and_pooled_conditions_pass() -> None:
    result = walk_forward.development_selection_gate(
        _passing_fold_metrics(), _metric(sample=100, coverage=0.10, accuracy=0.60, lower=0.51)
    )

    assert result["passed"] is True


def test_deterministic_tie_break_and_no_policy_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        experiments._CANDIDATES,
        "second_registered",
        experiments._CANDIDATES["baseline_ema_volume_atr_v1"],
    )
    passing = {
        "fold_metrics": [{"metrics": item} for item in _passing_fold_metrics()],
        "selection_gate": {"passed": True},
    }
    decision = walk_forward.select_candidate(
        {"second_registered": passing, "baseline_ema_volume_atr_v1": passing}
    )

    assert decision == {
        "decision": "selected",
        "selected_candidate_id": "baseline_ema_volume_atr_v1",
    }
    assert walk_forward.select_candidate(
        {"baseline_ema_volume_atr_v1": {"selection_gate": {"passed": False}}}
    ) == {
        "decision": "no_policy_selected",
        "selected_candidate_id": None,
    }


def test_candidate_and_unsafe_output_are_rejected_before_input_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        experiments,
        "_verified_experiment_input",
        lambda _: (_ for _ in ()).throw(AssertionError("input must not be read")),
    )

    with pytest.raises(walk_forward.RecommendationWalkForwardError, match="unregistered"):
        walk_forward.run_development_walk_forward(
            Path("reports/research/manifests/input.json"),
            "unknown",
            Path("reports/research/walk-forward/out.json"),
            BotSettings(),
            overwrite=False,
        )
    with pytest.raises(walk_forward.RecommendationWalkForwardError, match="output"):
        walk_forward.run_development_walk_forward(
            Path("reports/research/manifests/input.json"),
            "baseline_ema_volume_atr_v1",
            Path("reports/research/out.json"),
            BotSettings(),
            overwrite=False,
        )


def test_runner_writes_development_only_report_with_validation_windows_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest_path = tmp_path / "reports/research/manifests/development.json"
    snapshot = experiments._VerifiedExperimentInput(
        manifest_path,
        tmp_path / "data/raw/development.csv",
        [
            _candle(datetime(2022, 1, 1, tzinfo=UTC)),
            *[_candle(fold.validation_end - timedelta(hours=1)) for fold in walk_forward.FOLDS],
        ],
        set(),
        {
            "strict_oos_start": "2025-01-01T00:00:00Z",
            "dataset": {
                "range": {"start": "2022-01-01T00:00:00Z", "end": "2025-01-01T00:00:00Z"},
                "path": "data/raw/development.csv",
                "csv_sha256": "a" * 64,
                "generation_id": "generation",
                "candle_count": 4,
            },
        },
        "b" * 64,
    )
    contexts: list[list[Candle]] = []

    def fake_backfill(
        _: RecommendationEngine, candles: list[Candle], __: set[datetime]
    ) -> list[Recommendation]:
        contexts.append(candles)
        fold = next(
            item for item in walk_forward.FOLDS if candles[-1].close_time == item.validation_end
        )
        return [
            _recommendation(fold.validation_start, fold.identifier),
            _recommendation(fold.validation_end, f"{fold.identifier}-excluded"),
        ]

    monkeypatch.setattr(experiments, "_verified_experiment_input", lambda _: snapshot)
    monkeypatch.setattr(walk_forward, "backfill_recommendations", fake_backfill)
    monkeypatch.setattr(walk_forward, "evaluate_outcomes", lambda *args: [])
    output = Path("reports/research/walk-forward/result.json")

    report = walk_forward.run_development_walk_forward(
        Path("reports/research/manifests/development.json"),
        "baseline_ema_volume_atr_v1",
        output,
        BotSettings(),
        overwrite=False,
        now=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert all(
        context[-1].close_time <= fold.validation_end
        for context, fold in zip(contexts, walk_forward.FOLDS, strict=True)
    )
    assert [fold["recommendation_count"] for fold in report["folds"]] == [1, 1, 1]
    assert report["selection_decision"]["decision"] == "no_policy_selected"
    assert report["research_role"] == "development"
    assert report["strict_oos_evaluation_history"] is False
    assert report["research_claim_eligible"] is False
    assert (tmp_path / output).is_file()

    with pytest.raises(walk_forward.RecommendationWalkForwardError, match="already exists"):
        walk_forward.run_development_walk_forward(
            Path("reports/research/manifests/development.json"),
            "baseline_ema_volume_atr_v1",
            output,
            BotSettings(),
            overwrite=False,
        )


def test_walk_forward_has_no_execution_or_model_dependencies() -> None:
    source = inspect.getsource(walk_forward)
    for forbidden in (
        "RiskEngine",
        "BinanceBroker",
        "DryRunBroker",
        "OrderRequest",
        "api_key",
        "api_secret",
        "ProbabilityModel",
    ):
        assert forbidden not in source


def test_cli_runs_development_only_walk_forward(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "trading_bot.cli.run_development_walk_forward",
        lambda *args, **kwargs: {
            "protocol_version": walk_forward.PROTOCOL_VERSION,
            "candidate_id": "baseline_ema_volume_atr_v1",
            "research_role": "development",
            "strict_oos_evaluation_history": False,
            "research_claim_eligible": False,
            "selection_decision": {"decision": "no_policy_selected"},
            "safety_locks": {
                "live_trading_enabled": False,
                "broker_used": False,
                "orders_submitted": False,
                "ml_inference_used": False,
            },
        },
    )

    assert (
        main(
            [
                "run-recommendation-walk-forward",
                "--manifest",
                str(tmp_path / "manifest.json"),
                "--candidate",
                "baseline_ema_volume_atr_v1",
                "--output",
                str(tmp_path / "report.json"),
            ]
        )
        == 0
    )
    result = capsys.readouterr().out
    assert "development_walk_forward_v1" in result
    assert '"research_claim_eligible": false' in result
