from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from trading_bot.cli import main
from trading_bot.data.csv_store import write_candles_atomic
from trading_bot.data.validation import CandleValidationError
from trading_bot.domain.models import (
    Candle,
    Recommendation,
    RecommendationOutcome,
    RecommendationOutcomeStatus,
    RecommendationType,
    Side,
    StrategySignal,
)
from trading_bot.features.pipeline import FEATURE_SCHEMA_VERSION
from trading_bot.recommendations import engine as recommendation_module
from trading_bot.recommendations.engine import (
    HORIZONS,
    RecommendationEngine,
    RecommendationHistoryProvenance,
    RecommendationHistoryStore,
    accuracy_report,
    backfill_recommendations,
    evaluate_outcomes,
    merge_outcomes,
    recommendation_json,
)
from trading_bot.settings import BotSettings

BASE = datetime(2024, 1, 1, tzinfo=UTC)


def candle(index: int, price: Decimal | None = None, closed: bool = True) -> Candle:
    value = price if price is not None else Decimal("100") + Decimal(index)
    return Candle(
        BASE + timedelta(hours=index),
        BASE + timedelta(hours=index + 1),
        "BTC/USDT",
        "1h",
        value,
        value + 2,
        value - 2,
        value,
        Decimal("1000"),
        closed,
    )


def candles(count: int = 210) -> list[Candle]:
    return [candle(index) for index in range(count)]


def causal_backfill_reference(
    engine: RecommendationEngine, items: list[Candle]
) -> list[Recommendation]:
    return [engine.recommend(items[: index + 1]).recommendation for index in range(len(items))]


def real_candidate_candles() -> list[Candle]:
    """Closed data with an actual EMA20/EMA50 cross and volume confirmation."""

    result: list[Candle] = []
    for index in range(270):
        close = (
            Decimal("120")
            - Decimal(index) * Decimal("0.1")
            + (Decimal("0.2") if index % 2 else Decimal("0"))
            if index < 220
            else Decimal("150") + (index - 220) * Decimal("3")
        )
        result.append(
            Candle(
                BASE + timedelta(hours=index),
                BASE + timedelta(hours=index + 1),
                "BTC/USDT",
                "1h",
                close,
                close + Decimal("2"),
                close - Decimal("2"),
                close,
                Decimal("5000") if index >= 220 else Decimal("1000"),
                True,
            )
        )
    return result


def strict_provenance(
    input_path: Path, items: list[Candle], boundary: datetime
) -> RecommendationHistoryProvenance:
    return RecommendationHistoryProvenance(
        True,
        boundary,
        hashlib.sha256(input_path.read_bytes()).hexdigest(),
        items[0].close_time,
        items[-1].close_time,
    )


def candidate_engine(monkeypatch: pytest.MonkeyPatch, **kwargs: object) -> RecommendationEngine:
    engine = RecommendationEngine(BotSettings(), **kwargs)
    monkeypatch.setattr(
        engine.strategy,
        "entry",
        lambda *_: StrategySignal(
            BASE,
            Side.BUY,
            "test_rule_candidate",
            2.0,
            FEATURE_SCHEMA_VERSION,
            "test-signal",
        ),
    )
    return engine


def recommendation(
    identifier: str, kind: RecommendationType = RecommendationType.BUY_BIAS
) -> Recommendation:
    entry = None if kind == RecommendationType.AVOID else Decimal("100")
    invalidation = None if kind == RecommendationType.AVOID else Decimal("95")
    target = None if kind == RecommendationType.AVOID else Decimal("105")
    return Recommendation(
        identifier,
        BASE,
        candle(0).close_time,
        "BTC/USDT",
        "1h",
        HORIZONS,
        kind,
        None,
        0.55,
        None,
        FEATURE_SCHEMA_VERSION,
        "test",
        "validated_closed_contiguous",
        entry,
        invalidation,
        target,
    )


def test_recommendation_requires_closed_contiguous_candles(monkeypatch: pytest.MonkeyPatch) -> None:
    items = candles()
    items[-1] = candle(len(items) - 1, closed=False)
    report = candidate_engine(monkeypatch).recommend(items)
    assert report.recommendation.recommendation == RecommendationType.NEUTRAL
    assert report.recommendation.rule_reason == "invalid_or_gapped_candle_data"

    gapped = candles()
    original = gapped[100]
    gapped[100] = Candle(
        original.open_time + timedelta(hours=1),
        original.close_time + timedelta(hours=1),
        original.symbol,
        original.timeframe,
        original.open,
        original.high,
        original.low,
        original.close,
        original.volume,
        True,
    )
    gap_report = candidate_engine(monkeypatch).recommend(gapped)
    assert gap_report.recommendation.recommendation == RecommendationType.NEUTRAL


def test_recommendation_features_do_not_read_future_candles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = candidate_engine(monkeypatch)
    prefix = candles(210)
    first = engine.recommend(prefix).recommendation
    future_changed = prefix + [candle(210, Decimal("999999")), candle(211, Decimal("1"))]
    second = engine.recommend(future_changed[:210]).recommendation
    assert first.entry_reference == second.entry_reference
    assert first.invalidation_price == second.invalidation_price
    assert first.target_price == second.target_price


def test_model_missing_or_schema_mismatch_returns_neutral(monkeypatch: pytest.MonkeyPatch) -> None:
    class WrongSchemaModel:
        model_version = "test"
        feature_schema_version = "wrong"
        production_eligible = True
        live_trading_enabled = False

        def probability_up(self, values: object) -> float:
            del values
            return 0.9

    missing = candidate_engine(monkeypatch, require_model=True).recommend(candles()).recommendation
    assert missing.recommendation == RecommendationType.NEUTRAL
    assert missing.probability_up is None
    mismatch = (
        candidate_engine(monkeypatch, model=WrongSchemaModel()).recommend(candles()).recommendation
    )
    assert mismatch.recommendation == RecommendationType.NEUTRAL
    assert mismatch.rule_reason == "model_feature_schema_mismatch"


def test_rule_only_candidate_has_no_invented_ml_probability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = candidate_engine(monkeypatch).recommend(candles()).recommendation
    assert result.recommendation == RecommendationType.BUY_BIAS
    assert result.probability_up is None
    assert result.model_version is None
    assert result.confidence == pytest.approx(0.55)


def test_horizon_outcomes_use_after_cost_direction_and_stop_first() -> None:
    items = [candle(0, Decimal("100"))]
    for index in range(1, 25):
        value = Decimal("110") if index in {1, 4, 24} else Decimal("101")
        items.append(candle(index, value))
    outcomes = evaluate_outcomes([recommendation("buy")], items, BotSettings())
    assert [item.horizon for item in outcomes] == list(HORIZONS)
    assert all(item.status == RecommendationOutcomeStatus.RESOLVED for item in outcomes)
    assert all(item.direction_correct is True for item in outcomes)
    assert outcomes[0].target_hit is True
    assert outcomes[0].invalidation_hit is False

    stopped = items.copy()
    initial = stopped[1]
    stopped[1] = Candle(
        initial.open_time,
        initial.close_time,
        initial.symbol,
        initial.timeframe,
        initial.open,
        Decimal("111"),
        Decimal("94"),
        initial.close,
        initial.volume,
        True,
    )
    stop_outcome = evaluate_outcomes([recommendation("stop")], stopped, BotSettings())[0]
    assert stop_outcome.invalidation_hit is True
    assert stop_outcome.target_hit is False


def test_incomplete_future_data_is_not_counted_in_accuracy() -> None:
    outcomes = evaluate_outcomes([recommendation("short")], candles(4), BotSettings())
    assert outcomes[0].status == RecommendationOutcomeStatus.RESOLVED
    assert outcomes[1].status == RecommendationOutcomeStatus.INSUFFICIENT_FUTURE_DATA
    assert outcomes[2].status == RecommendationOutcomeStatus.INSUFFICIENT_FUTURE_DATA
    report = accuracy_report([recommendation("short")], outcomes)
    assert report["horizons"]["4h"]["resolved_recommendations"] == 0
    assert report["horizons"]["4h"]["directional_accuracy"] is None
    assert report["horizons"]["4h"]["inconclusive"] is True


def test_accuracy_metrics_include_buy_avoid_neutral_and_inconclusive() -> None:
    records = [
        recommendation("buy", RecommendationType.BUY_BIAS),
        recommendation("avoid", RecommendationType.AVOID),
        recommendation("neutral", RecommendationType.NEUTRAL),
    ]
    outcomes = [
        RecommendationOutcome(
            "buy",
            "1h",
            BASE,
            Decimal("0.01"),
            True,
            False,
            False,
            RecommendationOutcomeStatus.RESOLVED,
        ),
        RecommendationOutcome(
            "avoid",
            "1h",
            BASE,
            Decimal("-0.01"),
            True,
            False,
            False,
            RecommendationOutcomeStatus.RESOLVED,
        ),
        RecommendationOutcome(
            "neutral",
            "1h",
            BASE,
            Decimal("0.01"),
            None,
            False,
            False,
            RecommendationOutcomeStatus.RESOLVED,
        ),
    ]
    metrics = accuracy_report(records, outcomes)["horizons"]["1h"]
    assert metrics["total_recommendations"] == 3
    assert metrics["resolved_recommendations"] == 3
    assert metrics["coverage"] == 1.0
    assert metrics["directional_accuracy"] == 1.0
    assert metrics["buy_bias_precision"] == 1.0
    assert metrics["avoid_precision"] == 1.0
    assert metrics["neutral_rate"] == pytest.approx(1 / 3)
    assert metrics["brier_score"] is None
    assert metrics["calibration"] is None
    assert metrics["inconclusive"] is True


def test_accuracy_reports_brier_and_calibration_only_for_ml_probabilities() -> None:
    record = replace(recommendation("model"), probability_up=0.75, model_version="offline-model")
    outcome = RecommendationOutcome(
        "model",
        "1h",
        BASE,
        Decimal("0.01"),
        True,
        False,
        False,
        RecommendationOutcomeStatus.RESOLVED,
    )
    metrics = accuracy_report([record], [outcome])["horizons"]["1h"]
    assert metrics["brier_score"] == pytest.approx(0.0625)
    assert metrics["calibration"] == {
        "sample_size": 1,
        "mean_predicted_probability_up": 0.75,
        "observed_up_rate": 1.0,
    }


def test_history_persistence_restores_recommendations_and_outcomes(tmp_path: Path) -> None:
    store = RecommendationHistoryStore(tmp_path / "history.json")
    outcome = RecommendationOutcome(
        "persist",
        "1h",
        BASE,
        Decimal("0.01"),
        True,
        False,
        False,
        RecommendationOutcomeStatus.RESOLVED,
    )
    store.save([recommendation("persist")], [outcome])
    restored_recommendations, restored_outcomes = store.load()
    assert restored_recommendations == [recommendation("persist")]
    assert restored_outcomes == [outcome]
    assert "api_key" not in store.path.read_text(encoding="utf-8").lower()


def test_cli_recommend_and_evaluate_persist_ignored_json(tmp_path: Path) -> None:
    input_path = tmp_path / "btc.csv"
    latest = tmp_path / "latest.json"
    history = tmp_path / "history.json"
    accuracy = tmp_path / "accuracy.json"
    write_candles_atomic(input_path, candles())
    assert (
        main(
            [
                "recommend",
                "--input",
                str(input_path),
                "--output",
                str(latest),
                "--history",
                str(history),
            ]
        )
        == 0
    )
    latest_payload = json.loads(latest.read_text(encoding="utf-8"))
    assert latest_payload["mode"] == "recommendation_only"
    assert latest_payload["safety_locks"]["broker_used"] is False
    assert (
        main(["evaluate-recommendations", "--input", str(history), "--output", str(accuracy)]) == 0
    )
    assert json.loads(accuracy.read_text(encoding="utf-8"))["inconclusive"] is True


def test_recommendation_module_has_no_broker_or_order_dependency() -> None:
    source = inspect.getsource(recommendation_module)
    for forbidden in (
        "execution.broker",
        "RiskEngine",
        "OrderRequest",
        "DryRunBroker",
        "BinanceBroker",
    ):
        assert forbidden not in source


def test_live_mode_remains_locked() -> None:
    with pytest.raises(ValueError, match="live mode"):
        BotSettings(bot_mode="live")


def test_resolved_outcome_is_not_downgraded_by_partial_csv(tmp_path: Path) -> None:
    history = tmp_path / "history.json"
    old = recommendation("old")
    resolved = RecommendationOutcome(
        old.id,
        "1h",
        candle(1).close_time,
        Decimal("0.01"),
        True,
        False,
        False,
        RecommendationOutcomeStatus.RESOLVED,
    )
    RecommendationHistoryStore(history).save([old], [resolved])
    partial = tmp_path / "partial.csv"
    latest = tmp_path / "latest.json"
    write_candles_atomic(partial, [candle(index) for index in range(300, 510)])

    assert (
        main(
            [
                "recommend",
                "--input",
                str(partial),
                "--output",
                str(latest),
                "--history",
                str(history),
            ]
        )
        == 0
    )
    _, outcomes = RecommendationHistoryStore(history).load()
    assert next(item for item in outcomes if item.recommendation_id == old.id) == resolved


def test_outcome_merge_only_promotes_incomplete_outcome() -> None:
    resolved = RecommendationOutcome(
        "stable",
        "1h",
        BASE,
        Decimal("0.01"),
        True,
        False,
        False,
        RecommendationOutcomeStatus.RESOLVED,
    )
    incomplete = RecommendationOutcome(
        "stable",
        "1h",
        None,
        None,
        None,
        None,
        None,
        RecommendationOutcomeStatus.INSUFFICIENT_FUTURE_DATA,
    )
    assert merge_outcomes([resolved], [incomplete]) == [resolved]


def test_optimized_backfill_matches_unmocked_causal_reference() -> None:
    items = real_candidate_candles()
    reference = causal_backfill_reference(
        RecommendationEngine(BotSettings(), now=lambda: BASE), items
    )
    optimized = backfill_recommendations(
        RecommendationEngine(BotSettings(), now=lambda: BASE), items
    )

    assert optimized == reference
    assert any(item.recommendation == RecommendationType.BUY_BIAS for item in optimized)
    reference_outcomes = evaluate_outcomes(reference, items, BotSettings())
    optimized_outcomes = evaluate_outcomes(optimized, items, BotSettings())
    assert optimized_outcomes == reference_outcomes
    assert accuracy_report(optimized, optimized_outcomes) == accuracy_report(
        reference, reference_outcomes
    )


def test_backfill_is_causal_and_rejects_invalid_input(monkeypatch: pytest.MonkeyPatch) -> None:
    original = candles(212)
    changed = original.copy()
    changed[210] = candle(210, Decimal("999999"))
    changed[211] = candle(211, Decimal("10"))
    engine = candidate_engine(monkeypatch, now=lambda: BASE)
    first = backfill_recommendations(engine, original)
    second = backfill_recommendations(engine, changed)
    assert first[209] == second[209]

    open_items = candles()
    open_items[-1] = candle(len(open_items) - 1, closed=False)
    with pytest.raises(CandleValidationError, match="open candle"):
        backfill_recommendations(candidate_engine(monkeypatch), open_items)


def test_avoid_has_no_trade_levels(monkeypatch: pytest.MonkeyPatch) -> None:
    class LowProbabilityModel:
        model_version = "test-model"
        feature_schema_version = FEATURE_SCHEMA_VERSION
        production_eligible = True
        live_trading_enabled = False

        def probability_up(self, values: object) -> float:
            del values
            return 0.1

    result = (
        candidate_engine(monkeypatch, model=LowProbabilityModel())
        .recommend(candles())
        .recommendation
    )
    assert result.recommendation == RecommendationType.AVOID
    assert result.entry_reference is None
    assert result.target_price is None
    assert result.invalidation_price is None
    serialized = recommendation_json(result)
    assert serialized["entry_reference"] is None


def test_locked_oos_history_rejects_incompatible_reruns_without_mutation(tmp_path: Path) -> None:
    input_path = tmp_path / "btc.csv"
    history = tmp_path / "oos.json"
    report = tmp_path / "accuracy.json"
    items = real_candidate_candles()
    boundary = items[220].close_time.astimezone(UTC).isoformat().replace("+00:00", "Z")
    write_candles_atomic(input_path, items)
    command = [
        "backfill-recommendations",
        "--input",
        str(input_path),
        "--output",
        str(history),
        "--evaluation-start",
        boundary,
    ]

    assert main(command) == 0
    first_bytes = history.read_bytes()
    records, _, provenance, legacy = RecommendationHistoryStore(history).load_with_provenance()
    assert not legacy and provenance is not None and provenance.strict_oos
    assert all(item.signal_candle_time >= items[220].close_time for item in records)
    assert (
        main(["backfill-recommendations", "--input", str(input_path), "--output", str(history)])
        == 1
    )
    assert history.read_bytes() == first_bytes
    assert main(command) == 0
    assert history.read_bytes() == first_bytes
    different = items[221].close_time.astimezone(UTC).isoformat().replace("+00:00", "Z")
    assert main(command[:-1] + [different]) == 1
    assert history.read_bytes() == first_bytes
    assert main(["evaluate-recommendations", "--input", str(history), "--output", str(report)]) == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["strict_oos"] is True
    assert payload["history_provenance"]["evaluation_start"] == boundary
    assert payload["history_provenance"]["input_sha256"] is not None
    assert payload["history_provenance"]["input_first_close"] == "2024-01-01T01:00:00Z"
    assert payload["history_provenance"]["input_last_close"] == "2024-01-12T06:00:00Z"


def test_empty_strict_oos_history_lock_rejects_incompatible_reruns_without_mutation(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "btc.csv"
    changed_input = tmp_path / "changed.csv"
    history = tmp_path / "empty_oos.json"
    items = candles()
    boundary = items[200].close_time
    write_candles_atomic(input_path, items)
    RecommendationHistoryStore(history).save([], [], strict_provenance(input_path, items, boundary))
    original = history.read_bytes()

    assert (
        main(["backfill-recommendations", "--input", str(input_path), "--output", str(history)])
        == 1
    )
    assert history.read_bytes() == original
    changed_boundary = items[201].close_time.astimezone(UTC).isoformat().replace("+00:00", "Z")
    assert (
        main(
            [
                "backfill-recommendations",
                "--input",
                str(input_path),
                "--output",
                str(history),
                "--evaluation-start",
                changed_boundary,
            ]
        )
        == 1
    )
    assert history.read_bytes() == original

    changed = items.copy()
    last = changed[-1]
    changed[-1] = Candle(
        last.open_time,
        last.close_time,
        last.symbol,
        last.timeframe,
        last.open,
        last.high,
        last.low,
        last.close + Decimal("1"),
        last.volume,
        last.is_closed,
    )
    write_candles_atomic(changed_input, changed)
    assert (
        main(
            [
                "backfill-recommendations",
                "--input",
                str(changed_input),
                "--output",
                str(history),
                "--evaluation-start",
                boundary.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            ]
        )
        == 1
    )
    assert history.read_bytes() == original


def test_evaluate_rejects_pre_boundary_strict_oos_history_without_writing_report(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "btc.csv"
    history = tmp_path / "oos.json"
    report = tmp_path / "accuracy.json"
    items = candles()
    write_candles_atomic(input_path, items)
    RecommendationHistoryStore(history).save(
        [recommendation("pre-boundary")],
        [],
        strict_provenance(input_path, items, items[200].close_time),
    )
    report.write_text("preserve-this-report", encoding="utf-8")
    original = report.read_bytes()

    assert main(["evaluate-recommendations", "--input", str(history), "--output", str(report)]) == 1
    assert report.read_bytes() == original


def test_evaluate_valid_strict_oos_history_includes_full_provenance(tmp_path: Path) -> None:
    input_path = tmp_path / "btc.csv"
    history = tmp_path / "oos.json"
    report = tmp_path / "accuracy.json"
    items = candles()
    boundary = items[200].close_time
    write_candles_atomic(input_path, items)
    strict = strict_provenance(input_path, items, boundary)
    valid = replace(recommendation("in-boundary"), signal_candle_time=boundary)
    RecommendationHistoryStore(history).save([valid], [], strict)

    assert main(["evaluate-recommendations", "--input", str(history), "--output", str(report)]) == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["strict_oos"] is True
    assert payload["history_provenance"] == {
        "legacy": False,
        "evaluation_start": "2024-01-09T09:00:00Z",
        "input_sha256": strict.input_sha256,
        "input_first_close": "2024-01-01T01:00:00Z",
        "input_last_close": "2024-01-09T18:00:00Z",
    }


def test_legacy_history_cannot_be_adopted_as_strict_oos(tmp_path: Path) -> None:
    input_path = tmp_path / "btc.csv"
    history = tmp_path / "legacy.json"
    items = candles()
    write_candles_atomic(input_path, items)
    history.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "recommendations": [recommendation_json(recommendation("legacy"))],
                "outcomes": [],
            }
        ),
        encoding="utf-8",
    )
    before = history.read_bytes()
    boundary = items[-1].close_time.astimezone(UTC).isoformat().replace("+00:00", "Z")
    assert (
        main(
            [
                "backfill-recommendations",
                "--input",
                str(input_path),
                "--output",
                str(history),
                "--evaluation-start",
                boundary,
            ]
        )
        == 1
    )
    assert history.read_bytes() == before
