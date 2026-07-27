from __future__ import annotations

import inspect
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from trading_bot.cli import main
from trading_bot.data.csv_store import write_candles_atomic
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
    RecommendationHistoryStore,
    accuracy_report,
    evaluate_outcomes,
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
        Decimal("100"),
        Decimal("95"),
        Decimal("105"),
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
