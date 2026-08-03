from __future__ import annotations

import inspect
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pandas as pd
import pytest

from trading_bot.domain.models import Candle, RecommendationType
from trading_bot.recommendations import experiments
from trading_bot.recommendations.candidate_rules import (
    BASELINE_CANDIDATE_ID,
    TREND_PULLBACK_CANDIDATE_ID,
    candidate_protocol,
    is_trend_pullback_ema_atr_candidate,
)
from trading_bot.recommendations.engine import (
    RecommendationEngine,
    RecommendationError,
    backfill_recommendations,
)
from trading_bot.settings import BotSettings


def _feature_rows(**current_overrides: object) -> tuple[pd.Series, pd.Series]:
    previous = pd.Series(
        {
            "is_closed": True,
            "close": 105.0,
            "ema20": 106.0,
            "ema50": 100.0,
            "ema200": 90.0,
            "volume": 1000.0,
            "volume_sma20": 1000.0,
            "atr14": 2.0,
        }
    )
    current = pd.Series(
        {
            "is_closed": True,
            "close": 107.0,
            "ema20": 106.0,
            "ema50": 100.0,
            "ema200": 90.0,
            "volume": 1200.0,
            "volume_sma20": 1000.0,
            "atr14": 2.0,
        }
    )
    current.update(current_overrides)
    return current, previous


def _candle(index: int, close: Decimal, volume: Decimal = Decimal("1000")) -> Candle:
    opened = datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=index)
    return Candle(
        opened,
        opened + timedelta(hours=1),
        "BTC/USDT",
        "1h",
        close,
        close + Decimal("1"),
        close - Decimal("1"),
        close,
        volume,
        True,
    )


def _trend_pullback_candles() -> list[Candle]:
    candles = [
        _candle(index, Decimal("100") + Decimal(index) / Decimal("10")) for index in range(200)
    ]
    candles.append(_candle(200, Decimal("118")))
    candles.append(_candle(201, Decimal("121"), Decimal("1300")))
    candles.extend(
        _candle(index, Decimal("121") + Decimal(index - 201) / Decimal("10"))
        for index in range(202, 206)
    )
    return candles


def test_trend_pullback_predicate_requires_exact_closed_trend_reclaim_and_volume() -> None:
    current, previous = _feature_rows()

    assert is_trend_pullback_ema_atr_candidate(current, previous, BotSettings()) is True

    for changes in (
        {"close": 90.0},  # not above EMA200/EMA20
        {"ema20": 99.0},  # not above EMA50
        {"volume": 1199.0},
        {"atr14": 0.0},
        {"is_closed": False},
    ):
        candidate, preceding = _feature_rows(**changes)
        assert is_trend_pullback_ema_atr_candidate(candidate, preceding, BotSettings()) is False

    candidate, preceding = _feature_rows()
    preceding["close"] = preceding["ema20"] + 1.0
    assert is_trend_pullback_ema_atr_candidate(candidate, preceding, BotSettings()) is False


def test_trend_pullback_engine_is_causal_rule_only_and_baseline_is_unchanged() -> None:
    candles = _trend_pullback_candles()
    trend = RecommendationEngine(BotSettings(), candidate_id=TREND_PULLBACK_CANDIDATE_ID)
    recommendations = backfill_recommendations(trend, candles)
    signal_index = next(
        index
        for index, item in enumerate(recommendations)
        if item.recommendation == RecommendationType.BUY_BIAS
    )
    assert (
        recommendations[signal_index].rule_reason
        == "trend_pullback_ema_atr_rule_candidate_rule_only"
    )
    assert all(
        item.recommendation in {RecommendationType.BUY_BIAS, RecommendationType.NEUTRAL}
        for item in recommendations
    )

    mutated = list(candles)
    for index in range(signal_index + 1, len(mutated)):
        item = mutated[index]
        mutated[index] = Candle(
            item.open_time,
            item.close_time,
            item.symbol,
            item.timeframe,
            item.open + Decimal("50"),
            item.high + Decimal("50"),
            item.low + Decimal("50"),
            item.close + Decimal("50"),
            item.volume * Decimal("2"),
            True,
        )
    assert backfill_recommendations(trend, mutated)[signal_index] == recommendations[signal_index]

    default = backfill_recommendations(RecommendationEngine(BotSettings()), candles)
    explicit = backfill_recommendations(
        RecommendationEngine(BotSettings(), candidate_id=BASELINE_CANDIDATE_ID), candles
    )
    assert default == explicit


def test_trend_pullback_rejects_open_and_unverified_gap_input() -> None:
    candles = _trend_pullback_candles()
    engine = RecommendationEngine(BotSettings(), candidate_id=TREND_PULLBACK_CANDIDATE_ID)
    opened = candles[-1]
    open_candle = Candle(
        opened.open_time,
        opened.close_time,
        opened.symbol,
        opened.timeframe,
        opened.open,
        opened.high,
        opened.low,
        opened.close,
        opened.volume,
        False,
    )
    assert (
        engine.recommend([*candles[:-1], open_candle]).recommendation.recommendation
        == RecommendationType.NEUTRAL
    )
    assert (
        engine.recommend([*candles[:201], *candles[202:]]).recommendation.recommendation
        == RecommendationType.NEUTRAL
    )


@pytest.mark.parametrize(
    "field",
    [
        "entry_fee_rate",
        "exit_fee_rate",
        "entry_slippage_rate",
        "exit_slippage_rate",
    ],
)
def test_trend_pullback_contract_locks_each_cost_rate(field: str) -> None:
    with pytest.raises(experiments.RecommendationExperimentError, match="cost rate"):
        experiments._validate_candidate_settings(
            TREND_PULLBACK_CANDIDATE_ID, BotSettings(**{field: Decimal("0.002")})
        )
    assert experiments._CANDIDATES[TREND_PULLBACK_CANDIDATE_ID]["cost_model"] == {
        "entry_fee_rate": "0.001",
        "exit_fee_rate": "0.001",
        "entry_slippage_rate": "0.0005",
        "exit_slippage_rate": "0.0005",
    }


def test_unknown_candidate_and_rule_module_execution_dependencies_fail_closed() -> None:
    with pytest.raises(RecommendationError, match="unknown"):
        RecommendationEngine(BotSettings(), candidate_id="unknown")
    assert candidate_protocol(BASELINE_CANDIDATE_ID) == "development_walk_forward_v1"
    assert candidate_protocol(TREND_PULLBACK_CANDIDATE_ID) == "development_walk_forward_v2"
    with pytest.raises(ValueError, match="unknown"):
        candidate_protocol("unknown")

    source = inspect.getsource(
        __import__("trading_bot.recommendations.candidate_rules", fromlist=["_"])
    )
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
