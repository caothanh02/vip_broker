from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from trading_bot.data.validation import CandleValidationError, validate_candles
from trading_bot.domain.models import Candle
from trading_bot.features.pipeline import build_features
from trading_bot.risk.engine import RiskEngine
from trading_bot.settings import BotSettings


def candles(n: int = 240) -> list[Candle]:
    t = datetime(2024, 1, 1, tzinfo=UTC)
    return [
        Candle(
            t + timedelta(hours=i),
            t + timedelta(hours=i + 1),
            "BTC/USDT",
            "1h",
            Decimal(100 + i),
            Decimal(102 + i),
            Decimal(99 + i),
            Decimal(101 + i),
            Decimal(1000),
            True,
        )
        for i in range(n)
    ]


def test_feature_pipeline_uses_only_history() -> None:
    first = build_features(candles())
    changed = candles()
    changed[-1] = Candle(
        changed[-1].open_time,
        changed[-1].close_time,
        "BTC/USDT",
        "1h",
        Decimal(999),
        Decimal(999),
        Decimal(999),
        Decimal(999),
        Decimal(999),
        True,
    )
    second = build_features(changed)
    assert first.iloc[-2].equals(second.iloc[-2])


def test_validator_rejects_gap() -> None:
    data = candles(2)
    old = data[1]
    data[1] = Candle(
        old.open_time + timedelta(hours=1),
        old.close_time + timedelta(hours=1),
        old.symbol,
        old.timeframe,
        old.open,
        old.high,
        old.low,
        old.close,
        old.volume,
        old.is_closed,
    )
    with pytest.raises(CandleValidationError):
        validate_candles(data)


def test_risk_caps_exposure() -> None:
    d = RiskEngine(BotSettings()).decide(
        datetime.now(UTC),
        Decimal("10000"),
        Decimal("10000"),
        Decimal("50000"),
        Decimal("100"),
        False,
    )
    assert d.accepted and d.quantity * Decimal("50000") <= Decimal("3000")


def test_live_mode_is_locked() -> None:
    with pytest.raises(ValueError):
        BotSettings(bot_mode="live")
