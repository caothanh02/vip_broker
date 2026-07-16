"""Known quantitative-safety defects found by the independent review.

These strict xfails are executable reproductions.  They must be converted to
ordinary passing tests when the corresponding production fixes land.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pandas as pd
import pytest

from trading_bot.backtest.engine import CandleBacktester
from trading_bot.domain.models import Candle, Side, StrategySignal
from trading_bot.risk.engine import RiskEngine, RiskState
from trading_bot.settings import BotSettings


def candle(hour: int, open_: str, low: str, close: str) -> Candle:
    opened = datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=hour)
    return Candle(
        open_time=opened,
        close_time=opened + timedelta(hours=1),
        symbol="BTC/USDT",
        timeframe="1h",
        open=Decimal(open_),
        high=max(Decimal(open_), Decimal(close)),
        low=Decimal(low),
        close=Decimal(close),
        volume=Decimal("1000"),
        is_closed=True,
    )


@pytest.mark.xfail(
    strict=True, reason="backtester skips candle validation, enabling time-order leakage"
)
def test_backtester_rejects_unordered_candles_before_features() -> None:
    candles = [candle(1, "100", "99", "100"), candle(0, "100", "99", "100")]
    with pytest.raises(ValueError, match="gap|interval|order"):
        CandleBacktester(BotSettings()).run(candles)


@pytest.mark.xfail(
    strict=True,
    reason="initial stop is based on signal close, not the simulated next-open entry price",
)
def test_gap_between_signal_and_entry_cannot_exceed_risk_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backtester = CandleBacktester(BotSettings())
    signal = StrategySignal(datetime(2024, 1, 1, tzinfo=UTC), Side.BUY, "test", 10.0, "1.0.0")
    monkeypatch.setattr(
        backtester.strategy,
        "entry",
        lambda frame, has_position, cooldown, circuit_open: signal if len(frame) == 1 else None,
    )
    monkeypatch.setattr(backtester.strategy, "exit_crossover", lambda frame: False)

    result = backtester.run(
        [
            candle(0, "100", "95", "100"),
            candle(1, "200", "190", "200"),
            candle(2, "100", "70", "100"),
        ]
    )

    assert len(result.trades) == 1
    maximum_loss = Decimal("10000") * Decimal("0.005")
    assert -result.trades[0].pnl <= maximum_loss


@pytest.mark.xfail(
    strict=True,
    reason="trailing stop uses the just-closed candle's ATR before that candle's low is evaluated",
)
def test_trailing_stop_does_not_use_same_candle_atr_for_intrabar_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backtester = CandleBacktester(BotSettings())
    signal = StrategySignal(datetime(2024, 1, 1, tzinfo=UTC), Side.BUY, "test", 10.0, "1.0.0")
    monkeypatch.setattr(
        backtester.strategy,
        "entry",
        lambda frame, has_position, cooldown, circuit_open: signal if len(frame) == 1 else None,
    )
    monkeypatch.setattr(backtester.strategy, "exit_crossover", lambda frame: False)
    monkeypatch.setattr(
        "trading_bot.backtest.engine.build_features",
        lambda candles: pd.DataFrame({"atr14": [10.0, 1.0]}),
    )

    result = backtester.run([candle(0, "100", "95", "100"), candle(1, "100", "90", "100")])

    assert not result.trades


@pytest.mark.xfail(
    strict=True,
    reason="a gap through a stop is filled at the stop price rather than the worse opening price",
)
def test_gap_through_stop_is_not_filled_at_a_better_than_open_price(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backtester = CandleBacktester(BotSettings())
    signal = StrategySignal(datetime(2024, 1, 1, tzinfo=UTC), Side.BUY, "test", 10.0, "1.0.0")
    monkeypatch.setattr(
        backtester.strategy,
        "entry",
        lambda frame, has_position, cooldown, circuit_open: signal if len(frame) == 1 else None,
    )
    monkeypatch.setattr(backtester.strategy, "exit_crossover", lambda frame: False)

    result = backtester.run(
        [
            candle(0, "100", "95", "100"),
            candle(1, "100", "95", "100"),
            candle(2, "70", "60", "70"),
        ]
    )

    assert len(result.trades) == 1
    assert result.trades[0].exit_price <= Decimal("70")


@pytest.mark.xfail(strict=True, reason="daily loss state is never reset on a new UTC day")
def test_daily_loss_limit_resets_at_next_utc_day() -> None:
    settings = BotSettings()
    engine = RiskEngine(
        settings,
        RiskState(
            peak_equity=Decimal("10000"),
            day_start_equity=Decimal("10000"),
            daily_pnl=Decimal("-300"),
        ),
    )
    decision = engine.decide(
        datetime(2024, 1, 2, tzinfo=UTC),
        Decimal("10000"),
        Decimal("10000"),
        Decimal("50000"),
        Decimal("100"),
        False,
    )
    assert decision.accepted


@pytest.mark.xfail(strict=True, reason="risk sizing omits entry/exit fees and modeled slippage")
def test_risk_sizing_reserves_costs_in_addition_to_stop_distance() -> None:
    settings = BotSettings(max_exposure=Decimal("1"))
    decision = RiskEngine(settings).decide(
        datetime(2024, 1, 1, tzinfo=UTC),
        Decimal("10000"),
        Decimal("10000"),
        Decimal("100"),
        Decimal("10"),
        False,
    )
    assert decision.accepted and decision.stop_price is not None
    entry = Decimal("100") * (Decimal("1") + settings.entry_slippage_rate)
    stop_fill = decision.stop_price * (Decimal("1") - settings.exit_slippage_rate)
    realized_loss = decision.quantity * (entry - stop_fill)
    realized_loss += decision.quantity * entry * settings.entry_fee_rate
    realized_loss += decision.quantity * stop_fill * settings.exit_fee_rate
    assert realized_loss <= Decimal("10000") * settings.risk_per_trade
