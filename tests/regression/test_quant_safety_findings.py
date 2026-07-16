"""Regression coverage for quantitative-safety invariants."""

from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from math import isinf

import pandas as pd
import pytest

from trading_bot.backtest.engine import BacktestResult, CandleBacktester
from trading_bot.data.validation import validate_candles
from trading_bot.domain.models import Candle, Side, StrategySignal, Trade
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


def test_backtester_rejects_unordered_candles_before_features() -> None:
    candles = [candle(1, "100", "99", "100"), candle(0, "100", "99", "100")]
    with pytest.raises(ValueError, match="gap|interval|order"):
        CandleBacktester(BotSettings()).run(candles)


def test_entry_gap_uses_next_open_for_stop_and_planned_risk_budget(
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
            candle(2, "200", "170", "200"),
        ]
    )

    assert len(result.trades) == 1
    maximum_loss = Decimal("10000") * Decimal("0.005")
    assert -result.trades[0].pnl <= maximum_loss


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
    assert result.trades[0].exit_time == datetime(2024, 1, 1, 2, tzinfo=UTC)


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


def _signal() -> StrategySignal:
    return StrategySignal(datetime(2024, 1, 1, tzinfo=UTC), Side.BUY, "test", 10.0, "1.0.0")


def _always_signal_once(backtester: CandleBacktester, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        backtester.strategy,
        "entry",
        lambda frame, has_position, cooldown, circuit_open: _signal() if len(frame) == 1 else None,
    )
    monkeypatch.setattr(backtester.strategy, "exit_crossover", lambda frame: False)


def test_entry_gap_down_recalculates_stop_from_next_open(monkeypatch: pytest.MonkeyPatch) -> None:
    backtester = CandleBacktester(BotSettings())
    _always_signal_once(backtester, monkeypatch)

    result = backtester.run(
        [
            candle(0, "200", "195", "200"),
            candle(1, "100", "90", "100"),
            candle(2, "100", "70", "100"),
        ]
    )

    assert len(result.trades) == 1
    assert result.trades[0].entry_price == Decimal("100.0500")
    assert result.trades[0].exit_price == Decimal("80.00997500")


def test_same_bar_stop_uses_preexisting_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    backtester = CandleBacktester(BotSettings())
    _always_signal_once(backtester, monkeypatch)
    monkeypatch.setattr(
        "trading_bot.backtest.engine.build_features",
        lambda candles: pd.DataFrame({"atr14": [10.0, 1.0]}),
    )

    result = backtester.run([candle(0, "100", "95", "100"), candle(1, "100", "70", "100")])

    assert len(result.trades) == 1
    assert result.trades[0].exit_price == Decimal("80.00997500")
    assert result.trades[0].exit_time == datetime(2024, 1, 1, 2, tzinfo=UTC)


def test_final_open_position_is_marked_to_market(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = BotSettings()
    backtester = CandleBacktester(settings)
    _always_signal_once(backtester, monkeypatch)

    result = backtester.run([candle(0, "100", "95", "100"), candle(1, "100", "90", "110")])

    decision = RiskEngine(settings).decide(
        datetime(2024, 1, 1, 1, tzinfo=UTC),
        settings.starting_cash,
        settings.starting_cash,
        Decimal("100"),
        Decimal("10"),
        False,
    )
    assert decision.accepted
    entry = Decimal("100") * (1 + settings.entry_slippage_rate)
    entry_fee = entry * decision.quantity * settings.entry_fee_rate
    expected_equity = settings.starting_cash - entry * decision.quantity - entry_fee
    expected_equity += decision.quantity * Decimal("110")
    assert not result.trades
    assert result.equity_curve[-1][1] == expected_equity
    assert backtester.risk.state.peak_equity == expected_equity


def test_risk_caps_cash_and_exposure_after_entry_costs() -> None:
    settings = BotSettings(
        starting_cash=Decimal("1000"),
        risk_per_trade=Decimal("1"),
        max_exposure=Decimal("1"),
    )
    decision = RiskEngine(settings).decide(
        datetime(2024, 1, 1, tzinfo=UTC),
        Decimal("1000"),
        Decimal("1000"),
        Decimal("100"),
        Decimal("1"),
        False,
    )
    planned_entry = Decimal("100") * (1 + settings.entry_slippage_rate)
    assert decision.accepted
    assert decision.quantity * planned_entry * (1 + settings.entry_fee_rate) <= Decimal("1000")
    assert decision.quantity * planned_entry <= Decimal("1000")


def test_backtester_rejects_empty_and_open_candle_input() -> None:
    backtester = CandleBacktester(BotSettings())
    with pytest.raises(ValueError, match="no candles"):
        backtester.run([])
    open_candle = candle(0, "100", "99", "100")
    invalid = Candle(
        open_candle.open_time,
        open_candle.close_time,
        open_candle.symbol,
        open_candle.timeframe,
        open_candle.open,
        open_candle.high,
        open_candle.low,
        open_candle.close,
        open_candle.volume,
        False,
    )
    with pytest.raises(ValueError, match="open candle"):
        backtester.run([invalid])


def test_ema_exit_fills_at_next_candle_open(monkeypatch: pytest.MonkeyPatch) -> None:
    backtester = CandleBacktester(BotSettings())
    _always_signal_once(backtester, monkeypatch)
    monkeypatch.setattr(backtester.strategy, "exit_crossover", lambda frame: len(frame) == 2)

    result = backtester.run(
        [
            candle(0, "100", "95", "100"),
            candle(1, "100", "95", "100"),
            candle(2, "120", "110", "120"),
        ]
    )

    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == "ema_cross_down"
    assert result.trades[0].exit_time == datetime(2024, 1, 1, 2, tzinfo=UTC)
    assert result.trades[0].exit_price == Decimal("119.9400")


def test_circuit_breaker_liquidates_at_next_open(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = BotSettings(
        risk_per_trade=Decimal("1"),
        max_exposure=Decimal("1"),
        max_drawdown=Decimal("0.10"),
    )
    backtester = CandleBacktester(settings)
    _always_signal_once(backtester, monkeypatch)

    result = backtester.run(
        [
            candle(0, "100", "95", "100"),
            candle(1, "100", "85", "85"),
            candle(2, "80", "70", "80"),
        ]
    )

    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == "circuit_breaker_emergency_liquidation"
    assert result.trades[0].exit_time == datetime(2024, 1, 1, 2, tzinfo=UTC)
    assert result.trades[0].exit_price == Decimal("79.9600")


def test_unrealized_daily_equity_loss_opens_daily_breaker() -> None:
    settings = BotSettings(max_daily_loss=Decimal("0.03"))
    engine = RiskEngine(settings)
    now = datetime(2024, 1, 1, tzinfo=UTC)
    engine.mark_to_market(now, Decimal("10000"))
    engine.mark_to_market(now + timedelta(hours=1), Decimal("9600"))

    decision = engine.decide(
        now + timedelta(hours=1),
        Decimal("9600"),
        Decimal("9600"),
        Decimal("100"),
        Decimal("10"),
        False,
    )
    assert engine.state.daily_pnl == Decimal("0")
    assert not decision.accepted
    assert decision.reason == "daily_loss_limit"


def test_profit_factor_without_losses_is_infinite() -> None:
    started = datetime(2024, 1, 1, tzinfo=UTC)
    result = BacktestResult(
        trades=[
            Trade(
                "BTC/USDT",
                Decimal("1"),
                Decimal("100"),
                Decimal("110"),
                started,
                started + timedelta(hours=1),
                Decimal("10"),
                Decimal("0"),
                "test",
            )
        ],
        equity_curve=[
            (started, Decimal("10000")),
            (started + timedelta(hours=1), Decimal("10010")),
        ],
    )

    metrics = result.metrics()
    assert isinf(metrics["profit_factor"])
    assert {
        "max_drawdown",
        "expectancy",
        "average_win",
        "average_loss",
        "sharpe",
        "sortino",
        "exposure_time",
        "max_consecutive_wins",
        "max_consecutive_losses",
        "buy_and_hold_return",
    } <= metrics.keys()


@pytest.mark.parametrize(
    "opened",
    [datetime(2024, 1, 1), datetime(2024, 1, 1, tzinfo=timezone(timedelta(hours=7)))],
)
def test_validation_rejects_naive_and_non_utc_timestamps(opened: datetime) -> None:
    invalid = Candle(
        opened,
        opened + timedelta(hours=1),
        "BTC/USDT",
        "1h",
        Decimal("100"),
        Decimal("101"),
        Decimal("99"),
        Decimal("100"),
        Decimal("1000"),
        True,
    )
    with pytest.raises(ValueError, match="timestamp"):
        validate_candles([invalid])
