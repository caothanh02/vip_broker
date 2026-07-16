from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from trading_bot.data.validation import validate_candles
from trading_bot.domain.models import Candle, OrderRequest, Position, Side, StrategySignal, Trade
from trading_bot.execution.broker import SimulatedBroker
from trading_bot.features.pipeline import build_features
from trading_bot.risk.engine import RiskEngine
from trading_bot.settings import BotSettings
from trading_bot.strategy.ema_volume_atr import EmaVolumeAtrStrategy


@dataclass(slots=True)
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[tuple[datetime, Decimal]] = field(default_factory=list)
    total_fees: Decimal = Decimal("0")
    total_slippage: Decimal = Decimal("0")

    def metrics(self) -> dict[str, float]:
        start, end = self.equity_curve[0][1], self.equity_curve[-1][1]
        pnl = [float(t.pnl) for t in self.trades]
        wins = [x for x in pnl if x > 0]
        losses = [-x for x in pnl if x < 0]
        return {
            "total_return": float(end / start - 1),
            "number_of_trades": float(len(self.trades)),
            "win_rate": len(wins) / len(pnl) if pnl else 0.0,
            "profit_factor": sum(wins) / sum(losses) if losses else 0.0,
            "total_fees": float(self.total_fees),
            "total_modeled_slippage": float(self.total_slippage),
        }


class CandleBacktester:
    """Conservative OHLC simulator; a reachable stop wins an ambiguous same-bar outcome."""

    def __init__(self, settings: BotSettings) -> None:
        self.settings = settings
        self.strategy = EmaVolumeAtrStrategy(settings)
        self.risk = RiskEngine(settings)
        self.broker = SimulatedBroker(
            settings.starting_cash,
            settings.entry_fee_rate,
            settings.exit_fee_rate,
            settings.entry_slippage_rate,
            settings.exit_slippage_rate,
        )

    def run(self, candles: list[Candle]) -> BacktestResult:
        validate_candles(candles)
        result = BacktestResult()
        position: Position | None = None
        pending: StrategySignal | None = None
        cash = self.settings.starting_cash
        features = build_features(candles)
        for i, candle in enumerate(candles):
            opening_equity = cash + (position.quantity * candle.open if position else Decimal("0"))
            self.risk.mark_to_market(candle.open_time, opening_equity)
            if pending is not None:
                # The execution price, stop and size are all derived at the
                # next open.  A signal-close price is never used for sizing.
                decision = self.risk.decide(
                    candle.open_time,
                    cash,
                    cash,
                    candle.open,
                    Decimal(str(pending.atr)),
                    False,
                )
                if decision.accepted:
                    request = OrderRequest(
                        "BTC/USDT",
                        Side.BUY,
                        decision.quantity,
                        pending.signal_id,
                        pending.signal_id,
                    )
                    fill = self.broker.place_order(request, candle.open, candle.open_time)
                    entry_cost = fill.price * fill.quantity + fill.fee
                    if entry_cost > cash:
                        raise RuntimeError("risk-approved entry exceeds cash")
                    cash -= entry_cost
                    position = Position(
                        "BTC/USDT",
                        fill.quantity,
                        fill.price,
                        decision.stop_price or Decimal("0"),
                        fill.price,
                        candle.open_time,
                        fill.fee,
                    )
                    result.total_fees += fill.fee
                    result.total_slippage += fill.slippage
                pending = None
            if position:
                # Conservative OHLC rule: a reachable stop is filled before crossover exits.
                stopped = candle.low <= position.stop_price
                reason = "stop_loss_or_trailing" if stopped else None
                if reason is None and self.strategy.exit_crossover(features.iloc[: i + 1]):
                    reason = "ema_cross_down"
                if reason:
                    # A stop crossed by the opening gap cannot receive the
                    # better stop price.  The broker then applies exit slippage.
                    exit_price = (
                        candle.open
                        if stopped and candle.open <= position.stop_price
                        else (position.stop_price if stopped else candle.close)
                    )
                    fill = self.broker.place_order(
                        OrderRequest(
                            "BTC/USDT",
                            Side.SELL,
                            position.quantity,
                            f"exit-{position.opened_at.isoformat()}",
                            "exit",
                        ),
                        exit_price,
                        candle.close_time,
                    )
                    proceeds = fill.price * fill.quantity - fill.fee
                    cash += proceeds
                    pnl = proceeds - (position.entry_price * position.quantity + position.entry_fee)
                    trade = Trade(
                        "BTC/USDT",
                        position.quantity,
                        position.entry_price,
                        fill.price,
                        position.opened_at,
                        candle.close_time,
                        pnl,
                        position.entry_fee + fill.fee,
                        reason,
                    )
                    result.trades.append(trade)
                    result.total_fees += fill.fee
                    result.total_slippage += fill.slippage
                    self.risk.record_closed_trade(candle.close_time, pnl, cash)
                    position = None
                else:
                    # Current-candle high/low and ATR are available only after
                    # all intrabar events above have been resolved.  This stop
                    # applies from the following candle onward.
                    atr = features.iloc[i].atr14
                    if atr == atr and atr > 0:
                        position.highest_price = max(position.highest_price, candle.high)
                        position.stop_price = max(
                            position.stop_price,
                            position.highest_price
                            - Decimal(str(self.settings.trailing_atr_multiple * float(atr))),
                        )
            equity = cash + (position.quantity * candle.close if position else Decimal("0"))
            result.equity_curve.append((candle.close_time, equity))
            self.risk.mark_to_market(candle.close_time, equity)
            if i < len(candles) - 1 and position is None:
                entry_signal = self.strategy.entry(
                    features.iloc[: i + 1],
                    False,
                    bool(
                        self.risk.state.cooldown_until
                        and candle.close_time < self.risk.state.cooldown_until
                    ),
                    self.risk.state.circuit_open,
                )
                if entry_signal:
                    pending = entry_signal
        return result
