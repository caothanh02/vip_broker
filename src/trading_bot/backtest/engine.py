from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from trading_bot.domain.models import Candle, OrderRequest, Position, Side, Trade
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
        result = BacktestResult()
        position: Position | None = None
        pending = None
        cash = self.settings.starting_cash
        features = build_features(candles)
        for i, candle in enumerate(candles):
            if not candle.is_closed:
                continue
            if pending is not None:
                decision, signal = pending
                request = OrderRequest(
                    "BTC/USDT", Side.BUY, decision.quantity, signal.signal_id, signal.signal_id
                )
                fill = self.broker.place_order(request, candle.open, candle.open_time)
                cash -= fill.price * fill.quantity + fill.fee
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
                atr = features.iloc[i].atr14
                if atr == atr and atr > 0:
                    position.stop_price = max(
                        position.stop_price,
                        position.highest_price
                        - Decimal(str(self.settings.trailing_atr_multiple * float(atr))),
                    )
                position.highest_price = max(position.highest_price, candle.high)
                # Conservative OHLC rule: a reachable stop is filled before crossover exits.
                reason = (
                    "stop_loss_or_trailing"
                    if candle.low <= position.stop_price
                    else (
                        "ema_cross_down"
                        if self.strategy.exit_crossover(features.iloc[: i + 1])
                        else None
                    )
                )
                if reason:
                    fill = self.broker.place_order(
                        OrderRequest(
                            "BTC/USDT",
                            Side.SELL,
                            position.quantity,
                            f"exit-{position.opened_at.isoformat()}",
                            "exit",
                        ),
                        position.stop_price if reason.startswith("stop") else candle.close,
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
            equity = cash + (position.quantity * candle.close if position else Decimal("0"))
            result.equity_curve.append((candle.close_time, equity))
            if i < len(candles) - 1 and position is None:
                entry_signal = self.strategy.entry(
                    features.iloc[: i + 1], False, False, self.risk.state.circuit_open
                )
                if entry_signal:
                    decision = self.risk.decide(
                        candle.close_time,
                        equity,
                        cash,
                        candle.close,
                        Decimal(str(entry_signal.atr)),
                        False,
                    )
                    if decision.accepted:
                        pending = (decision, entry_signal)
        return result
