from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from math import inf, sqrt
from statistics import stdev

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
    exposure_candles: int = 0
    buy_and_hold_return: float = 0.0

    def metrics(self) -> dict[str, float]:
        start, end = self.equity_curve[0][1], self.equity_curve[-1][1]
        pnl = [float(t.pnl) for t in self.trades]
        wins = [x for x in pnl if x > 0]
        losses = [-x for x in pnl if x < 0]
        returns = [
            float(current / previous - 1)
            for (_, previous), (_, current) in zip(
                self.equity_curve, self.equity_curve[1:], strict=False
            )
            if previous > 0
        ]
        peak = float(start)
        max_drawdown = 0.0
        for _, equity in self.equity_curve:
            value = float(equity)
            peak = max(peak, value)
            if peak > 0:
                max_drawdown = max(max_drawdown, (peak - value) / peak)
        average_return = sum(returns) / len(returns) if returns else 0.0
        return_std = stdev(returns) if len(returns) > 1 else 0.0
        downside = [value for value in returns if value < 0]
        downside_std = (
            sqrt(sum(value * value for value in downside) / len(downside)) if downside else 0.0
        )
        annualization = sqrt(24 * 365)
        sharpe = average_return / return_std * annualization if return_std > 0 else 0.0
        sortino = (
            average_return / downside_std * annualization
            if downside_std > 0
            else (inf if average_return > 0 else 0.0)
        )
        max_consecutive_wins, max_consecutive_losses = 0, 0
        consecutive_wins, consecutive_losses = 0, 0
        for trade_pnl in pnl:
            if trade_pnl > 0:
                consecutive_wins += 1
                consecutive_losses = 0
            elif trade_pnl < 0:
                consecutive_losses += 1
                consecutive_wins = 0
            else:
                consecutive_wins = consecutive_losses = 0
            max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        return {
            "total_return": float(end / start - 1),
            "number_of_trades": float(len(self.trades)),
            "win_rate": len(wins) / len(pnl) if pnl else 0.0,
            "profit_factor": sum(wins) / sum(losses) if losses else (inf if wins else 0.0),
            "expectancy": sum(pnl) / len(pnl) if pnl else 0.0,
            "average_win": sum(wins) / len(wins) if wins else 0.0,
            "average_loss": -sum(losses) / len(losses) if losses else 0.0,
            "max_drawdown": max_drawdown,
            "sharpe": sharpe,
            "sortino": sortino,
            "exposure_time": self.exposure_candles / len(self.equity_curve),
            "max_consecutive_wins": float(max_consecutive_wins),
            "max_consecutive_losses": float(max_consecutive_losses),
            "buy_and_hold_return": self.buy_and_hold_return,
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

    def _close_position(
        self,
        result: BacktestResult,
        position: Position,
        price: Decimal,
        timestamp: datetime,
        reason: str,
        cash: Decimal,
        risk_timestamp: datetime | None = None,
    ) -> tuple[Decimal, Trade]:
        fill = self.broker.place_order(
            OrderRequest(
                "BTC/USDT",
                Side.SELL,
                position.quantity,
                f"exit-{position.opened_at.isoformat()}",
                "exit",
            ),
            price,
            timestamp,
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
            timestamp,
            pnl,
            position.entry_fee + fill.fee,
            reason,
        )
        result.trades.append(trade)
        result.total_fees += fill.fee
        result.total_slippage += fill.slippage
        self.risk.record_closed_trade(risk_timestamp or timestamp, pnl, cash)
        return cash, trade

    def run(self, candles: list[Candle]) -> BacktestResult:
        validate_candles(candles)
        result = BacktestResult()
        position: Position | None = None
        pending: StrategySignal | None = None
        cash = self.settings.starting_cash
        features = build_features(candles)
        result.buy_and_hold_return = float(candles[-1].close / candles[0].open - 1)
        pending_exit: str | None = None
        for i, candle in enumerate(candles):
            opening_equity = cash + (position.quantity * candle.open if position else Decimal("0"))
            exit_to_fill = pending_exit
            pending_exit = None
            self.risk.mark_to_market(candle.open_time, opening_equity)
            if exit_to_fill is not None and position is not None:
                cash, _ = self._close_position(
                    result,
                    position,
                    candle.open,
                    candle.open_time,
                    exit_to_fill,
                    cash,
                )
                position = None
            elif position is not None and self.risk.state.circuit_open:
                # A drawdown discovered from the known opening price is an
                # emergency: liquidate now, before this candle's intrabar path.
                cash, _ = self._close_position(
                    result,
                    position,
                    candle.open,
                    candle.open_time,
                    "circuit_breaker_emergency_liquidation",
                    cash,
                )
                position = None
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
            exposed_during_candle = position is not None
            if position:
                # Conservative OHLC rule: a reachable stop is filled before crossover exits.
                stopped = candle.low <= position.stop_price
                if stopped:
                    # A stop crossed by the opening gap cannot receive the
                    # better stop price.  The broker then applies exit slippage.
                    gap_through_stop = candle.open <= position.stop_price
                    exit_price = candle.open if gap_through_stop else position.stop_price
                    # Intrabar timing is unknown, so it is reported at the
                    # candle close; an opening gap is known at candle open.
                    exit_time = candle.open_time if gap_through_stop else candle.close_time
                    cash, _ = self._close_position(
                        result,
                        position,
                        exit_price,
                        exit_time,
                        "stop_loss_or_trailing",
                        cash,
                        candle.open_time,
                    )
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
                    if self.strategy.exit_crossover(features.iloc[: i + 1]):
                        pending_exit = "ema_cross_down"
            equity = cash + (position.quantity * candle.close if position else Decimal("0"))
            result.equity_curve.append((candle.close_time, equity))
            self.risk.mark_to_market(candle.close_time, equity)
            if position is not None and self.risk.state.circuit_open:
                pending_exit = "circuit_breaker_emergency_liquidation"
            if exposed_during_candle:
                result.exposure_candles += 1
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
