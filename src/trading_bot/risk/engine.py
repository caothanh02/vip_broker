from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import ROUND_DOWN, Decimal

from trading_bot.domain.models import RiskDecision
from trading_bot.settings import BotSettings


@dataclass(slots=True)
class RiskState:
    peak_equity: Decimal
    day_start_equity: Decimal
    daily_pnl: Decimal = Decimal("0")
    consecutive_losses: int = 0
    cooldown_until: datetime | None = None
    circuit_open: bool = False


class RiskEngine:
    def __init__(self, settings: BotSettings, state: RiskState | None = None) -> None:
        self.settings, self.state = (
            settings,
            state or RiskState(settings.starting_cash, settings.starting_cash),
        )

    def decide(
        self,
        now: datetime,
        equity: Decimal,
        cash: Decimal,
        entry: Decimal,
        atr: Decimal,
        has_position: bool,
        ml_accepted: bool = True,
        healthy: bool = True,
    ) -> RiskDecision:
        if has_position:
            return RiskDecision(False, "position_already_open")
        if not healthy or self.state.circuit_open:
            return RiskDecision(False, "circuit_breaker_or_unhealthy")
        if not ml_accepted:
            return RiskDecision(False, "ml_filter_rejected")
        if self.state.cooldown_until and now < self.state.cooldown_until:
            return RiskDecision(False, "cooldown")
        if atr <= 0:
            return RiskDecision(False, "invalid_atr")
        if self.state.daily_pnl <= -(self.state.day_start_equity * self.settings.max_daily_loss):
            return RiskDecision(False, "daily_loss_limit")
        if equity <= self.state.peak_equity * (1 - self.settings.max_drawdown):
            self.state.circuit_open = True
            return RiskDecision(False, "drawdown_circuit_breaker")
        stop = entry - Decimal(str(self.settings.stop_atr_multiple)) * atr
        if stop <= 0 or stop >= entry:
            return RiskDecision(False, "invalid_stop")
        quantity = (equity * self.settings.risk_per_trade / (entry - stop)).quantize(
            self.settings.quantity_step, rounding=ROUND_DOWN
        )
        quantity = min(
            quantity,
            (equity * self.settings.max_exposure / entry).quantize(
                self.settings.quantity_step, rounding=ROUND_DOWN
            ),
        )
        if quantity * entry < self.settings.min_notional:
            return RiskDecision(False, "below_min_notional")
        if quantity * entry > cash:
            return RiskDecision(False, "insufficient_balance")
        return RiskDecision(True, "accepted", quantity, stop)

    def record_closed_trade(self, now: datetime, pnl: Decimal, equity: Decimal) -> None:
        self.state.peak_equity = max(self.state.peak_equity, equity)
        self.state.daily_pnl += pnl
        self.state.consecutive_losses = self.state.consecutive_losses + 1 if pnl < 0 else 0
        if self.state.consecutive_losses >= self.settings.consecutive_loss_limit:
            self.state.cooldown_until = now + timedelta(hours=self.settings.cooldown_hours)
