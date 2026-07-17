from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import ROUND_DOWN, Decimal

from trading_bot.domain.models import RiskDecision
from trading_bot.settings import BotSettings


@dataclass(slots=True)
class RiskState:
    peak_equity: Decimal
    day_start_equity: Decimal
    daily_pnl: Decimal = Decimal("0")
    daily_pnl_date: date | None = None
    daily_loss_open: bool = False
    last_daily_loss_breach_date: date | None = None
    consecutive_losses: int = 0
    cooldown_until: datetime | None = None
    circuit_open: bool = False


class RiskEngine:
    def __init__(self, settings: BotSettings, state: RiskState | None = None) -> None:
        self.settings, self.state = (
            settings,
            state or RiskState(settings.starting_cash, settings.starting_cash),
        )

    def _reset_daily_state_if_needed(self, now: datetime, equity: Decimal) -> None:
        """Start daily-loss accounting on the UTC date containing ``now``."""
        today = now.astimezone(UTC).date()
        if self.state.daily_pnl_date != today:
            self.state.day_start_equity = equity
            self.state.daily_pnl = Decimal("0")
            self.state.daily_pnl_date = today
            self.state.daily_loss_open = False

    def _mark_daily_loss_if_needed(self, equity: Decimal) -> None:
        if equity <= self.state.day_start_equity * (1 - self.settings.max_daily_loss):
            self.state.daily_loss_open = True
            self.state.last_daily_loss_breach_date = self.state.daily_pnl_date

    def mark_to_market(self, now: datetime, equity: Decimal) -> None:
        """Record candle-close equity before considering a subsequent entry."""
        self._reset_daily_state_if_needed(now, equity)
        self.state.peak_equity = max(self.state.peak_equity, equity)
        self._mark_daily_loss_if_needed(equity)
        if equity <= self.state.peak_equity * (1 - self.settings.max_drawdown):
            self.state.circuit_open = True

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
        self._reset_daily_state_if_needed(now, equity)
        if has_position:
            return RiskDecision(False, "position_already_open")
        if cash <= 0:
            return RiskDecision(False, "insufficient_balance")
        if not healthy or self.state.circuit_open:
            return RiskDecision(False, "circuit_breaker_or_unhealthy")
        if not ml_accepted:
            return RiskDecision(False, "ml_filter_rejected")
        if self.state.cooldown_until and now < self.state.cooldown_until:
            return RiskDecision(False, "cooldown")
        if atr <= 0:
            return RiskDecision(False, "invalid_atr")
        if self.state.daily_loss_open or equity <= self.state.day_start_equity * (
            1 - self.settings.max_daily_loss
        ):
            self._mark_daily_loss_if_needed(equity)
            return RiskDecision(False, "daily_loss_limit")
        if equity <= self.state.peak_equity * (1 - self.settings.max_drawdown):
            self.state.circuit_open = True
            return RiskDecision(False, "drawdown_circuit_breaker")
        # ``entry`` is the unadjusted market reference price.  Buys pay
        # adverse slippage, so all sizing and stop placement use the planned
        # fill rather than an optimistic quote.
        entry = entry * (1 + self.settings.entry_slippage_rate)
        stop = entry - Decimal(str(self.settings.stop_atr_multiple)) * atr
        if stop <= 0 or stop >= entry:
            return RiskDecision(False, "invalid_stop")

        # Size against the worst planned stop fill: adverse entry/exit slippage
        # and both commissions are costs of the same risk budget.
        entry_cost_per_unit = entry * (1 + self.settings.entry_fee_rate)
        planned_stop_fill = stop * (1 - self.settings.exit_slippage_rate)
        stop_proceeds_per_unit = planned_stop_fill * (1 - self.settings.exit_fee_rate)
        loss_per_unit = entry_cost_per_unit - stop_proceeds_per_unit
        if loss_per_unit <= 0:
            return RiskDecision(False, "invalid_planned_loss")
        quantity = (equity * self.settings.risk_per_trade / loss_per_unit).quantize(
            self.settings.quantity_step, rounding=ROUND_DOWN
        )
        quantity = min(
            quantity,
            (equity * self.settings.max_exposure / entry).quantize(
                self.settings.quantity_step, rounding=ROUND_DOWN
            ),
            (cash / entry_cost_per_unit).quantize(self.settings.quantity_step, rounding=ROUND_DOWN),
        )
        if quantity * entry < self.settings.min_notional:
            return RiskDecision(False, "below_min_notional")
        if quantity * entry > cash:
            return RiskDecision(False, "insufficient_balance")
        return RiskDecision(True, "accepted", quantity, stop)

    def record_closed_trade(self, now: datetime, pnl: Decimal, equity: Decimal) -> None:
        self._reset_daily_state_if_needed(now, equity)
        self.state.peak_equity = max(self.state.peak_equity, equity)
        self.state.daily_pnl += pnl
        self._mark_daily_loss_if_needed(equity)
        self.state.consecutive_losses = self.state.consecutive_losses + 1 if pnl < 0 else 0
        if self.state.consecutive_losses >= self.settings.consecutive_loss_limit:
            self.state.cooldown_until = now + timedelta(hours=self.settings.cooldown_hours)
