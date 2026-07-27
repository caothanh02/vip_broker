from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path

from trading_bot.data.binance_historical import BinanceHistoricalDataClient
from trading_bot.data.csv_store import write_json_atomic
from trading_bot.data.validation import CandleValidationError, validate_candles
from trading_bot.domain.models import Candle, OrderRequest, Position, Side, StrategySignal, Trade
from trading_bot.execution.broker import DryRunBroker
from trading_bot.features.pipeline import build_features
from trading_bot.risk.engine import RiskEngine, RiskState
from trading_bot.settings import BotSettings
from trading_bot.strategy.ema_volume_atr import EmaVolumeAtrStrategy

_INTERVAL = timedelta(hours=1)
_STATE_VERSION = 1
_WARMUP_CANDLES = 240


class DryRunError(RuntimeError):
    """A paper-runtime invariant or public market-data operation failed."""


def _utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_time(value: object, field: str) -> datetime:
    if not isinstance(value, str):
        raise DryRunError(f"dry-run state {field} must be an ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DryRunError(f"dry-run state {field} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise DryRunError(f"dry-run state {field} must be UTC")
    return parsed.astimezone(UTC)


def _decimal(value: object, field: str) -> Decimal:
    if not isinstance(value, str):
        raise DryRunError(f"dry-run state {field} must be a decimal string")
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise DryRunError(f"dry-run state {field} is invalid") from exc
    if not parsed.is_finite():
        raise DryRunError(f"dry-run state {field} must be finite")
    return parsed


def _optional_time(value: object, field: str) -> datetime | None:
    return None if value is None else _parse_time(value, field)


def _optional_date(value: object, field: str) -> date | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise DryRunError(f"dry-run state {field} must be an ISO date")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise DryRunError(f"dry-run state {field} is invalid") from exc


@dataclass(slots=True)
class DryRunState:
    cash: Decimal
    risk: RiskState
    position: Position | None = None
    pending: StrategySignal | None = None
    pending_exit: str | None = None
    history: list[Candle] = field(default_factory=list)
    trades: list[Trade] = field(default_factory=list)


@dataclass(slots=True)
class DryRunHealth:
    ready: bool = False
    websocket_connected: bool = False
    last_candle_close: datetime | None = None
    last_error: str | None = None

    def as_dict(self) -> dict[str, str | bool | None]:
        return {
            "status": "ok" if self.ready else "unavailable",
            "mode": "dry_run",
            "websocket_connected": self.websocket_connected,
            "last_closed_candle": _utc(self.last_candle_close)
            if self.last_candle_close is not None
            else None,
            "last_error": self.last_error,
        }


class DryRunStateStore:
    """Versioned atomic persistence for paper-only position and risk state."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self, settings: BotSettings) -> DryRunState | None:
        if not self.path.exists():
            return None
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DryRunError("could not read dry-run state") from exc
        if not isinstance(raw, dict) or raw.get("state_version") != _STATE_VERSION:
            raise DryRunError("unsupported dry-run state version")
        if (
            raw.get("mode") != "dry_run"
            or raw.get("symbol") != "BTC/USDT"
            or raw.get("timeframe") != "1h"
        ):
            raise DryRunError("dry-run state market or mode is incompatible")
        risk_raw = raw.get("risk")
        if not isinstance(risk_raw, dict):
            raise DryRunError("dry-run state risk is invalid")
        risk = RiskState(
            peak_equity=_decimal(risk_raw.get("peak_equity"), "risk.peak_equity"),
            day_start_equity=_decimal(risk_raw.get("day_start_equity"), "risk.day_start_equity"),
            daily_pnl=_decimal(risk_raw.get("daily_pnl"), "risk.daily_pnl"),
            daily_pnl_date=_optional_date(risk_raw.get("daily_pnl_date"), "risk.daily_pnl_date"),
            daily_loss_open=_bool(risk_raw.get("daily_loss_open"), "risk.daily_loss_open"),
            last_daily_loss_breach_date=_optional_date(
                risk_raw.get("last_daily_loss_breach_date"), "risk.last_daily_loss_breach_date"
            ),
            consecutive_losses=_nonnegative_int(
                risk_raw.get("consecutive_losses"), "risk.consecutive_losses"
            ),
            cooldown_until=_optional_time(risk_raw.get("cooldown_until"), "risk.cooldown_until"),
            circuit_open=_bool(risk_raw.get("circuit_open"), "risk.circuit_open"),
        )
        history = _candles_from_json(raw.get("history"), "history")
        if history:
            validate_candles(history)
        position = _position_from_json(raw.get("position"))
        pending = _signal_from_json(raw.get("pending"))
        pending_exit = raw.get("pending_exit")
        if pending_exit is not None and not isinstance(pending_exit, str):
            raise DryRunError("dry-run state pending_exit is invalid")
        trades = _trades_from_json(raw.get("trades"))
        cash = _decimal(raw.get("cash"), "cash")
        if cash < 0:
            raise DryRunError("dry-run state cash cannot be negative")
        state = DryRunState(cash, risk, position, pending, pending_exit, history, trades)
        _validate_state(state, settings)
        return state

    def save(self, state: DryRunState) -> None:
        write_json_atomic(
            self.path,
            {
                "state_version": _STATE_VERSION,
                "mode": "dry_run",
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "cash": str(state.cash),
                "risk": _risk_json(state.risk),
                "position": _position_json(state.position),
                "pending": _signal_json(state.pending),
                "pending_exit": state.pending_exit,
                "history": [_candle_json(item) for item in state.history[-_WARMUP_CANDLES:]],
                "trades": [_trade_json(item) for item in state.trades],
            },
        )


def _bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise DryRunError(f"dry-run state {field} must be boolean")
    return value


def _nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DryRunError(f"dry-run state {field} must be a non-negative integer")
    return value


def _candle_json(candle: Candle) -> dict[str, str | bool]:
    return {
        "open_time": _utc(candle.open_time),
        "close_time": _utc(candle.close_time),
        "symbol": candle.symbol,
        "timeframe": candle.timeframe,
        "open": str(candle.open),
        "high": str(candle.high),
        "low": str(candle.low),
        "close": str(candle.close),
        "volume": str(candle.volume),
        "is_closed": candle.is_closed,
    }


def _candles_from_json(value: object, field: str) -> list[Candle]:
    if not isinstance(value, list):
        raise DryRunError(f"dry-run state {field} must be a list")
    candles: list[Candle] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise DryRunError(f"dry-run state {field}[{index}] is invalid")
        symbol = item.get("symbol")
        timeframe = item.get("timeframe")
        if not isinstance(symbol, str) or not isinstance(timeframe, str):
            raise DryRunError(f"dry-run state {field}[{index}] market is invalid")
        candles.append(
            Candle(
                _parse_time(item.get("open_time"), f"{field}[{index}].open_time"),
                _parse_time(item.get("close_time"), f"{field}[{index}].close_time"),
                symbol,
                timeframe,
                _decimal(item.get("open"), f"{field}[{index}].open"),
                _decimal(item.get("high"), f"{field}[{index}].high"),
                _decimal(item.get("low"), f"{field}[{index}].low"),
                _decimal(item.get("close"), f"{field}[{index}].close"),
                _decimal(item.get("volume"), f"{field}[{index}].volume"),
                _bool(item.get("is_closed"), f"{field}[{index}].is_closed"),
            )
        )
    return candles


def _position_json(position: Position | None) -> dict[str, str] | None:
    if position is None:
        return None
    return {
        "symbol": position.symbol,
        "quantity": str(position.quantity),
        "entry_price": str(position.entry_price),
        "stop_price": str(position.stop_price),
        "highest_price": str(position.highest_price),
        "opened_at": _utc(position.opened_at),
        "entry_fee": str(position.entry_fee),
    }


def _position_from_json(value: object) -> Position | None:
    if value is None:
        return None
    if not isinstance(value, dict) or value.get("symbol") != "BTC/USDT":
        raise DryRunError("dry-run state position is invalid")
    position = Position(
        "BTC/USDT",
        _decimal(value.get("quantity"), "position.quantity"),
        _decimal(value.get("entry_price"), "position.entry_price"),
        _decimal(value.get("stop_price"), "position.stop_price"),
        _decimal(value.get("highest_price"), "position.highest_price"),
        _parse_time(value.get("opened_at"), "position.opened_at"),
        _decimal(value.get("entry_fee"), "position.entry_fee"),
    )
    if position.quantity <= 0 or position.entry_price <= 0 or position.stop_price <= 0:
        raise DryRunError("dry-run state position has invalid values")
    return position


def _signal_json(signal: StrategySignal | None) -> dict[str, str | float] | None:
    if signal is None:
        return None
    return {
        "candle_time": _utc(signal.candle_time),
        "side": signal.side.value,
        "reason": signal.reason,
        "atr": signal.atr,
        "feature_schema_version": signal.feature_schema_version,
        "signal_id": signal.signal_id,
    }


def _signal_from_json(value: object) -> StrategySignal | None:
    if value is None:
        return None
    if not isinstance(value, dict) or value.get("side") != Side.BUY.value:
        raise DryRunError("dry-run state pending signal is invalid")
    atr = value.get("atr")
    if isinstance(atr, bool) or not isinstance(atr, (int, float)) or atr <= 0:
        raise DryRunError("dry-run state pending signal ATR is invalid")
    for name in ("reason", "feature_schema_version", "signal_id"):
        if not isinstance(value.get(name), str) or not value[name]:
            raise DryRunError(f"dry-run state pending signal {name} is invalid")
    return StrategySignal(
        _parse_time(value.get("candle_time"), "pending.candle_time"),
        Side.BUY,
        value["reason"],
        float(atr),
        value["feature_schema_version"],
        value["signal_id"],
    )


def _trade_json(trade: Trade) -> dict[str, str]:
    return {
        "symbol": trade.symbol,
        "quantity": str(trade.quantity),
        "entry_price": str(trade.entry_price),
        "exit_price": str(trade.exit_price),
        "entry_time": _utc(trade.entry_time),
        "exit_time": _utc(trade.exit_time),
        "pnl": str(trade.pnl),
        "fees": str(trade.fees),
        "exit_reason": trade.exit_reason,
    }


def _trades_from_json(value: object) -> list[Trade]:
    if not isinstance(value, list):
        raise DryRunError("dry-run state trades must be a list")
    result: list[Trade] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict) or item.get("symbol") != "BTC/USDT":
            raise DryRunError(f"dry-run state trades[{index}] is invalid")
        reason = item.get("exit_reason")
        if not isinstance(reason, str):
            raise DryRunError(f"dry-run state trades[{index}].exit_reason is invalid")
        result.append(
            Trade(
                "BTC/USDT",
                _decimal(item.get("quantity"), "trade.quantity"),
                _decimal(item.get("entry_price"), "trade.entry_price"),
                _decimal(item.get("exit_price"), "trade.exit_price"),
                _parse_time(item.get("entry_time"), "trade.entry_time"),
                _parse_time(item.get("exit_time"), "trade.exit_time"),
                _decimal(item.get("pnl"), "trade.pnl"),
                _decimal(item.get("fees"), "trade.fees"),
                reason,
            )
        )
    return result


def _risk_json(state: RiskState) -> dict[str, str | int | bool | None]:
    return {
        "peak_equity": str(state.peak_equity),
        "day_start_equity": str(state.day_start_equity),
        "daily_pnl": str(state.daily_pnl),
        "daily_pnl_date": state.daily_pnl_date.isoformat() if state.daily_pnl_date else None,
        "daily_loss_open": state.daily_loss_open,
        "last_daily_loss_breach_date": state.last_daily_loss_breach_date.isoformat()
        if state.last_daily_loss_breach_date
        else None,
        "consecutive_losses": state.consecutive_losses,
        "cooldown_until": _utc(state.cooldown_until) if state.cooldown_until else None,
        "circuit_open": state.circuit_open,
    }


def _validate_state(state: DryRunState, settings: BotSettings) -> None:
    if state.position is not None and state.cash < 0:
        raise DryRunError("dry-run state cash cannot be negative")
    if state.history and len(state.history) > _WARMUP_CANDLES:
        raise DryRunError("dry-run state history is too large")
    if state.risk.peak_equity <= 0 or state.risk.day_start_equity <= 0:
        raise DryRunError("dry-run state risk equity is invalid")
    if settings.bot_mode == "live":
        raise DryRunError("live mode is disabled")


class DryRunEngine:
    """Event-driven paper executor for validated, closed public candles only."""

    def __init__(self, settings: BotSettings, store: DryRunStateStore) -> None:
        if settings.bot_mode == "live":
            raise DryRunError("live mode is disabled")
        if settings.symbol != "BTC/USDT" or settings.timeframe != "1h":
            raise DryRunError("dry-run supports BTC/USDT 1h only")
        self.settings = settings
        self.store = store
        loaded = store.load(settings)
        self.state = loaded or DryRunState(
            settings.starting_cash, RiskState(settings.starting_cash, settings.starting_cash)
        )
        self.broker = DryRunBroker(
            settings.starting_cash,
            settings.entry_fee_rate,
            settings.exit_fee_rate,
            settings.entry_slippage_rate,
            settings.exit_slippage_rate,
        )
        self.broker.cash = self.state.cash
        self.risk = RiskEngine(settings, self.state.risk)
        self.strategy = EmaVolumeAtrStrategy(settings)
        self.health = DryRunHealth(ready=True, last_candle_close=self._last_close())

    def status(self) -> dict[str, str | bool | None]:
        return self.health.as_dict()

    def persist(self) -> None:
        self.state.risk = self.risk.state
        self.state.cash = self.broker.cash
        self.store.save(self.state)

    def warm(self, candles: Iterable[Candle]) -> None:
        items = list(candles)
        validate_candles(items)
        self.state.history = items[-_WARMUP_CANDLES:]
        self.health.last_candle_close = self._last_close()
        self.persist()

    def process(self, candle: Candle) -> None:
        try:
            validate_candles([candle])
            last = self._last_open()
            if last is not None:
                if candle.open_time == last:
                    return
                if candle.open_time != last + _INTERVAL:
                    raise DryRunError(
                        "closed-candle gap must be recovered through REST before processing"
                    )
            self._process_contiguous(candle)
            self.persist()
            self.health.ready = True
            self.health.last_error = None
            self.health.last_candle_close = candle.close_time
        except (CandleValidationError, DryRunError) as exc:
            self.health.ready = False
            self.health.last_error = str(exc)
            self.persist()
            raise DryRunError(str(exc)) from exc

    def _process_contiguous(self, candle: Candle) -> None:
        position = self.state.position
        opening_equity = self.broker.cash + (
            position.quantity * candle.open if position else Decimal("0")
        )
        exit_to_fill, self.state.pending_exit = self.state.pending_exit, None
        self.risk.mark_to_market(candle.open_time, opening_equity)
        if exit_to_fill is not None and position is not None:
            self._close(position, candle.open, candle.open_time, exit_to_fill)
            position = None
        elif position is not None and self.risk.state.circuit_open:
            self._close(
                position, candle.open, candle.open_time, "circuit_breaker_emergency_liquidation"
            )
            position = None
        if self.state.pending is not None:
            decision = self.risk.decide(
                candle.open_time,
                self.broker.cash,
                self.broker.cash,
                candle.open,
                Decimal(str(self.state.pending.atr)),
                False,
            )
            if decision.accepted:
                fill = self.broker.place_order(
                    OrderRequest(
                        "BTC/USDT",
                        Side.BUY,
                        decision.quantity,
                        self.state.pending.signal_id,
                        self.state.pending.signal_id,
                    ),
                    candle.open,
                    candle.open_time,
                )
                cost = fill.price * fill.quantity + fill.fee
                if cost > self.broker.cash:
                    raise DryRunError("risk-approved paper entry exceeds cash")
                self.broker.cash -= cost
                position = Position(
                    "BTC/USDT",
                    fill.quantity,
                    fill.price,
                    decision.stop_price or Decimal("0"),
                    fill.price,
                    candle.open_time,
                    fill.fee,
                )
            self.state.pending = None
        exposed = position is not None
        self.state.history.append(candle)
        self.state.history = self.state.history[-_WARMUP_CANDLES:]
        features = build_features(self.state.history)
        if position is not None:
            if candle.low <= position.stop_price:
                gap = candle.open <= position.stop_price
                self._close(
                    position,
                    candle.open if gap else position.stop_price,
                    candle.open_time if gap else candle.close_time,
                    "stop_loss_or_trailing",
                    candle.open_time,
                )
                position = None
            else:
                atr = features.iloc[-1].atr14
                if atr == atr and atr > 0:
                    position.highest_price = max(position.highest_price, candle.high)
                    position.stop_price = max(
                        position.stop_price,
                        position.highest_price
                        - Decimal(str(self.settings.trailing_atr_multiple * float(atr))),
                    )
                if self.strategy.exit_crossover(features):
                    self.state.pending_exit = "ema_cross_down"
        equity = self.broker.cash + (position.quantity * candle.close if position else Decimal("0"))
        self.risk.mark_to_market(candle.close_time, equity)
        if position is not None and self.risk.state.circuit_open:
            self.state.pending_exit = "circuit_breaker_emergency_liquidation"
        if position is None:
            signal = self.strategy.entry(
                features,
                False,
                bool(
                    self.risk.state.cooldown_until
                    and candle.close_time < self.risk.state.cooldown_until
                ),
                self.risk.state.circuit_open,
            )
            if signal is not None:
                self.state.pending = signal
        self.state.position = position
        del exposed  # Kept explicit above to make candle ordering auditable.

    def _close(
        self,
        position: Position,
        price: Decimal,
        timestamp: datetime,
        reason: str,
        risk_timestamp: datetime | None = None,
    ) -> None:
        fill = self.broker.place_order(
            OrderRequest(
                "BTC/USDT",
                Side.SELL,
                position.quantity,
                f"dry-run-exit-{position.opened_at.isoformat()}",
                "dry-run-exit",
            ),
            price,
            timestamp,
        )
        proceeds = fill.price * fill.quantity - fill.fee
        self.broker.cash += proceeds
        pnl = proceeds - (position.entry_price * position.quantity + position.entry_fee)
        self.state.trades.append(
            Trade(
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
        )
        self.risk.record_closed_trade(risk_timestamp or timestamp, pnl, self.broker.cash)

    def _last_open(self) -> datetime | None:
        return self.state.history[-1].open_time if self.state.history else None

    def _last_close(self) -> datetime | None:
        return self.state.history[-1].close_time if self.state.history else None


class DryRunService:
    """REST bootstrap/gap-recovery coordinator for a ``DryRunEngine``."""

    def __init__(
        self,
        engine: DryRunEngine,
        rest: BinanceHistoricalDataClient,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.engine, self.rest = engine, rest
        self.now = now or (lambda: datetime.now(UTC))

    async def bootstrap(self) -> None:
        end = self._closed_boundary()
        last = self.engine._last_open()
        if last is None:
            candles = await self.rest.fetch_closed(end - _WARMUP_CANDLES * _INTERVAL, end)
            if len(candles) < _WARMUP_CANDLES:
                raise DryRunError("REST bootstrap returned insufficient closed candles")
            self.engine.warm(candles)
            return
        expected = last + _INTERVAL
        if expected < end:
            await self._recover(expected, end)

    async def consume(self, stream: AsyncIterator[Candle], max_candles: int | None = None) -> int:
        processed = 0
        async for candle in stream:
            self.engine.health.websocket_connected = True
            expected = self.engine._last_open()
            if expected is not None:
                expected += _INTERVAL
                if candle.open_time > expected:
                    await self._recover(expected, candle.open_time)
            self.engine.process(candle)
            processed += 1
            if max_candles is not None and processed >= max_candles:
                break
        return processed

    async def _recover(self, start: datetime, end: datetime) -> None:
        candles = await self.rest.fetch_closed(start, end)
        if end > start and not candles:
            raise DryRunError("REST gap recovery returned no closed candles")
        try:
            validate_candles(candles)
        except CandleValidationError as exc:
            raise DryRunError("REST gap recovery returned invalid candles") from exc
        if candles[0].open_time != start or candles[-1].close_time != end:
            raise DryRunError("REST gap recovery did not cover the complete gap")
        for candle in candles:
            self.engine.process(candle)

    def _closed_boundary(self) -> datetime:
        now = self.now()
        if now.tzinfo is None or now.utcoffset() is None:
            raise DryRunError("dry-run clock must be timezone aware")
        return now.astimezone(UTC).replace(minute=0, second=0, microsecond=0)


async def replay(service: DryRunService, candles: Iterable[Candle]) -> int:
    async def source() -> AsyncIterator[Candle]:
        for candle in candles:
            yield candle

    return await service.consume(source())
