from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any
from uuid import uuid4


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(StrEnum):
    NEW = "new"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class Candle:
    open_time: datetime
    close_time: datetime
    symbol: str
    timeframe: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    is_closed: bool


@dataclass(frozen=True, slots=True)
class FeatureRow:
    candle_time: datetime
    values: dict[str, float]
    schema_version: str


@dataclass(frozen=True, slots=True)
class StrategySignal:
    candle_time: datetime
    side: Side
    reason: str
    atr: float
    feature_schema_version: str
    signal_id: str = field(default_factory=lambda: str(uuid4()))


@dataclass(frozen=True, slots=True)
class MLScore:
    model_version: str
    probability: float
    threshold: float
    accepted: bool
    feature_schema_version: str


@dataclass(frozen=True, slots=True)
class RiskDecision:
    accepted: bool
    reason: str
    quantity: Decimal = Decimal("0")
    stop_price: Decimal | None = None


@dataclass(frozen=True, slots=True)
class OrderRequest:
    symbol: str
    side: Side
    quantity: Decimal
    client_order_id: str
    signal_id: str


@dataclass(slots=True)
class Order:
    request: OrderRequest
    status: OrderStatus
    created_at: datetime
    order_id: str = field(default_factory=lambda: str(uuid4()))


@dataclass(frozen=True, slots=True)
class Fill:
    order_id: str
    side: Side
    quantity: Decimal
    price: Decimal
    fee: Decimal
    slippage: Decimal
    timestamp: datetime


@dataclass(slots=True)
class Position:
    symbol: str
    quantity: Decimal
    entry_price: Decimal
    stop_price: Decimal
    highest_price: Decimal
    opened_at: datetime
    entry_fee: Decimal = Decimal("0")


@dataclass(frozen=True, slots=True)
class Trade:
    symbol: str
    quantity: Decimal
    entry_price: Decimal
    exit_price: Decimal
    entry_time: datetime
    exit_time: datetime
    pnl: Decimal
    fees: Decimal
    exit_reason: str


@dataclass(frozen=True, slots=True)
class Balance:
    asset: str
    free: Decimal


@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    timestamp: datetime
    cash: Decimal
    equity: Decimal
    position_value: Decimal


@dataclass(frozen=True, slots=True)
class BotEvent:
    timestamp: datetime
    event_type: str
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    version: str
    model_type: str
    created_at: datetime
    feature_schema_version: str
    features: list[str]
    threshold: float
    ranges: dict[str, str]
    metrics: dict[str, float]
    checksum: str
