from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from decimal import Decimal

from trading_bot.domain.models import Balance, Fill, OrderRequest, Side


class Broker(ABC):
    @abstractmethod
    def get_balance(self) -> Balance: ...
    @abstractmethod
    def place_order(self, request: OrderRequest, price: Decimal, timestamp: datetime) -> Fill: ...


class SimulatedBroker(Broker):
    def __init__(
        self,
        cash: Decimal,
        entry_fee_rate: Decimal,
        exit_fee_rate: Decimal,
        entry_slippage: Decimal,
        exit_slippage: Decimal,
    ) -> None:
        self.cash, self.entry_fee_rate, self.exit_fee_rate = cash, entry_fee_rate, exit_fee_rate
        self.entry_slippage, self.exit_slippage = entry_slippage, exit_slippage
        self.orders: dict[str, Fill] = {}

    def get_balance(self) -> Balance:
        return Balance("USDT", self.cash)

    def place_order(self, request: OrderRequest, price: Decimal, timestamp: datetime) -> Fill:
        if request.client_order_id in self.orders:
            return self.orders[request.client_order_id]
        rate = self.entry_slippage if request.side == Side.BUY else self.exit_slippage
        actual = price * (1 + rate if request.side == Side.BUY else 1 - rate)
        fee_rate = self.entry_fee_rate if request.side == Side.BUY else self.exit_fee_rate
        fee = actual * request.quantity * fee_rate
        fill = Fill(
            request.client_order_id,
            request.side,
            request.quantity,
            actual,
            fee,
            actual * request.quantity * rate,
            timestamp,
        )
        self.orders[request.client_order_id] = fill
        return fill


class DryRunBroker(SimulatedBroker):
    """Paper broker; its inherited implementation cannot send exchange orders."""


class BinanceBroker(Broker):
    """Intentional safety lock: live order submission is not implemented."""

    def get_balance(self) -> Balance:
        raise RuntimeError("live broker is disabled")

    def place_order(self, request: OrderRequest, price: Decimal, timestamp: datetime) -> Fill:
        raise RuntimeError("live trading is disabled")
