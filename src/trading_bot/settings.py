from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Literal

import yaml
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class BotSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", protected_namespaces=("settings_",)
    )
    bot_mode: Literal["backtest", "dry_run", "live"] = "backtest"
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    ml_filter_enabled: bool = False
    model_version: str = ""
    live_trading_confirmation: str = ""
    binance_api_key: str = ""
    binance_api_secret: str = ""
    database_url: str = "sqlite:///data/trading_bot.db"
    starting_cash: Decimal = Decimal("10000")
    entry_fee_rate: Decimal = Decimal("0.001")
    exit_fee_rate: Decimal = Decimal("0.001")
    entry_slippage_rate: Decimal = Decimal("0.0005")
    exit_slippage_rate: Decimal = Decimal("0.0005")
    risk_per_trade: Decimal = Decimal("0.005")
    max_exposure: Decimal = Decimal("0.30")
    max_daily_loss: Decimal = Decimal("0.03")
    max_drawdown: Decimal = Decimal("0.10")
    consecutive_loss_limit: int = 3
    cooldown_hours: int = 24
    min_notional: Decimal = Decimal("10")
    quantity_step: Decimal = Decimal("0.000001")
    ema_fast: int = 20
    ema_slow: int = 50
    ema_trend: int = 200
    volume_window: int = 20
    volume_multiplier: float = 1.2
    atr_window: int = 14
    stop_atr_multiple: float = 2.0
    trailing_atr_multiple: float = 2.0

    @model_validator(mode="after")
    def block_live_without_explicit_acknowledgement(self) -> BotSettings:
        if self.symbol != "BTC/USDT" or self.timeframe != "1h":
            raise ValueError("this bot supports BTC/USDT 1h only")
        if self.bot_mode == "live":
            raise ValueError("live mode is deliberately disabled in this build")
        return self


def load_settings(path: Path = Path("config/base.yaml")) -> BotSettings:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    flat: dict[str, object] = {}
    for section, values in raw.items():
        if isinstance(values, dict):
            for key, value in values.items():
                flat["bot_mode" if section == "bot" and key == "mode" else key] = value
    return BotSettings.model_validate(flat)
