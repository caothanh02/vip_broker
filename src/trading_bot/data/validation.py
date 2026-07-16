from __future__ import annotations

from datetime import timedelta

from trading_bot.domain.models import Candle


class CandleValidationError(ValueError):
    pass


def validate_candles(candles: list[Candle], max_age: timedelta | None = None) -> None:
    if not candles:
        raise CandleValidationError("no candles")
    seen = set()
    previous = None
    interval = timedelta(hours=1)
    for candle in candles:
        if candle.symbol != "BTC/USDT" or candle.timeframe != "1h":
            raise CandleValidationError("unsupported market")
        if not candle.is_closed:
            raise CandleValidationError("open candle")
        if candle.open_time in seen:
            raise CandleValidationError("duplicate timestamp")
        if min(candle.open, candle.high, candle.low, candle.close, candle.volume) < 0:
            raise CandleValidationError("negative OHLCV")
        if candle.high < max(candle.open, candle.close, candle.low) or candle.low > min(
            candle.open, candle.close, candle.high
        ):
            raise CandleValidationError("invalid OHLC")
        if previous and candle.open_time - previous != interval:
            raise CandleValidationError("data gap or bad interval")
        seen.add(candle.open_time)
        previous = candle.open_time


def deduplicate(candles: list[Candle]) -> list[Candle]:
    return list({c.open_time: c for c in sorted(candles, key=lambda x: x.open_time)}.values())
