from __future__ import annotations

from datetime import UTC, datetime, timedelta

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
        if candle.open_time.tzinfo is None or candle.close_time.tzinfo is None:
            raise CandleValidationError("naive timestamp")
        if candle.open_time.tzinfo is not UTC or candle.close_time.tzinfo is not UTC:
            raise CandleValidationError("timestamp must be UTC")
        if candle.symbol != "BTC/USDT" or candle.timeframe != "1h":
            raise CandleValidationError("unsupported market")
        if not candle.is_closed:
            raise CandleValidationError("open candle")
        if candle.open_time in seen:
            raise CandleValidationError("duplicate timestamp")
        if min(candle.open, candle.high, candle.low, candle.close) <= 0 or candle.volume < 0:
            raise CandleValidationError("invalid OHLCV")
        if candle.close_time - candle.open_time != interval:
            raise CandleValidationError("invalid candle duration")
        if candle.high < max(candle.open, candle.close, candle.low) or candle.low > min(
            candle.open, candle.close, candle.high
        ):
            raise CandleValidationError("invalid OHLC")
        if previous and candle.open_time - previous != interval:
            raise CandleValidationError("data gap or bad interval")
        seen.add(candle.open_time)
        previous = candle.open_time
    if max_age is not None and datetime.now(UTC) - candles[-1].close_time > max_age:
        raise CandleValidationError("stale data")


def deduplicate(candles: list[Candle]) -> list[Candle]:
    return list({c.open_time: c for c in sorted(candles, key=lambda x: x.open_time)}.values())
