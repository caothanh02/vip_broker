from __future__ import annotations

import numpy as np
import pandas as pd

from trading_bot.domain.models import Candle

FEATURE_SCHEMA_VERSION = "1.0.0"
FEATURE_COLUMNS = [
    "ema20",
    "ema50",
    "ema200",
    "ema20_slope",
    "ema50_slope",
    "ema20_ema50_distance",
    "close_ema200_distance",
    "rsi14",
    "macd",
    "macd_signal",
    "adx14",
    "atr14",
    "atr_close",
    "rolling_volatility_24",
    "bb_width",
    "volume_sma20",
    "volume_ratio",
    "volume_change",
    "return_1",
    "return_3",
    "return_6",
    "return_12",
    "return_24",
    "hour_utc",
    "day_of_week",
]


def build_features(candles: list[Candle]) -> pd.DataFrame:
    rows = [
        {
            "time": c.close_time,
            "open": float(c.open),
            "high": float(c.high),
            "low": float(c.low),
            "close": float(c.close),
            "volume": float(c.volume),
            "is_closed": c.is_closed,
        }
        for c in candles
    ]
    frame = pd.DataFrame(rows).set_index("time")
    close, high, low, volume = frame.close, frame.high, frame.low, frame.volume
    for span, name in [(20, "ema20"), (50, "ema50"), (200, "ema200")]:
        frame[name] = close.ewm(span=span, adjust=False, min_periods=span).mean()
    frame["ema20_slope"] = frame.ema20.pct_change(fill_method=None)
    frame["ema50_slope"] = frame.ema50.pct_change(fill_method=None)
    frame["ema20_ema50_distance"] = (frame.ema20 - frame.ema50) / close
    frame["close_ema200_distance"] = (close - frame.ema200) / close
    delta = close.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    rs = (
        up.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        / down.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    )
    frame["rsi14"] = 100 - 100 / (1 + rs)
    fast, slow = close.ewm(span=12, adjust=False).mean(), close.ewm(span=26, adjust=False).mean()
    frame["macd"] = fast - slow
    frame["macd_signal"] = frame.macd.ewm(span=9, adjust=False).mean()
    previous = close.shift(1)
    tr = pd.concat([high - low, (high - previous).abs(), (low - previous).abs()], axis=1).max(
        axis=1
    )
    frame["atr14"] = tr.rolling(14, min_periods=14).mean()
    frame["atr_close"] = frame.atr14 / close
    plus_dm = (high.diff()).clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    atr_sum = tr.rolling(14).sum()
    pdi = 100 * plus_dm.rolling(14).sum() / atr_sum
    mdi = 100 * minus_dm.rolling(14).sum() / atr_sum
    frame["adx14"] = (100 * (pdi - mdi).abs() / (pdi + mdi)).rolling(14).mean()
    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    frame["bb_width"] = (4 * std) / mid
    frame["rolling_volatility_24"] = close.pct_change(fill_method=None).rolling(24).std()
    frame["volume_sma20"] = volume.rolling(20).mean()
    frame["volume_ratio"] = volume / frame.volume_sma20
    frame["volume_change"] = volume.pct_change(fill_method=None)
    for n in [1, 3, 6, 12, 24]:
        frame[f"return_{n}"] = close.pct_change(n, fill_method=None)
    frame["hour_utc"] = frame.index.hour.astype(float)
    frame["day_of_week"] = frame.index.dayofweek.astype(float)
    frame["schema_version"] = FEATURE_SCHEMA_VERSION
    frame.replace([np.inf, -np.inf], np.nan, inplace=True)
    return frame
