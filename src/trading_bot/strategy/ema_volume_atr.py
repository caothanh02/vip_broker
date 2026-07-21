from __future__ import annotations

import pandas as pd

from trading_bot.domain.models import Side, StrategySignal
from trading_bot.features.pipeline import FEATURE_SCHEMA_VERSION
from trading_bot.settings import BotSettings


def is_long_entry_candidate(current: pd.Series, previous: pd.Series, settings: BotSettings) -> bool:
    """Pure rule predicate shared by execution and candidate-only ML data."""
    needed = ["ema20", "ema50", "ema200", "volume_sma20", "atr14"]
    if not bool(current.get("is_closed", False)) or current[needed].isna().any():
        return False
    crossed = current.ema20 > current.ema50 and previous.ema20 <= previous.ema50
    volume_ok = current.volume > settings.volume_multiplier * current.volume_sma20
    return bool(crossed and current.close > current.ema200 and volume_ok and current.atr14 > 0)


def long_entry_candidate_mask(frame: pd.DataFrame, settings: BotSettings) -> pd.Series:
    """Vectorized execution-equivalent candidate predicate for research data."""
    needed = ["ema20", "ema50", "ema200", "volume_sma20", "atr14"]
    complete = frame[needed].notna().all(axis=1)
    crossed = (frame.ema20 > frame.ema50) & (frame.ema20.shift(1) <= frame.ema50.shift(1))
    volume_ok = frame.volume > settings.volume_multiplier * frame.volume_sma20
    return (
        frame.is_closed.astype(bool)
        & complete
        & crossed
        & (frame.close > frame.ema200)
        & volume_ok
        & (frame.atr14 > 0)
    ).fillna(False)


class EmaVolumeAtrStrategy:
    name = "EmaVolumeAtrStrategy"

    def __init__(self, settings: BotSettings) -> None:
        self.settings = settings

    def entry(
        self, frame: pd.DataFrame, has_position: bool, cooldown: bool, circuit_open: bool
    ) -> StrategySignal | None:
        if len(frame) < 2 or has_position or cooldown or circuit_open:
            return None
        current, previous = frame.iloc[-1], frame.iloc[-2]
        if is_long_entry_candidate(current, previous, self.settings):
            return StrategySignal(
                frame.index[-1].to_pydatetime(),
                Side.BUY,
                "ema20_crossed_ema50_volume_confirmed_above_ema200",
                float(current.atr14),
                FEATURE_SCHEMA_VERSION,
            )
        return None

    def exit_crossover(self, frame: pd.DataFrame) -> bool:
        if len(frame) < 2:
            return False
        c, p = frame.iloc[-1], frame.iloc[-2]
        return bool(c.ema20 < c.ema50 and p.ema20 >= p.ema50)
