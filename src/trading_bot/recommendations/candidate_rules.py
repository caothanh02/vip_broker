"""Registered, non-executable recommendation candidate predicates.

Each predicate receives only the causal feature window ending at the decision
candle.  This module deliberately has no broker, order, risk, model, or market
I/O dependency.
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd

BASELINE_CANDIDATE_ID = "baseline_ema_volume_atr_v1"
TREND_PULLBACK_CANDIDATE_ID = "trend_pullback_ema_atr_v2"

_CANDIDATE_PROTOCOLS = {
    BASELINE_CANDIDATE_ID: "development_walk_forward_v1",
    TREND_PULLBACK_CANDIDATE_ID: "development_walk_forward_v2",
}


class CandidateSettings(Protocol):
    """Immutable setting subset used by registered rule candidates."""

    @property
    def volume_multiplier(self) -> float: ...


def candidate_protocol(candidate_id: str) -> str:
    """Return the immutable protocol assigned to one registered candidate."""

    try:
        return _CANDIDATE_PROTOCOLS[candidate_id]
    except KeyError as exc:
        raise ValueError("recommendation candidate is unknown or unregistered") from exc


def is_trend_pullback_ema_atr_candidate(
    current: pd.Series, previous: pd.Series, settings: CandidateSettings
) -> bool:
    """Return whether the current closed candle reclaims EMA20 in a long trend."""

    needed = ["ema20", "ema50", "ema200", "volume_sma20", "atr14"]
    if not bool(current.get("is_closed", False)) or current[needed].isna().any():
        return False
    trend_regime = current.close > current.ema200 and current.ema20 > current.ema50
    pullback_reclaim = previous.close <= previous.ema20 and current.close > current.ema20
    volume_confirmation = current.volume >= settings.volume_multiplier * current.volume_sma20
    return bool(trend_regime and pullback_reclaim and volume_confirmation and current.atr14 > 0)
