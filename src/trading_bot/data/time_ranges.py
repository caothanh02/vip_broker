from __future__ import annotations

from datetime import UTC, datetime


def validate_hour_aligned_range(start: datetime, end: datetime) -> tuple[datetime, datetime]:
    """Normalize and validate a half-open UTC range for 1-hour candles."""
    normalized_start = _utc_hour(start)
    normalized_end = _utc_hour(end)
    if normalized_end <= normalized_start:
        raise ValueError(
            "start and end must be aligned to UTC 1-hour boundaries and end must be after start"
        )
    return normalized_start, normalized_end


def _utc_hour(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("start and end must be aligned to UTC 1-hour boundaries")
    normalized = value.astimezone(UTC)
    if normalized.minute or normalized.second or normalized.microsecond:
        raise ValueError("start and end must be aligned to UTC 1-hour boundaries")
    return normalized
