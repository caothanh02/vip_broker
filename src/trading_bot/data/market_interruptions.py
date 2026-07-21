"""Exact, audited exceptions for verified Binance Spot market interruptions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final


@dataclass(frozen=True, slots=True)
class KnownMarketInterruption:
    event_id: str
    market_type: str
    symbol: str
    timeframe: str
    archive_name: str
    archive_sha256: str
    raw_open_timestamp: int
    raw_close_timestamp: int
    event_start: datetime
    event_end: datetime
    missing_open_times: tuple[datetime, ...]
    official_source_urls: tuple[str, ...]


_EVENT_ID: Final = "binance-spot-2023-03-24-trailing-stop-maintenance"
_SOURCES: Final = (
    "https://www.binance.com/en/support/announcement/detail/813a31506e9f478ea8c1058b425df87a",
    "https://www.binance.com/en/square/post/344026",
)
_EVENT_START: Final = datetime(2023, 3, 24, 11, 27, tzinfo=UTC)
_EVENT_END: Final = datetime(2023, 3, 24, 14, tzinfo=UTC)
_MISSING: Final = (datetime(2023, 3, 24, 13, tzinfo=UTC),)


KNOWN_MARKET_INTERRUPTIONS: Final = (
    KnownMarketInterruption(
        _EVENT_ID,
        "spot",
        "BTCUSDT",
        "1h",
        "BTCUSDT-1h-2023-03.zip",
        "7f2afb8e0179a57ac31eab5205660298ba5eb77039ac2e21aef9b715ff3d06ce",
        1679659200000,
        1679661581646,
        _EVENT_START,
        _EVENT_END,
        _MISSING,
        _SOURCES,
    ),
    KnownMarketInterruption(
        _EVENT_ID,
        "spot",
        "BTCUSDT",
        "1h",
        "BTCUSDT-1h-2023-03-24.zip",
        "ea9d94f28a39ad8029c9c2863cbb7769137188edd957fea35d0313ae4183561f",
        1679659200000,
        1679661581646,
        _EVENT_START,
        _EVENT_END,
        _MISSING,
        _SOURCES,
    ),
)


def find_interruption(
    interruptions: tuple[KnownMarketInterruption, ...],
    *,
    archive_name: str,
    archive_sha256: str,
    raw_open_timestamp: int,
    raw_close_timestamp: int,
) -> KnownMarketInterruption | None:
    for event in interruptions:
        if (
            event.archive_name == archive_name
            and event.archive_sha256 == archive_sha256
            and event.raw_open_timestamp == raw_open_timestamp
            and event.raw_close_timestamp == raw_close_timestamp
        ):
            return event
    return None
