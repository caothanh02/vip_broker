from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from trading_bot.data.binance_parser import BinanceKlineParseError, parse_binance_spot_1h_kline
from trading_bot.data.binance_vision import (
    ArchiveTimestampPolicyError,
    ArchiveTimestampQuality,
    parse_verified_archive_kline,
)


def _row(open_time: int, close_time: int) -> list[str]:
    return [str(open_time), "100", "101", "99", "100", "10", str(close_time)]


def _parse(row: list[str], verified: bool = True):  # type: ignore[no-untyped-def]
    return parse_verified_archive_kline(
        row,
        archive_name="BTCUSDT-1h-2021-12.zip",
        archive_sha256="a" * 64,
        row_number=556,
        checksum_verified=verified,
    )


def test_verified_archive_accepts_exact_close() -> None:
    opened = 1_640_318_400_000
    parsed = _parse(_row(opened, opened + 3_599_999))
    assert parsed.quality is ArchiveTimestampQuality.EXACT
    assert parsed.early_close_deviation_us == 0


def test_verified_archive_accepts_5637ms_early_close() -> None:
    opened = 1_640_318_400_000
    parsed = _parse(_row(opened, opened + 3_594_362))
    assert parsed.quality is ArchiveTimestampQuality.EARLY_CLOSE_WITHIN_TOLERANCE
    assert parsed.early_close_deviation_us == 5_637_000


def test_accepted_early_close_is_canonicalized_to_one_hour() -> None:
    opened = 1_640_318_400_000
    parsed = _parse(_row(opened, opened + 3_594_362))
    assert parsed.candle.open_time == datetime(2021, 12, 24, 4, tzinfo=UTC)
    assert parsed.candle.close_time == parsed.candle.open_time + timedelta(hours=1)


def test_verified_microsecond_archive_timestamp_is_accepted() -> None:
    opened = int(datetime(2025, 1, 1, tzinfo=UTC).timestamp() * 1_000_000)
    parsed = _parse(_row(opened, opened + 3_599_999_999))
    assert parsed.timestamp_unit == "microseconds"
    assert parsed.candle.open_time == datetime(2025, 1, 1, tzinfo=UTC)


def test_unverified_archive_cannot_use_relaxed_policy() -> None:
    opened = 1_640_318_400_000
    with pytest.raises(ArchiveTimestampPolicyError, match="verified checksum"):
        _parse(_row(opened, opened + 3_594_362), verified=False)


@pytest.mark.parametrize("delta", [3_539_998, 3_600_000])
def test_archive_rejects_large_early_close_and_late_close(delta: int) -> None:
    opened = 1_640_318_400_000
    with pytest.raises(ArchiveTimestampPolicyError):
        _parse(_row(opened, opened + delta))


def test_rest_parser_remains_strict() -> None:
    opened = 1_640_318_400_000
    with pytest.raises(BinanceKlineParseError):
        parse_binance_spot_1h_kline(_row(opened, opened + 3_594_362))
