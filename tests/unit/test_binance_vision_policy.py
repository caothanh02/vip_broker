from __future__ import annotations

import io
import zipfile
from datetime import UTC, datetime, timedelta

import pytest

from trading_bot.data.binance_parser import BinanceKlineParseError, parse_binance_spot_1h_kline
from trading_bot.data.binance_vision import (
    ArchiveTimestampPolicyError,
    ArchiveTimestampQuality,
    BinanceVisionError,
    _parse_archive,
    _verify_anomaly_continuity,
    anomaly_report,
    parse_verified_archive_kline,
)


def _row(open_time: int, close_time: int) -> list[str]:
    return [str(open_time), "100", "101", "99", "100", "10", str(close_time)]


def _parse(
    row: list[str],
    verified: bool = True,
    archive_name: str = "BTCUSDT-1h-2021-12.zip",
):  # type: ignore[no-untyped-def]
    return parse_verified_archive_kline(
        row,
        archive_name=archive_name,
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
    parsed = _parse(_row(opened, opened + 3_599_999_999), archive_name="BTCUSDT-1h-2025-01.zip")
    assert parsed.timestamp_unit == "microseconds"
    assert parsed.candle.open_time == datetime(2025, 1, 1, tzinfo=UTC)


def test_pre_2025_millisecond_archive_timestamp_is_accepted() -> None:
    opened = int(datetime(2024, 12, 31, 23, tzinfo=UTC).timestamp() * 1_000)
    assert _parse(_row(opened, opened + 3_599_999), archive_name="BTCUSDT-1h-2024-12.zip")


def test_pre_2025_microsecond_archive_timestamp_is_rejected() -> None:
    opened = int(datetime(2024, 12, 31, 23, tzinfo=UTC).timestamp() * 1_000_000)
    with pytest.raises(ArchiveTimestampPolicyError, match="milliseconds required"):
        _parse(_row(opened, opened + 3_599_999_999), archive_name="BTCUSDT-1h-2024-12.zip")


def test_post_2025_millisecond_archive_timestamp_is_rejected() -> None:
    opened = int(datetime(2025, 1, 1, tzinfo=UTC).timestamp() * 1_000)
    with pytest.raises(ArchiveTimestampPolicyError, match="microseconds required"):
        _parse(_row(opened, opened + 3_599_999), archive_name="BTCUSDT-1h-2025-01.zip")


def test_archive_rejects_mixed_open_close_timestamp_units() -> None:
    opened = int(datetime(2024, 12, 31, 23, tzinfo=UTC).timestamp() * 1_000)
    with pytest.raises(ArchiveTimestampPolicyError, match="mixed"):
        _parse(
            _row(opened, (opened + 3_599_999) * 1_000),
            archive_name="BTCUSDT-1h-2024-12.zip",
        )


def test_unit_boundary_at_exact_2025_01_01() -> None:
    opened = int(datetime(2025, 1, 1, tzinfo=UTC).timestamp() * 1_000_000)
    assert (
        _parse(
            _row(opened, opened + 3_599_999_999), archive_name="BTCUSDT-1h-2025-01.zip"
        ).timestamp_unit
        == "microseconds"
    )


def test_monthly_archive_accepts_rows_inside_named_month() -> None:
    opened = int(datetime(2024, 3, 15, tzinfo=UTC).timestamp() * 1_000)
    assert _parse(_row(opened, opened + 3_599_999), archive_name="BTCUSDT-1h-2024-03.zip")


def test_monthly_archive_rejects_row_from_different_month() -> None:
    opened = int(datetime(2024, 4, 1, tzinfo=UTC).timestamp() * 1_000)
    with pytest.raises(ArchiveTimestampPolicyError, match="outside"):
        _parse(_row(opened, opened + 3_599_999), archive_name="BTCUSDT-1h-2024-03.zip")


def test_daily_archive_accepts_rows_inside_named_day() -> None:
    opened = int(datetime(2024, 3, 15, 23, tzinfo=UTC).timestamp() * 1_000)
    assert _parse(_row(opened, opened + 3_599_999), archive_name="BTCUSDT-1h-2024-03-15.zip")


def test_daily_archive_rejects_row_from_different_day() -> None:
    opened = int(datetime(2024, 3, 16, tzinfo=UTC).timestamp() * 1_000)
    with pytest.raises(ArchiveTimestampPolicyError, match="outside"):
        _parse(_row(opened, opened + 3_599_999), archive_name="BTCUSDT-1h-2024-03-15.zip")


def test_archive_rejects_invalid_filename() -> None:
    opened = int(datetime(2024, 3, 1, tzinfo=UTC).timestamp() * 1_000)
    with pytest.raises(ArchiveTimestampPolicyError, match="filename"):
        _parse(_row(opened, opened + 3_599_999), archive_name="BTCUSDT-1h-2024-03-extra.zip")


def _archive_bytes(rows: list[list[str]]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("BTCUSDT-1h.csv", "\n".join(",".join(row) for row in rows))
    return stream.getvalue()


def _day_rows(day: datetime) -> list[list[str]]:
    start = int(day.timestamp() * 1_000)
    return [
        _row(start + hour * 3_600_000, start + hour * 3_600_000 + 3_599_999) for hour in range(24)
    ]


def test_daily_archive_requires_exactly_24_candles() -> None:
    with pytest.raises(BinanceVisionError, match="expected 24"):
        _parse_archive(
            _archive_bytes(_day_rows(datetime(2024, 3, 15, tzinfo=UTC))[:-1]),
            "BTCUSDT-1h-2024-03-15.zip",
            "a" * 64,
        )


def test_daily_archive_rejects_duplicate_timestamp() -> None:
    rows = _day_rows(datetime(2024, 3, 15, tzinfo=UTC))
    rows[5] = rows[4]
    with pytest.raises(BinanceVisionError, match="coverage error"):
        _parse_archive(_archive_bytes(rows), "BTCUSDT-1h-2024-03-15.zip", "a" * 64)


@pytest.mark.parametrize("omitted", [0, -1, 12])
def test_monthly_archive_rejects_missing_first_last_or_internal_candle(omitted: int) -> None:
    start = datetime(2024, 2, 1, tzinfo=UTC)
    rows = [row for day in range(29) for row in _day_rows(start + timedelta(days=day))]
    rows.pop(omitted)
    with pytest.raises(BinanceVisionError):
        _parse_archive(_archive_bytes(rows), "BTCUSDT-1h-2024-02.zip", "a" * 64)


def test_microsecond_anomaly_report_preserves_sub_millisecond_precision() -> None:
    start = datetime(2025, 1, 2, tzinfo=UTC)
    raw = int(start.timestamp() * 1_000_000)
    previous = _parse(_row(raw - 3_600_000_000, raw - 1), archive_name="BTCUSDT-1h-2025-01.zip")
    anomaly = _parse(
        _row(raw, raw + 3_599_999_999 - 5_637_999), archive_name="BTCUSDT-1h-2025-01.zip"
    )
    following = _parse(
        _row(raw + 3_600_000_000, raw + 7_199_999_999), archive_name="BTCUSDT-1h-2025-01.zip"
    )
    verified = _verify_anomaly_continuity(
        [previous, anomaly, following], [item.candle for item in (previous, anomaly, following)]
    )
    record = anomaly_report(verified)["anomalies"][0]
    assert record["early_close_deviation_us"] == 5_637_999
    assert record["early_close_deviation_ms"] == "5637.999"


def test_anomaly_continuity_is_computed_not_hardcoded() -> None:
    opened = int(datetime(2024, 3, 1, 1, tzinfo=UTC).timestamp() * 1_000)
    previous = _parse(_row(opened - 3_600_000, opened - 1), archive_name="BTCUSDT-1h-2024-03.zip")
    anomaly = _parse(_row(opened, opened + 3_594_362), archive_name="BTCUSDT-1h-2024-03.zip")
    following = _parse(
        _row(opened + 3_600_000, opened + 7_199_999), archive_name="BTCUSDT-1h-2024-03.zip"
    )
    assert anomaly.adjacent_continuity_verified is False
    verified = _verify_anomaly_continuity(
        [previous, anomaly, following], [item.candle for item in (previous, anomaly, following)]
    )
    assert verified[1].adjacent_continuity_verified is True


def test_anomaly_at_archive_boundary_requires_neighbor_archive() -> None:
    midnight = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1_000)
    previous = _parse(
        _row(midnight - 3_600_000, midnight - 1), archive_name="BTCUSDT-1h-2023-12.zip"
    )
    anomaly = _parse(_row(midnight, midnight + 3_594_362), archive_name="BTCUSDT-1h-2024-01.zip")
    following = _parse(
        _row(midnight + 3_600_000, midnight + 7_199_999), archive_name="BTCUSDT-1h-2024-01.zip"
    )
    verified = _verify_anomaly_continuity(
        [previous, anomaly, following], [item.candle for item in (previous, anomaly, following)]
    )
    assert verified[1].adjacent_continuity_verified is True


def test_anomaly_without_next_neighbor_is_rejected() -> None:
    opened = int(datetime(2024, 3, 1, tzinfo=UTC).timestamp() * 1_000)
    anomaly = _parse(_row(opened, opened + 3_594_362), archive_name="BTCUSDT-1h-2024-03.zip")
    with pytest.raises(BinanceVisionError, match="cannot verify"):
        _verify_anomaly_continuity([anomaly], [anomaly.candle])


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
