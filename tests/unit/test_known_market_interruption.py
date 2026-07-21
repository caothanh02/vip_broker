from __future__ import annotations

import asyncio
import io
import json
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from trading_bot.cli import main
from trading_bot.data.binance_vision import (
    ArchiveTimestampPolicyError,
    ArchiveTimestampQuality,
    _parse_archive,
    _verify_anomaly_continuity,
    parse_verified_archive_kline,
)
from trading_bot.data.csv_store import (
    CsvDataError,
    csv_sha256,
    read_candles,
    verified_missing_open_times,
    verify_metadata_checksum,
)
from trading_bot.data.historical import download_vision_historical_csv
from trading_bot.data.market_interruptions import KNOWN_MARKET_INTERRUPTIONS
from trading_bot.domain.models import Candle

EVENT = KNOWN_MARKET_INTERRUPTIONS[0]
BASE = datetime(2023, 3, 24, 11, tzinfo=UTC)


def _row(opened: int, closed: int) -> list[str]:
    return [str(opened), "100", "101", "99", "100", "1", str(closed)]


def _parsed(hour: int, close_offset: int = 3_599_999):
    opened = int((BASE + timedelta(hours=hour)).timestamp() * 1_000)
    if hour == 1:
        return parse_verified_archive_kline(
            _row(EVENT.raw_open_timestamp, EVENT.raw_close_timestamp),
            archive_name=EVENT.archive_name,
            archive_sha256=EVENT.archive_sha256,
            row_number=564,
            checksum_verified=True,
        )
    return parse_verified_archive_kline(
        _row(opened, opened + close_offset),
        archive_name=EVENT.archive_name,
        archive_sha256=EVENT.archive_sha256,
        row_number=hour,
        checksum_verified=True,
    )


def test_exact_official_interruption_identity_is_accepted() -> None:
    parsed = _parsed(1)
    assert parsed.quality is ArchiveTimestampQuality.KNOWN_MARKET_INTERRUPTION
    assert parsed.interruption == EVENT
    assert parsed.candle.close_time == parsed.candle.open_time + timedelta(hours=1)


@pytest.mark.parametrize(
    "archive_name,archive_sha256,raw_open,raw_close",
    [
        (
            EVENT.archive_name,
            EVENT.archive_sha256,
            EVENT.raw_open_timestamp,
            EVENT.raw_close_timestamp + 1,
        ),
        (
            EVENT.archive_name,
            EVENT.archive_sha256,
            EVENT.raw_open_timestamp + 3_600_000,
            EVENT.raw_close_timestamp,
        ),
        (EVENT.archive_name, "0" * 64, EVENT.raw_open_timestamp, EVENT.raw_close_timestamp),
        (
            "BTCUSDT-1h-2023-04.zip",
            EVENT.archive_sha256,
            EVENT.raw_open_timestamp,
            EVENT.raw_close_timestamp,
        ),
    ],
)
def test_interruption_requires_every_exact_identity_field(
    archive_name: str, archive_sha256: str, raw_open: int, raw_close: int
) -> None:
    with pytest.raises(ArchiveTimestampPolicyError):
        parse_verified_archive_kline(
            _row(raw_open, raw_close),
            archive_name=archive_name,
            archive_sha256=archive_sha256,
            row_number=564,
            checksum_verified=True,
        )


def test_non_allowlisted_large_early_close_remains_rejected() -> None:
    opened = int(datetime(2023, 3, 25, tzinfo=UTC).timestamp() * 1_000)
    with pytest.raises(ArchiveTimestampPolicyError, match="invalid_early_close"):
        parse_verified_archive_kline(
            _row(opened, opened + 2_381_646),
            archive_name="BTCUSDT-1h-2023-03.zip",
            archive_sha256=EVENT.archive_sha256,
            row_number=0,
            checksum_verified=True,
        )


def test_daily_archive_accepts_only_the_audited_missing_hour() -> None:
    day = datetime(2023, 3, 24, tzinfo=UTC)
    rows = []
    for hour in range(24):
        if hour == 13:
            continue
        opened = int((day + timedelta(hours=hour)).timestamp() * 1_000)
        closed = EVENT.raw_close_timestamp if hour == 12 else opened + 3_599_999
        rows.append(",".join(_row(opened, closed)))
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("BTCUSDT-1h-2023-03-24.csv", "\n".join(rows))
    parsed = _parse_archive(
        stream.getvalue(),
        EVENT.archive_name.replace("2023-03.zip", "2023-03-24.zip"),
        KNOWN_MARKET_INTERRUPTIONS[1].archive_sha256,
    )
    assert len(parsed) == 23
    assert parsed[12].quality is ArchiveTimestampQuality.KNOWN_MARKET_INTERRUPTION


class _InterruptionClient:
    def __init__(self, parsed: list[object]) -> None:
        self.parsed = parsed
        self.now = lambda: BASE + timedelta(hours=5)
        self.checksum_verification_mode = "official_online"
        self.archive_urls = [
            "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2023-03.zip"
        ]
        self.monthly_archives = [EVENT.archive_name]
        self.daily_archives: list[str] = []
        self.archive_checksums = {EVENT.archive_name: EVENT.archive_sha256}
        self.rest_suffix = None
        self.rest_suffix_candle_count = 0

    async def fetch_closed(self, start: datetime, end: datetime) -> list[Candle]:
        return [item.candle for item in self.parsed if start <= item.candle.open_time < end]


def _audited_client() -> _InterruptionClient:
    parsed = _verify_anomaly_continuity(
        [_parsed(0), _parsed(1), _parsed(3)],
        [_parsed(0).candle, _parsed(1).candle, _parsed(3).candle],
    )
    return _InterruptionClient(parsed)


def test_verified_dataset_records_non_tradable_event_and_backtest_refuses(tmp_path: Path) -> None:
    output = tmp_path / "btc.csv"
    asyncio.run(
        download_vision_historical_csv(
            _audited_client(), BASE, BASE + timedelta(hours=4), output, True
        )
    )
    metadata = json.loads(output.with_name("btc.csv.metadata.json").read_text(encoding="utf-8"))
    report = json.loads(output.with_suffix(".anomalies.json").read_text(encoding="utf-8"))
    assert metadata["contains_non_tradable_intervals"] is True
    assert metadata["market_interruption_event_count"] == 1
    assert report["market_interruptions"][0]["tradable"] is False
    assert verify_metadata_checksum(output) is True
    assert verified_missing_open_times(output) == {datetime(2023, 3, 24, 13, tzinfo=UTC)}
    with pytest.raises(CsvDataError, match="data gap"):
        read_candles(output)
    assert (
        main(["backtest", "--input", str(output), "--output", str(tmp_path / "report.json")]) == 1
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("event_id", "tampered"),
        ("archive", "bad.zip"),
        ("archive_sha256", "0" * 64),
        ("raw_close_timestamp", 1),
        ("symbol", "ETHUSDT"),
        ("timeframe", "4h"),
        ("official_source_urls", []),
    ],
)
def test_validator_rejects_tampered_market_interruption_identity(
    tmp_path: Path, field: str, value: object
) -> None:
    output = tmp_path / "btc.csv"
    asyncio.run(
        download_vision_historical_csv(
            _audited_client(), BASE, BASE + timedelta(hours=4), output, True
        )
    )
    report_path = output.with_suffix(".anomalies.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["market_interruptions"][0][field] = value
    report_path.write_text(json.dumps(report), encoding="utf-8")
    metadata_path = output.with_name("btc.csv.metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["anomaly_report_sha256"] = csv_sha256(report_path)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(CsvDataError, match="interruption"):
        verify_metadata_checksum(output)
