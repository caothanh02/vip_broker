from __future__ import annotations

import asyncio
import hashlib
import io
import json
import zipfile
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from trading_bot.data.binance_historical import BinanceHistoricalDataClient
from trading_bot.data.binance_vision import BinanceVisionError, BinanceVisionHistoricalClient
from trading_bot.data.csv_store import (
    CsvDataError,
    merge_candles,
    read_candles,
    verify_metadata_checksum,
)
from trading_bot.data.historical import download_vision_historical_csv
from trading_bot.domain.models import Candle

BASE = datetime(2024, 1, 1, tzinfo=UTC)


def _candle(hour: int, price: str = "100") -> Candle:
    value = Decimal(price)
    return Candle(
        BASE + timedelta(hours=hour),
        BASE + timedelta(hours=hour + 1),
        "BTC/USDT",
        "1h",
        value,
        value + 1,
        value - 1,
        value,
        Decimal("1"),
        True,
    )


def _row(hour: int, *, close_offset: int = 3_599_999) -> str:
    opened = int((BASE + timedelta(hours=hour)).timestamp() * 1000)
    return f"{opened},100,101,99,100,1,{opened + close_offset}\n"


def _zip(rows: str) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("BTCUSDT-1h.csv", rows)
    return stream.getvalue()


def _archive_transport(archives: dict[str, bytes]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        key = request.url.path.rsplit("/", maxsplit=1)[-1]
        is_checksum = key.endswith(".CHECKSUM")
        archive_name = key.removesuffix(".CHECKSUM")
        data = archives.get(archive_name)
        if data is None:
            return httpx.Response(404, request=request)
        if is_checksum:
            return httpx.Response(
                200, text=f"{hashlib.sha256(data).hexdigest()}  {archive_name}\n", request=request
            )
        return httpx.Response(200, content=data, request=request)

    return httpx.MockTransport(handler)


def _rest_client(rows: list[list[object]], now: datetime) -> BinanceHistoricalDataClient:
    return BinanceHistoricalDataClient(
        base_url="https://rest.example/api/v3",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=rows, request=request)
        ),
        now=lambda: now,
        backoff_seconds=0,
    )


def test_vision_uses_daily_archive_then_strict_rest_suffix(tmp_path: Path) -> None:
    daily = _zip("".join(_row(hour) for hour in range(24)))
    suffix: list[list[object]] = []
    for hour in range(24, 36):
        opened = int((BASE + timedelta(hours=hour)).timestamp() * 1000)
        suffix.append([opened, "100", "101", "99", "100", "1", opened + 3_599_999])
    client = BinanceVisionHistoricalClient(
        tmp_path / "cache",
        _rest_client(suffix, BASE + timedelta(days=3)),
        base_url="https://vision.example/data/spot",
        transport=_archive_transport({"BTCUSDT-1h-2024-01-01.zip": daily}),
        now=lambda: BASE + timedelta(days=3),
        backoff_seconds=0,
    )
    candles = asyncio.run(client.fetch_closed(BASE, BASE + timedelta(hours=36)))
    assert len(candles) == 36
    assert client.daily_archives == ["BTCUSDT-1h-2024-01-01.zip"]
    assert client.rest_suffix == (BASE + timedelta(days=1), BASE + timedelta(hours=36))


def test_vision_rejects_missing_historical_archives_instead_of_rest_patching(
    tmp_path: Path,
) -> None:
    client = BinanceVisionHistoricalClient(
        tmp_path / "cache",
        _rest_client([], datetime(2024, 3, 1, tzinfo=UTC)),
        base_url="https://vision.example/data/spot",
        transport=_archive_transport({}),
        now=lambda: datetime(2024, 3, 1, tzinfo=UTC),
        backoff_seconds=0,
    )
    with pytest.raises(BinanceVisionError, match="missing required daily archive"):
        asyncio.run(client.fetch_closed(BASE, datetime(2024, 2, 1, tzinfo=UTC)))


def test_vision_archive_checksum_mismatch_is_rejected(tmp_path: Path) -> None:
    data = _zip(_row(0))
    bad_checksum = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            text=("0" * 64 + " file.zip") if request.url.path.endswith("CHECKSUM") else None,
            content=data if request.url.path.endswith(".zip") else None,
            request=request,
        )
    )
    client = BinanceVisionHistoricalClient(
        tmp_path / "cache",
        _rest_client([], BASE + timedelta(days=2)),
        transport=bad_checksum,
        now=lambda: BASE + timedelta(days=2),
        backoff_seconds=0,
    )
    with pytest.raises(BinanceVisionError, match="checksum mismatch"):
        asyncio.run(client.fetch_closed(BASE, BASE + timedelta(days=1)))


def test_merge_deduplicates_identical_candles_and_rejects_conflicts() -> None:
    assert merge_candles([_candle(0)], [_candle(0)]) == [_candle(0)]
    with pytest.raises(CsvDataError, match="conflicting"):
        merge_candles([_candle(0)], [_candle(0, "101")])


class _FakeVisionClient:
    def __init__(self, candles: list[Candle]) -> None:
        self.candles = candles
        self.now = lambda: BASE + timedelta(days=2)
        self.parsed: list[object] = []
        self.archive_urls = ["https://vision.example/monthly/klines/BTCUSDT/1h/test.zip"]
        self.monthly_archives = ["test.zip"]
        self.daily_archives: list[str] = []
        self.archive_checksums = {"test.zip": "a" * 64}
        self.rest_suffix: tuple[datetime, datetime] | None = None

    async def fetch_closed(self, start_time: datetime, end_time: datetime) -> list[Candle]:
        return [candle for candle in self.candles if start_time <= candle.open_time < end_time]


def test_vision_generation_publishes_three_matching_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "btc.csv"
    client = _FakeVisionClient([_candle(0), _candle(1)])
    asyncio.run(
        download_vision_historical_csv(client, BASE, BASE + timedelta(hours=2), output, True)
    )
    metadata = json.loads((tmp_path / "btc.csv.metadata.json").read_text(encoding="utf-8"))
    report = json.loads((tmp_path / "btc.anomalies.json").read_text(encoding="utf-8"))
    assert metadata["generation_id"] == report["generation_id"]
    assert metadata["stored_candle_count"] == 2
    assert verify_metadata_checksum(output) is True


def test_vision_generation_rolls_back_if_commit_marker_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import trading_bot.data.historical as historical

    output = tmp_path / "btc.csv"
    client = _FakeVisionClient([_candle(0), _candle(1)])
    asyncio.run(
        download_vision_historical_csv(client, BASE, BASE + timedelta(hours=2), output, True)
    )
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    original_replace = historical._replace_generation_file

    def fail_metadata(source: Path, target: Path) -> None:
        if Path(target) == output.with_name("btc.csv.metadata.json"):
            raise OSError("simulated metadata publish failure")
        original_replace(source, target)

    monkeypatch.setattr(historical, "_replace_generation_file", fail_metadata)
    with pytest.raises(OSError, match="simulated"):
        asyncio.run(
            download_vision_historical_csv(client, BASE, BASE + timedelta(hours=2), output, True)
        )
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before
    assert verify_metadata_checksum(output) is True


def test_metadata_rejects_missing_or_mismatched_anomaly_sidecar(tmp_path: Path) -> None:
    output = tmp_path / "btc.csv"
    asyncio.run(
        download_vision_historical_csv(
            _FakeVisionClient([_candle(0), _candle(1)]),
            BASE,
            BASE + timedelta(hours=2),
            output,
            True,
        )
    )
    report_path = tmp_path / "btc.anomalies.json"
    report_path.unlink()
    with pytest.raises(CsvDataError, match="anomaly report is missing"):
        verify_metadata_checksum(output)
    report_path.write_text("{}", encoding="utf-8")
    with pytest.raises(CsvDataError, match="checksum"):
        verify_metadata_checksum(output)
    assert len(read_candles(output)) == 2


def test_metadata_rejects_csv_checksum_mismatch(tmp_path: Path) -> None:
    output = tmp_path / "btc.csv"
    asyncio.run(
        download_vision_historical_csv(
            _FakeVisionClient([_candle(0), _candle(1)]),
            BASE,
            BASE + timedelta(hours=2),
            output,
            True,
        )
    )
    output.write_text("tampered", encoding="utf-8")
    with pytest.raises(CsvDataError, match="CSV checksum mismatch"):
        verify_metadata_checksum(output)
