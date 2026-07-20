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
from trading_bot.data.binance_vision import (
    BinanceVisionError,
    BinanceVisionHistoricalClient,
    _checksum_from_text,
    _parse_archive,
    _verify_anomaly_continuity,
    parse_verified_archive_kline,
)
from trading_bot.data.csv_store import (
    CsvDataError,
    csv_sha256,
    merge_candles,
    read_candles,
    verify_metadata_checksum,
)
from trading_bot.data.historical import DataCoverageError, download_vision_historical_csv
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


def _zip(rows: str, inner_name: str = "BTCUSDT-1h.csv") -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(inner_name, rows)
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
    daily = _zip("".join(_row(hour) for hour in range(24)), "BTCUSDT-1h-2024-01-01.csv")
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
            text=("0" * 64 + "  BTCUSDT-1h-2024-01-01.zip")
            if request.url.path.endswith("CHECKSUM")
            else None,
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


def _verification_client(
    tmp_path: Path, handler: httpx.MockTransport
) -> BinanceVisionHistoricalClient:
    return BinanceVisionHistoricalClient(
        tmp_path / "cache",
        _rest_client([], BASE + timedelta(days=2)),
        base_url="https://vision.example/data/spot",
        transport=handler,
        now=lambda: BASE + timedelta(days=2),
        backoff_seconds=0,
        max_retries=0,
    )


def test_online_checksum_refresh_reuses_matching_cached_zip(tmp_path: Path) -> None:
    archive_name = "BTCUSDT-1h-2024-01-01.zip"
    data = b"verified zip bytes"
    calls = {"checksum": 0, "zip": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith(".CHECKSUM"):
            calls["checksum"] += 1
            return httpx.Response(
                200, text=f"{hashlib.sha256(data).hexdigest()}  {archive_name}", request=request
            )
        calls["zip"] += 1
        return httpx.Response(200, content=data, request=request)

    client = _verification_client(tmp_path, httpx.MockTransport(handler))
    assert (
        asyncio.run(client._verified_bytes(f"daily/klines/BTCUSDT/1h/{archive_name}", False))[0]
        == data
    )
    assert (
        asyncio.run(client._verified_bytes(f"daily/klines/BTCUSDT/1h/{archive_name}", False))[0]
        == data
    )
    assert calls == {"checksum": 2, "zip": 1}


def test_changed_official_checksum_replaces_cached_zip(tmp_path: Path) -> None:
    archive_name = "BTCUSDT-1h-2024-01-01.zip"
    state = {"data": b"old", "zip": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith(".CHECKSUM"):
            return httpx.Response(
                200,
                text=f"{hashlib.sha256(state['data']).hexdigest()}  {archive_name}",
                request=request,
            )
        state["zip"] += 1
        return httpx.Response(200, content=state["data"], request=request)

    client = _verification_client(tmp_path, httpx.MockTransport(handler))
    relative = f"daily/klines/BTCUSDT/1h/{archive_name}"
    assert asyncio.run(client._verified_bytes(relative, False))[0] == b"old"
    state["data"] = b"new"
    assert asyncio.run(client._verified_bytes(relative, False))[0] == b"new"
    assert state["zip"] == 2


def test_checksum_refresh_failure_does_not_fallback_to_cache(tmp_path: Path) -> None:
    archive_name = "BTCUSDT-1h-2024-01-01.zip"
    data = b"cached"
    cache = tmp_path / "cache" / "daily" / "klines" / "BTCUSDT" / "1h"
    cache.mkdir(parents=True)
    (cache / archive_name).write_bytes(data)
    (cache / f"{archive_name}.CHECKSUM").write_text(
        f"{hashlib.sha256(data).hexdigest()}  {archive_name}\n", encoding="utf-8"
    )
    client = _verification_client(
        tmp_path, httpx.MockTransport(lambda request: httpx.Response(503, request=request))
    )
    with pytest.raises(BinanceVisionError, match="request failed"):
        asyncio.run(client._verified_bytes(f"daily/klines/BTCUSDT/1h/{archive_name}", False))


def test_corrupt_cached_zip_is_refetched_and_verified(tmp_path: Path) -> None:
    archive_name = "BTCUSDT-1h-2024-01-01.zip"
    data = b"fresh"
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        if request.url.path.endswith(".CHECKSUM"):
            return httpx.Response(
                200, text=f"{hashlib.sha256(data).hexdigest()}  {archive_name}", request=request
            )
        calls += 1
        return httpx.Response(200, content=data, request=request)

    client = _verification_client(tmp_path, httpx.MockTransport(handler))
    relative = f"daily/klines/BTCUSDT/1h/{archive_name}"
    asyncio.run(client._verified_bytes(relative, False))
    (tmp_path / "cache" / relative).write_bytes(b"corrupt")
    assert asyncio.run(client._verified_bytes(relative, False))[0] == data
    assert calls == 2


@pytest.mark.parametrize(
    "sidecar",
    [
        "a" * 64 + "  other.zip",
        "a" * 64,
        "a" * 64 + "  BTCUSDT-1h-2024-01-01.zip\n" + "b" * 64 + "  second.zip",
        "a" * 64 + "  ../BTCUSDT-1h-2024-01-01.zip",
    ],
)
def test_checksum_sidecar_requires_single_matching_safe_filename(sidecar: str) -> None:
    with pytest.raises(BinanceVisionError):
        _checksum_from_text(sidecar, "BTCUSDT-1h-2024-01-01.zip")


@pytest.mark.parametrize(
    "inner_name",
    [
        "BTCUSDT-1h-2024-01-02.csv",
        "ETHUSDT-1h-2024-01-01.csv",
        "BTCUSDT-4h-2024-01-01.csv",
        "../BTCUSDT-1h-2024-01-01.csv",
    ],
)
def test_archive_rejects_wrong_or_unsafe_inner_csv_identity(inner_name: str) -> None:
    with pytest.raises(BinanceVisionError, match="inner CSV identity"):
        _parse_archive(_zip("", inner_name), "BTCUSDT-1h-2024-01-01.zip", "a" * 64)


def test_merge_deduplicates_identical_candles_and_rejects_conflicts() -> None:
    assert merge_candles([_candle(0)], [_candle(0)]) == [_candle(0)]
    with pytest.raises(CsvDataError, match="conflicting"):
        merge_candles([_candle(0)], [_candle(0, "101")])


class _FakeVisionClient:
    def __init__(self, candles: list[Candle], parsed: list[object] | None = None) -> None:
        self.candles = candles
        self.now = lambda: BASE + timedelta(days=2)
        self.parsed = parsed or []
        self.archive_urls = ["https://vision.example/monthly/klines/BTCUSDT/1h/test.zip"]
        self.monthly_archives = ["test.zip"]
        self.daily_archives: list[str] = []
        self.archive_checksums = {"test.zip": "a" * 64}
        self.checksum_verification_mode = "official_online"
        self.rest_suffix: tuple[datetime, datetime] | None = None
        self.rest_suffix_candle_count = len(candles) - len(self.parsed)

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


def test_metadata_and_report_record_checksum_verification_mode(tmp_path: Path) -> None:
    output = tmp_path / "btc.csv"
    client = _FakeVisionClient([_candle(0), _candle(1)])
    asyncio.run(
        download_vision_historical_csv(client, BASE, BASE + timedelta(hours=2), output, True)
    )
    metadata = json.loads(output.with_name("btc.csv.metadata.json").read_text(encoding="utf-8"))
    report = json.loads(output.with_suffix(".anomalies.json").read_text(encoding="utf-8"))
    assert metadata["checksum_verification_mode"] == "official_online"
    assert report["policy"]["checksum_verification_mode"] == "official_online"


@pytest.mark.parametrize("mode", ["cached_offline", None, 1])
def test_publisher_rejects_non_official_or_invalid_verification_mode(
    tmp_path: Path, mode: object
) -> None:
    output = tmp_path / "btc.csv"
    client = _FakeVisionClient([_candle(0), _candle(1)])
    client.checksum_verification_mode = mode
    with pytest.raises(DataCoverageError, match="official_online"):
        asyncio.run(
            download_vision_historical_csv(client, BASE, BASE + timedelta(hours=2), output, True)
        )
    assert not output.exists()


def test_publisher_rejects_client_without_verification_mode(tmp_path: Path) -> None:
    output = tmp_path / "btc.csv"
    client = _FakeVisionClient([_candle(0), _candle(1)])
    del client.checksum_verification_mode
    with pytest.raises(DataCoverageError, match="official_online"):
        asyncio.run(
            download_vision_historical_csv(client, BASE, BASE + timedelta(hours=2), output, True)
        )
    assert not output.exists()


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


def _audited_archive_client() -> _FakeVisionClient:
    parsed = []
    for hour, close_offset in ((0, 3_599_999), (1, 3_594_362), (2, 3_599_999)):
        opened = int((BASE + timedelta(hours=hour)).timestamp() * 1_000)
        parsed.append(
            parse_verified_archive_kline(
                [str(opened), "100", "101", "99", "100", "1", str(opened + close_offset)],
                archive_name="BTCUSDT-1h-2024-01.zip",
                archive_sha256="a" * 64,
                row_number=hour,
                checksum_verified=True,
            )
        )
    verified = _verify_anomaly_continuity(parsed, [item.candle for item in parsed])
    return _FakeVisionClient([_candle(hour) for hour in range(3)], verified)


def _rewrite_report_and_checksum(output: Path, report: dict[str, object]) -> None:
    report_path = output.with_suffix(".anomalies.json")
    report_path.write_text(json.dumps(report), encoding="utf-8")
    metadata_path = output.with_name(f"{output.name}.metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["anomaly_report_sha256"] = csv_sha256(report_path)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")


def _rewrite_metadata(output: Path, metadata: dict[str, object]) -> None:
    output.with_name(f"{output.name}.metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )


@pytest.mark.parametrize("mode", [None, "cached_offline"])
def test_validator_rejects_missing_or_non_official_metadata_verification_mode(
    tmp_path: Path, mode: object
) -> None:
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
    metadata_path = output.with_name("btc.csv.metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if mode is None:
        del metadata["checksum_verification_mode"]
    else:
        metadata["checksum_verification_mode"] = mode
    _rewrite_metadata(output, metadata)
    with pytest.raises(CsvDataError, match="verification mode"):
        verify_metadata_checksum(output)


@pytest.mark.parametrize(
    "case",
    ["missing_policy", "policy_not_dictionary", "missing_mode", "non_official_mode", "mismatch"],
)
def test_validator_rejects_invalid_report_verification_mode(tmp_path: Path, case: str) -> None:
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
    report_path = output.with_suffix(".anomalies.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if case == "missing_policy":
        del report["policy"]
    elif case == "policy_not_dictionary":
        report["policy"] = []
    elif case == "missing_mode":
        del report["policy"]["checksum_verification_mode"]
    else:
        report["policy"]["checksum_verification_mode"] = "cached_offline"
    _rewrite_report_and_checksum(output, report)
    with pytest.raises(CsvDataError, match="verification mode"):
        verify_metadata_checksum(output)


def test_metadata_maximum_deviation_matches_report(tmp_path: Path) -> None:
    output = tmp_path / "btc.csv"
    asyncio.run(
        download_vision_historical_csv(
            _audited_archive_client(), BASE, BASE + timedelta(hours=3), output, True
        )
    )
    metadata = json.loads(output.with_name("btc.csv.metadata.json").read_text(encoding="utf-8"))
    assert metadata["maximum_timestamp_deviation_us"] == 5_637_000
    assert metadata["maximum_timestamp_deviation_ms"] == "5637"
    assert verify_metadata_checksum(output) is True


def test_validator_rejects_truncated_anomaly_deviation(tmp_path: Path) -> None:
    output = tmp_path / "btc.csv"
    asyncio.run(
        download_vision_historical_csv(
            _audited_archive_client(), BASE, BASE + timedelta(hours=3), output, True
        )
    )
    report_path = output.with_suffix(".anomalies.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["anomalies"][0]["early_close_deviation_ms"] = "5637.0"
    _rewrite_report_and_checksum(output, report)
    with pytest.raises(CsvDataError, match="millisecond deviation"):
        verify_metadata_checksum(output)


@pytest.mark.parametrize("field", ["archive_candle_count", "rest_suffix_candle_count"])
def test_validator_rejects_inconsistent_archive_or_rest_counts(tmp_path: Path, field: str) -> None:
    output = tmp_path / "btc.csv"
    asyncio.run(
        download_vision_historical_csv(
            _audited_archive_client(), BASE, BASE + timedelta(hours=3), output, True
        )
    )
    metadata_path = output.with_name("btc.csv.metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata[field] += 1
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(CsvDataError, match="counts"):
        verify_metadata_checksum(output)
