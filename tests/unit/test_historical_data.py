from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest

from trading_bot.cli import _settings_snapshot, main, parse_utc
from trading_bot.data.binance import BinancePublicClient
from trading_bot.data.binance_historical import (
    BinanceHistoricalDataClient,
    BinanceRateLimitError,
    BinanceResponseError,
)
from trading_bot.data.binance_parser import (
    BinanceKlineParseError,
    datetime_from_milliseconds,
    parse_binance_spot_1h_kline,
    parse_binance_spot_1h_websocket_kline,
)
from trading_bot.data.csv_store import (
    CsvDataError,
    csv_sha256,
    merge_candles,
    read_candles,
    write_candles_atomic,
)
from trading_bot.data.historical import DataCoverageError, download_historical_csv
from trading_bot.data.validation import CandleValidationError, validate_candles
from trading_bot.domain.models import Candle
from trading_bot.settings import BotSettings

BASE = datetime(2024, 1, 1, tzinfo=UTC)


def candle(hour: int, price: str = "100") -> Candle:
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
        Decimal("123.4567890123456789"),
        True,
    )


def kline(hour: int, price: str = "100") -> list[Any]:
    opened = int((BASE + timedelta(hours=hour)).timestamp() * 1000)
    value = Decimal(price)
    return [
        opened,
        price,
        str(value + 1),
        str(value - 1),
        price,
        "123.4567890123456789",
        opened + 3_599_999,
    ]


def client(handler: httpx.MockTransport, **kwargs: Any) -> BinanceHistoricalDataClient:
    return BinanceHistoricalDataClient(
        transport=handler,
        now=lambda: BASE + timedelta(days=30),
        backoff_seconds=0,
        **kwargs,
    )


def test_binance_client_canonicalizes_close_and_preserves_decimal() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json=[kline(0, "100.123456789")])
    )
    candles = asyncio.run(client(transport).fetch_closed(BASE, BASE + timedelta(hours=1)))
    assert candles == [candle(0, "100.123456789")]
    assert candles[0].close_time == candles[0].open_time + timedelta(hours=1)
    assert candles[0].volume == Decimal("123.4567890123456789")


def test_binance_client_paginates_without_duplicates() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        start = int(request.url.params["startTime"])
        if start == int(BASE.timestamp() * 1000):
            return httpx.Response(200, json=[kline(0), kline(1)])
        if start == int((BASE + timedelta(hours=2)).timestamp() * 1000):
            return httpx.Response(200, json=[kline(2)])
        return httpx.Response(200, json=[])

    candles = asyncio.run(
        client(httpx.MockTransport(handler), page_limit=2).fetch_closed(
            BASE, BASE + timedelta(hours=4)
        )
    )
    assert [item.open_time for item in candles] == [
        BASE + timedelta(hours=index) for index in range(3)
    ]


def test_binance_client_omits_current_open_candle() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=[kline(0)])

    candles = asyncio.run(
        BinanceHistoricalDataClient(
            transport=httpx.MockTransport(handler),
            now=lambda: BASE + timedelta(minutes=30),
            backoff_seconds=0,
        ).fetch_closed(BASE, BASE + timedelta(hours=1))
    )
    assert candles == []
    assert calls == 0


def test_binance_client_retries_429_and_fails_malformed_payload() -> None:
    calls = 0
    sleeps: list[float] = []

    async def sleep(value: float) -> None:
        sleeps.append(value)

    def retry_handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(429 if calls == 1 else 200, json=[])

    assert (
        asyncio.run(
            client(httpx.MockTransport(retry_handler), sleep=sleep).fetch_closed(
                BASE, BASE + timedelta(hours=1)
            )
        )
        == []
    )
    assert calls == 2 and sleeps == [0]
    malformed = httpx.MockTransport(lambda request: httpx.Response(200, json={"bad": "payload"}))
    with pytest.raises(BinanceResponseError):
        asyncio.run(client(malformed).fetch_closed(BASE, BASE + timedelta(hours=1)))


def test_binance_client_retries_5xx_and_rejects_invalid_kline_fields() -> None:
    calls = 0

    def retry_handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503 if calls == 1 else 200, json=[])

    assert (
        asyncio.run(
            client(httpx.MockTransport(retry_handler)).fetch_closed(BASE, BASE + timedelta(hours=1))
        )
        == []
    )
    invalid = kline(0)
    invalid[1] = "not-a-decimal"
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=[invalid]))
    with pytest.raises(BinanceResponseError, match="decimal"):
        asyncio.run(client(transport).fetch_closed(BASE, BASE + timedelta(hours=1)))


def test_binance_client_detects_non_progress_and_retry_exhaustion() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=[kline(0)]))
    with pytest.raises(BinanceResponseError, match="before the cursor"):
        asyncio.run(client(transport).fetch_closed(BASE, BASE + timedelta(hours=2)))
    rate_limited = httpx.MockTransport(lambda request: httpx.Response(429, json=[]))
    with pytest.raises(BinanceRateLimitError):
        asyncio.run(
            client(rate_limited, max_retries=0).fetch_closed(BASE, BASE + timedelta(hours=1))
        )


def test_csv_round_trip_deduplicates_and_rejects_conflicts(tmp_path: Path) -> None:
    path = tmp_path / "btc.csv"
    written = write_candles_atomic(path, [candle(1), candle(0), candle(1)])
    assert [item.open_time for item in written] == [BASE, BASE + timedelta(hours=1)]
    assert read_candles(path) == written
    with pytest.raises(CsvDataError, match="conflicting"):
        merge_candles([candle(0)], [candle(0, "101")])
    original = path.read_text(encoding="utf-8")
    with pytest.raises(CsvDataError):
        write_candles_atomic(path, [candle(0), candle(2)])
    assert path.read_text(encoding="utf-8") == original


class FakeClient:
    def __init__(self, response: list[Candle]) -> None:
        self.response = response
        self.calls: list[tuple[datetime, datetime]] = []
        self.now = lambda: BASE + timedelta(days=30)

    async def fetch_closed(self, start: datetime, end: datetime) -> list[Candle]:
        self.calls.append((start, end))
        return self.response


def test_incremental_download_merges_and_skips_complete_range(tmp_path: Path) -> None:
    path = tmp_path / "btc.csv"
    write_candles_atomic(path, [candle(0), candle(1)])
    fake = FakeClient([candle(2)])
    summary = asyncio.run(download_historical_csv(fake, BASE, BASE + timedelta(hours=3), path))
    assert fake.calls == [(BASE + timedelta(hours=2), BASE + timedelta(hours=3))]
    assert summary.old_candles == 2 and summary.new_candles == 1
    assert len(read_candles(path)) == 3
    metadata = json.loads((tmp_path / "btc.csv.metadata.json").read_text(encoding="utf-8"))
    assert metadata["stored_candle_count"] == 3
    complete = FakeClient([])
    asyncio.run(download_historical_csv(complete, BASE, BASE + timedelta(hours=3), path))
    assert complete.calls == []


def test_incremental_overwrite_replaces_only_after_success(tmp_path: Path) -> None:
    path = tmp_path / "btc.csv"
    write_candles_atomic(path, [candle(0), candle(1)])
    fake = FakeClient([candle(4), candle(5)])
    summary = asyncio.run(
        download_historical_csv(
            fake, BASE + timedelta(hours=4), BASE + timedelta(hours=6), path, True
        )
    )
    assert summary.old_candles == 0
    assert [item.open_time for item in read_candles(path)] == [
        BASE + timedelta(hours=4),
        BASE + timedelta(hours=5),
    ]


def test_cli_backtests_csv_and_writes_report(tmp_path: Path) -> None:
    data = tmp_path / "btc.csv"
    report = tmp_path / "report.json"
    write_candles_atomic(data, [candle(index, str(100 + index)) for index in range(240)])
    assert main(["backtest", "--input", str(data), "--output", str(report)]) == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["input_file"] == str(data)
    assert payload["candle_count"] == 240
    assert main(["validate-data", "--input", str(tmp_path / "missing.csv")]) == 1


def test_cli_accepts_date_only_as_utc_midnight() -> None:
    assert parse_utc("2024-01-01") == datetime(2024, 1, 1, tzinfo=UTC)


def test_cli_accepts_z_timestamp() -> None:
    assert parse_utc("2024-01-01T00:00:00Z") == datetime(2024, 1, 1, tzinfo=UTC)


def test_cli_accepts_offset_timestamp_and_normalizes_to_utc() -> None:
    assert parse_utc("2024-01-01T07:00:00+07:00") == datetime(2024, 1, 1, tzinfo=UTC)


@pytest.mark.parametrize("value", ["2024-01-01T12:00:00", "2024-01-01 12:00:00"])
def test_cli_rejects_naive_datetime_with_time(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="timezone offset"):
        parse_utc(value)


@pytest.mark.parametrize(
    "value", ["2024/01/01", "01-01-2024", "", "not-a-date", "2024-02-30", "2024-01-01T00:00:00UTC"]
)
def test_cli_rejects_invalid_date_formats(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_utc(value)


def test_makefile_default_date_format_is_supported() -> None:
    makefile = (Path(__file__).parents[2] / "Makefile").read_text(encoding="utf-8")
    start = next(
        line.split("?=", maxsplit=1)[1].strip()
        for line in makefile.splitlines()
        if line.startswith("DATA_START")
    )
    end = next(
        line.split("?=", maxsplit=1)[1].strip()
        for line in makefile.splitlines()
        if line.startswith("DATA_END")
    )
    assert parse_utc(start) == datetime(2024, 1, 1, tzinfo=UTC)
    assert parse_utc(end) == datetime(2024, 2, 1, tzinfo=UTC)


class RangeClient(FakeClient):
    def __init__(self, responses: dict[tuple[datetime, datetime], list[Candle]]) -> None:
        super().__init__([])
        self.responses = responses

    async def fetch_closed(self, start: datetime, end: datetime) -> list[Candle]:
        self.calls.append((start, end))
        return self.responses[(start, end)]


def test_incremental_backfills_missing_requested_prefix(tmp_path: Path) -> None:
    path = tmp_path / "btc.csv"
    write_candles_atomic(path, [candle(2), candle(3)])
    fake = RangeClient(
        {
            (BASE, BASE + timedelta(hours=2)): [candle(0), candle(1)],
            (BASE + timedelta(hours=4), BASE + timedelta(hours=5)): [candle(4)],
        }
    )
    summary = asyncio.run(download_historical_csv(fake, BASE, BASE + timedelta(hours=5), path))
    assert fake.calls == [
        (BASE, BASE + timedelta(hours=2)),
        (BASE + timedelta(hours=4), BASE + timedelta(hours=5)),
    ]
    assert summary.requested_range_candle_count == 5
    assert [item.open_time for item in read_candles(path)] == [
        BASE + timedelta(hours=index) for index in range(5)
    ]


def test_incremental_existing_file_covers_requested_range_without_api_call(tmp_path: Path) -> None:
    path = tmp_path / "btc.csv"
    write_candles_atomic(path, [candle(index) for index in range(5)])
    fake = FakeClient([])
    summary = asyncio.run(
        download_historical_csv(fake, BASE + timedelta(hours=1), BASE + timedelta(hours=4), path)
    )
    assert fake.calls == []
    assert summary.old_candles == 5 and summary.new_candles == 0
    assert summary.requested_range_candle_count == 3


def test_incremental_partial_response_does_not_replace_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "btc.csv"
    write_candles_atomic(path, [candle(0), candle(1)])
    original = path.read_text(encoding="utf-8")
    with pytest.raises(DataCoverageError):
        asyncio.run(
            download_historical_csv(FakeClient([candle(2)]), BASE, BASE + timedelta(hours=4), path)
        )
    assert path.read_text(encoding="utf-8") == original


def test_incremental_existing_gap_fails_before_api_call(tmp_path: Path) -> None:
    path = tmp_path / "btc.csv"
    write_candles_atomic(path, [candle(0), candle(1), candle(2)])
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join([lines[0], lines[1], lines[3]]) + "\n", encoding="utf-8")
    fake = FakeClient([])
    with pytest.raises(CsvDataError):
        asyncio.run(download_historical_csv(fake, BASE, BASE + timedelta(hours=3), path))
    assert fake.calls == []


def test_settings_snapshot_and_backtest_report_exclude_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = BotSettings(
        database_url="postgresql://alice:super-secret@db.example.com/trading",
        binance_api_key="test-api-key",
        binance_api_secret="test-api-secret",
        live_trading_confirmation="unsafe-value",
    )
    snapshot = _settings_snapshot(settings)
    assert "database_url" not in snapshot
    assert {"binance_api_key", "binance_api_secret", "live_trading_confirmation"}.isdisjoint(
        snapshot
    )
    monkeypatch.setattr("trading_bot.cli.load_settings", lambda: settings)
    report = tmp_path / "report.json"
    assert main(["backtest", "--fixture", "--output", str(report)]) == 0
    raw = report.read_text(encoding="utf-8")
    for secret in ("super-secret", "test-api-key", "test-api-secret", "unsafe-value", "alice"):
        assert secret not in raw


def test_all_binance_sources_use_canonical_timestamp_and_legacy_delegates() -> None:
    parsed = parse_binance_spot_1h_kline(kline(0, "100.123456789"))
    websocket = parse_binance_spot_1h_websocket_kline(
        {
            "t": kline(0)[0],
            "T": kline(0)[6],
            "o": "100.123456789",
            "h": "101",
            "l": "99",
            "c": "100.123456789",
            "v": "1.234567890123456789",
        }
    )
    assert parsed.close_time - parsed.open_time == timedelta(hours=1)
    assert websocket.close_time - websocket.open_time == timedelta(hours=1)
    assert parsed.open == Decimal("100.123456789")
    invalid = kline(0)
    invalid[6] = invalid[0] + 3_600_000
    with pytest.raises(BinanceKlineParseError):
        parse_binance_spot_1h_kline(invalid)

    fake = FakeClient([candle(0)])
    result = asyncio.run(
        BinancePublicClient(fake).historical(
            int(BASE.timestamp() * 1000), int((BASE + timedelta(hours=1)).timestamp() * 1000) - 1
        )
    )
    assert result == [candle(0)]
    assert fake.calls == [(BASE, BASE + timedelta(hours=1))]


def test_parser_accepts_exact_binance_1h_timestamps_and_preserves_milliseconds() -> None:
    parsed = parse_binance_spot_1h_kline(kline(0))
    assert parsed.open_time == BASE
    assert parsed.close_time == BASE + timedelta(hours=1)


@pytest.mark.parametrize(
    "open_offset,close_offset",
    [
        (0, 300_000),
        (0, 3_600_000),
        (1, 3_599_999),
        (0.0, 3_599_999),
    ],
)
def test_parser_rejects_invalid_binance_1h_timestamps(
    open_offset: float, close_offset: int
) -> None:
    invalid = kline(0)
    invalid[0] = invalid[0] + open_offset
    invalid[6] = int(BASE.timestamp() * 1000) + close_offset
    with pytest.raises(BinanceKlineParseError):
        parse_binance_spot_1h_kline(invalid)


@pytest.mark.parametrize("value", [True, 1.0, "1.0", "01", -1])
def test_millisecond_conversion_rejects_lossy_or_invalid_values(value: object) -> None:
    with pytest.raises(BinanceKlineParseError):
        datetime_from_milliseconds(value)


def test_legacy_adapter_rejects_lossy_timestamp_before_fetch() -> None:
    fake = FakeClient([])
    with pytest.raises(BinanceKlineParseError):
        asyncio.run(BinancePublicClient(fake).historical(1.0))  # type: ignore[arg-type]
    assert fake.calls == []


def test_historical_client_rejects_unaligned_range_before_request() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=[])

    with pytest.raises(ValueError):
        asyncio.run(
            client(httpx.MockTransport(handler)).fetch_closed(
                BASE + timedelta(minutes=1), BASE + timedelta(hours=1)
            )
        )
    assert calls == 0


def test_download_rejects_invalid_ranges_before_api_call(tmp_path: Path) -> None:
    path = tmp_path / "btc.csv"
    cases = [
        (BASE + timedelta(minutes=30), BASE + timedelta(hours=2)),
        (BASE, BASE + timedelta(hours=1, minutes=30)),
        (datetime(2024, 1, 1), BASE + timedelta(hours=1)),
        (BASE + timedelta(hours=1), BASE),
    ]
    for start, end in cases:
        fake = FakeClient([])
        with pytest.raises(ValueError):
            asyncio.run(download_historical_csv(fake, start, end, path))
        assert fake.calls == []


def test_future_end_is_capped_at_last_closed_hour(tmp_path: Path) -> None:
    path = tmp_path / "btc.csv"
    fake = FakeClient([candle(0), candle(1)])
    fake.now = lambda: BASE + timedelta(hours=2, minutes=37)
    summary = asyncio.run(download_historical_csv(fake, BASE, BASE + timedelta(hours=5), path))
    assert fake.calls == [(BASE, BASE + timedelta(hours=2))]
    assert summary.effective_end == BASE + timedelta(hours=2)
    assert summary.last_candle_close == BASE + timedelta(hours=2)


def test_metadata_checksum_is_verified_and_missing_sidecar_warns(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "btc.csv"
    fake = FakeClient([candle(0), candle(1)])
    asyncio.run(download_historical_csv(fake, BASE, BASE + timedelta(hours=2), path))
    metadata_path = path.with_name(f"{path.name}.metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["csv_sha256"] == csv_sha256(path)
    assert main(["validate-data", "--input", str(path)]) == 0
    metadata["csv_sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    assert main(["validate-data", "--input", str(path)]) == 1
    metadata_path.unlink()
    assert main(["validate-data", "--input", str(path)]) == 0
    assert "metadata checksum sidecar is missing" in capsys.readouterr().err


def test_incremental_noop_metadata_checksum_is_correct(tmp_path: Path) -> None:
    path = tmp_path / "btc.csv"
    write_candles_atomic(path, [candle(0), candle(1)])
    fake = FakeClient([])
    asyncio.run(download_historical_csv(fake, BASE, BASE + timedelta(hours=2), path))
    metadata = json.loads((tmp_path / "btc.csv.metadata.json").read_text(encoding="utf-8"))
    assert fake.calls == []
    assert metadata["csv_sha256"] == csv_sha256(path)


@pytest.mark.parametrize("value", ["NaN", "sNaN", "Infinity", "-Infinity"])
def test_data_validation_rejects_nonfinite_decimals(value: str) -> None:
    invalid = candle(0)
    invalid = Candle(
        invalid.open_time,
        invalid.close_time,
        invalid.symbol,
        invalid.timeframe,
        Decimal(value),
        invalid.high,
        invalid.low,
        invalid.close,
        invalid.volume,
        invalid.is_closed,
    )
    with pytest.raises(CandleValidationError, match="non-finite"):
        validate_candles([invalid])


@pytest.mark.parametrize("value", ["NaN", "sNaN", "Infinity", "-Infinity"])
def test_binance_parser_rejects_nonfinite_decimals(value: str) -> None:
    invalid = kline(0)
    invalid[1] = value
    with pytest.raises(BinanceKlineParseError, match="non-finite"):
        parse_binance_spot_1h_kline(invalid)


@pytest.mark.parametrize("value", ["NaN", "sNaN", "Infinity", "-Infinity"])
def test_csv_reader_rejects_nonfinite_decimals(tmp_path: Path, value: str) -> None:
    path = tmp_path / "btc.csv"
    path.write_text(
        "open_time,close_time,symbol,timeframe,open,high,low,close,volume,is_closed\n"
        f"2024-01-01T00:00:00Z,2024-01-01T01:00:00Z,BTC/USDT,1h,{value},101,99,100,1,true\n",
        encoding="utf-8",
    )
    with pytest.raises(CsvDataError, match="malformed"):
        read_candles(path)
