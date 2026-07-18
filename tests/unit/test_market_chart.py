from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from trading_bot.charting.market_chart import ChartError, build_market_chart, open_market_chart
from trading_bot.cli import main
from trading_bot.data.csv_store import (
    csv_sha256,
    metadata_path,
    write_candles_atomic,
    write_metadata_atomic,
)
from trading_bot.domain.models import Candle, Trade


def _candles() -> list[Candle]:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    return [
        Candle(
            start + timedelta(hours=index),
            start + timedelta(hours=index + 1),
            "BTC/USDT",
            "1h",
            Decimal("100") + index,
            Decimal("102") + index,
            Decimal("99") + index,
            Decimal("101") + index,
            Decimal("1000") + index,
            True,
        )
        for index in range(2)
    ]


def _trade() -> Trade:
    candles = _candles()
    return Trade(
        "BTC/USDT",
        Decimal("1"),
        Decimal("100"),
        Decimal("101"),
        candles[0].open_time,
        candles[1].open_time,
        Decimal("1"),
        Decimal("0"),
        "signal_exit",
    )


def test_build_market_chart_writes_self_contained_interactive_html(tmp_path: Path) -> None:
    output = tmp_path / "market.html"
    summary = build_market_chart(_candles(), output, [_trade()], "January BTC/USDT")
    content = output.read_text(encoding="utf-8")
    assert summary.candle_count == 2
    assert summary.trade_count == 1
    assert summary.first_open == datetime(2024, 1, 1, tzinfo=UTC)
    assert summary.last_close == datetime(2024, 1, 1, 2, tzinfo=UTC)
    assert "January BTC" in content and "USDT" in content
    assert "candlestick" in content.lower()
    assert "plotly.js v" in content
    assert '"Entry"' in content and '"Exit"' in content


def test_chart_requires_html_output_and_existing_file_to_open(tmp_path: Path) -> None:
    with pytest.raises(ChartError, match=".html"):
        build_market_chart(_candles(), tmp_path / "market.json")
    with pytest.raises(ChartError, match="does not exist"):
        open_market_chart(tmp_path / "missing.html")


def test_open_market_chart_uses_local_file_uri(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "market.html"
    build_market_chart(_candles(), output)
    opened: list[str] = []
    monkeypatch.setattr(
        "trading_bot.charting.market_chart.webbrowser.open",
        lambda uri: opened.append(uri) or True,
    )
    open_market_chart(output)
    assert opened == [output.resolve().as_uri()]


def test_chart_cli_validates_checksum_and_writes_html(tmp_path: Path) -> None:
    input_path = tmp_path / "btc.csv"
    output = tmp_path / "market.html"
    candles = write_candles_atomic(input_path, _candles())
    write_metadata_atomic(metadata_path(input_path), {"csv_sha256": csv_sha256(input_path)})
    assert main(["chart-data", "--input", str(input_path), "--output", str(output)]) == 0
    assert output.is_file()
    write_metadata_atomic(metadata_path(input_path), {"csv_sha256": "0" * 64})
    assert main(["chart-data", "--input", str(input_path), "--output", str(output)]) == 1
    assert candles == _candles()
