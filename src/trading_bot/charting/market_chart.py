from __future__ import annotations

import webbrowser
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from trading_bot.data.validation import validate_candles
from trading_bot.domain.models import Candle, Trade


class ChartError(ValueError):
    """A market chart could not be built or opened safely."""


@dataclass(frozen=True, slots=True)
class ChartSummary:
    output: Path
    candle_count: int
    first_open: datetime
    last_close: datetime
    trade_count: int

    def as_dict(self) -> dict[str, str | int]:
        return {
            "output": str(self.output),
            "candle_count": self.candle_count,
            "first_open": _utc_iso(self.first_open),
            "last_close": _utc_iso(self.last_close),
            "trade_count": self.trade_count,
        }


def build_market_chart(
    candles: list[Candle],
    output: Path,
    trades: list[Trade] | None = None,
    title: str | None = None,
) -> ChartSummary:
    """Write a self-contained interactive candlestick chart for validated candles."""
    validate_candles(candles)
    if output.suffix.lower() != ".html":
        raise ChartError("chart output must be an .html file")
    selected_trades = trades or []
    figure = _figure(candles, selected_trades, title)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        str(output),
        config={"displaylogo": False, "responsive": True},
        full_html=True,
        include_plotlyjs=True,
    )
    return ChartSummary(
        output=output,
        candle_count=len(candles),
        first_open=candles[0].open_time,
        last_close=candles[-1].close_time,
        trade_count=len(selected_trades),
    )


def open_market_chart(output: Path) -> None:
    """Open an already-generated chart through the operating system's browser handler."""
    if not output.is_file():
        raise ChartError(f"chart output does not exist: {output}")
    if not webbrowser.open(output.resolve().as_uri()):
        raise ChartError("could not open chart in a browser")


def _figure(candles: list[Candle], trades: list[Trade], title: str | None) -> go.Figure:
    times = [candle.open_time for candle in candles]
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.78, 0.22],
    )
    figure.add_trace(
        go.Candlestick(
            x=times,
            open=[float(candle.open) for candle in candles],
            high=[float(candle.high) for candle in candles],
            low=[float(candle.low) for candle in candles],
            close=[float(candle.close) for candle in candles],
            name="BTC/USDT",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=times,
            y=[float(candle.volume) for candle in candles],
            name="Volume",
            marker_color="#4C78A8",
        ),
        row=2,
        col=1,
    )
    if trades:
        figure.add_trace(
            go.Scatter(
                x=[trade.entry_time for trade in trades],
                y=[float(trade.entry_price) for trade in trades],
                mode="markers",
                name="Entry",
                marker={"color": "#2CA02C", "size": 10, "symbol": "triangle-up"},
                hovertemplate="Entry %{x}<br>Price %{y}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=[trade.exit_time for trade in trades],
                y=[float(trade.exit_price) for trade in trades],
                mode="markers",
                name="Exit",
                marker={"color": "#D62728", "size": 10, "symbol": "triangle-down"},
                hovertemplate="Exit %{x}<br>Price %{y}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    figure.update_layout(
        title=title or "BTC/USDT Spot 1h",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        template="plotly_white",
        legend={"orientation": "h", "y": 1.02, "x": 0},
        margin={"l": 60, "r": 30, "t": 80, "b": 45},
    )
    figure.update_yaxes(title_text="Price (USDT)", row=1, col=1)
    figure.update_yaxes(title_text="Volume", row=2, col=1)
    figure.update_xaxes(title_text="UTC", row=2, col=1)
    return figure


def _utc_iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
