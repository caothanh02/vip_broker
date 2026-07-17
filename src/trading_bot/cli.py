from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from trading_bot.backtest.engine import CandleBacktester
from trading_bot.data.binance_historical import BinanceDataError, BinanceHistoricalDataClient
from trading_bot.data.csv_store import CsvDataError, read_candles, write_json_atomic
from trading_bot.data.historical import download_historical_csv, summary_json
from trading_bot.data.validation import CandleValidationError, validate_candles
from trading_bot.domain.models import Candle, Trade
from trading_bot.settings import BotSettings, load_settings


def fixture() -> list[Candle]:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    result = []
    for i in range(260):
        price = Decimal("40000") + Decimal(i) * Decimal("10")
        result.append(
            Candle(
                base + timedelta(hours=i),
                base + timedelta(hours=i + 1),
                "BTC/USDT",
                "1h",
                price,
                price + 20,
                price - 20,
                price + 5,
                Decimal("1000") if i != 220 else Decimal("5000"),
                True,
            )
        )
    return result


def parse_utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid UTC timestamp: {value}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _trade_json(trade: Trade) -> dict[str, str]:
    return {
        "symbol": trade.symbol,
        "quantity": str(trade.quantity),
        "entry_price": str(trade.entry_price),
        "exit_price": str(trade.exit_price),
        "entry_time": trade.entry_time.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "exit_time": trade.exit_time.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "pnl": str(trade.pnl),
        "fees": str(trade.fees),
        "exit_reason": trade.exit_reason,
    }


def _settings_snapshot(settings: BotSettings) -> dict[str, Any]:
    excluded = {"binance_api_key", "binance_api_secret", "live_trading_confirmation"}
    return {
        key: str(value) if isinstance(value, Decimal) else value
        for key, value in settings.model_dump().items()
        if key not in excluded
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trading-bot")
    subcommands = parser.add_subparsers(dest="command", required=True)
    download = subcommands.add_parser("download-data")
    download.add_argument("--symbol", default="BTCUSDT")
    download.add_argument("--timeframe", default="1h")
    download.add_argument("--start", type=parse_utc, required=True)
    download.add_argument("--end", type=parse_utc, required=True)
    download.add_argument("--output", type=Path, required=True)
    download.add_argument("--overwrite", action="store_true")
    validate = subcommands.add_parser("validate-data")
    validate.add_argument("--input", type=Path, required=True)
    validate.add_argument("--max-age-hours", type=float)
    backtest = subcommands.add_parser("backtest")
    source = backtest.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path)
    source.add_argument("--fixture", action="store_true")
    backtest.add_argument("--output", type=Path, required=True)
    for name in ("build-dataset", "train", "evaluate", "walk-forward", "dry-run", "report"):
        subcommands.add_parser(name)
    return parser


def _ensure_market(symbol: str, timeframe: str) -> None:
    if symbol != "BTCUSDT" or timeframe != "1h":
        raise ValueError("only Binance Spot BTCUSDT 1h is supported")


def _download(args: argparse.Namespace, settings: BotSettings) -> None:
    _ensure_market(args.symbol, args.timeframe)
    if args.end <= args.start:
        raise ValueError("--end must be after --start")
    client = BinanceHistoricalDataClient(
        base_url=settings.binance_public_base_url,
        timeout_seconds=settings.http_timeout_seconds,
        max_retries=settings.http_max_retries,
    )
    summary = asyncio.run(
        download_historical_csv(client, args.start, args.end, args.output, args.overwrite)
    )
    print(json.dumps(summary_json(summary), indent=2))


def _validate(args: argparse.Namespace) -> None:
    candles = read_candles(args.input)
    max_age = timedelta(hours=args.max_age_hours) if args.max_age_hours is not None else None
    validate_candles(candles, max_age=max_age)
    print(
        json.dumps(
            {
                "status": "valid",
                "candle_count": len(candles),
                "first_open": candles[0].open_time.isoformat(),
                "last_close": candles[-1].close_time.isoformat(),
                "input": str(args.input),
            },
            indent=2,
        )
    )


def _backtest(args: argparse.Namespace, settings: BotSettings) -> None:
    candles = fixture() if args.fixture else read_candles(args.input)
    validate_candles(candles)
    result = CandleBacktester(settings).run(candles)
    report = {
        "run_metadata": {"created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z")},
        "settings": _settings_snapshot(settings),
        "input_file": "fixture" if args.fixture else str(args.input),
        "data_range": {
            "first_open": candles[0].open_time.isoformat().replace("+00:00", "Z"),
            "last_close": candles[-1].close_time.isoformat().replace("+00:00", "Z"),
        },
        "candle_count": len(candles),
        "metrics": result.metrics(),
        "trades": [_trade_json(trade) for trade in result.trades],
    }
    write_json_atomic(args.output, _json_safe(report))
    print(json.dumps({"output": str(args.output), "metrics": result.metrics()}, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = load_settings()
    try:
        if args.command == "download-data":
            _download(args, settings)
        elif args.command == "validate-data":
            _validate(args)
        elif args.command == "backtest":
            _backtest(args, settings)
        elif args.command == "dry-run":
            print(
                "Dry-run safety check complete: paper broker only; no exchange orders can be sent."
            )
        else:
            print(f"{args.command}: no live activity performed.")
    except (BinanceDataError, CandleValidationError, CsvDataError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0
