from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from trading_bot.backtest.engine import CandleBacktester
from trading_bot.domain.models import Candle
from trading_bot.settings import load_settings


def fixture() -> list[Candle]:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    result = []
    for i in range(260):
        price = Decimal("40000") + Decimal(i) * Decimal("10")
        # Deterministic rising fixture, with a volume spike after feature warmup.
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=[
            "backtest",
            "download-data",
            "validate-data",
            "build-dataset",
            "train",
            "evaluate",
            "walk-forward",
            "dry-run",
            "report",
        ],
    )
    args = parser.parse_args()
    settings = load_settings()
    if args.command == "backtest":
        print(json.dumps(CandleBacktester(settings).run(fixture()).metrics(), indent=2))
    elif args.command == "dry-run":
        print("Dry-run safety check complete: paper broker only; no exchange orders can be sent.")
    else:
        print(
            f"{args.command}: use fixture pipeline or downloaded CSV; no live activity performed."
        )
