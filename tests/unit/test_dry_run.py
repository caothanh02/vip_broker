from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from trading_bot.cli import main
from trading_bot.data.csv_store import write_candles_atomic
from trading_bot.domain.models import Candle, Position
from trading_bot.execution.broker import DryRunBroker
from trading_bot.risk.engine import RiskState
from trading_bot.runtime.dry_run import (
    DryRunEngine,
    DryRunError,
    DryRunService,
    DryRunState,
    DryRunStateStore,
    replay,
)
from trading_bot.settings import BotSettings

BASE = datetime(2024, 1, 1, tzinfo=UTC)


def candle(index: int, closed: bool = True) -> Candle:
    price = Decimal("40000") + Decimal(index)
    return Candle(
        BASE + index * timedelta(hours=1),
        BASE + (index + 1) * timedelta(hours=1),
        "BTC/USDT",
        "1h",
        price,
        price + 10,
        price - 10,
        price + 2,
        Decimal("1000"),
        closed,
    )


class FakeRest:
    def __init__(self, responses: dict[tuple[datetime, datetime], list[Candle]]) -> None:
        self.responses = responses
        self.calls: list[tuple[datetime, datetime]] = []

    async def fetch_closed(self, start: datetime, end: datetime) -> list[Candle]:
        self.calls.append((start, end))
        return self.responses[(start, end)]


def engine(tmp_path: Path) -> DryRunEngine:
    return DryRunEngine(BotSettings(bot_mode="dry_run"), DryRunStateStore(tmp_path / "state.json"))


def test_replay_uses_paper_broker_and_persists_resume_state(tmp_path: Path) -> None:
    first = engine(tmp_path)
    assert isinstance(first.broker, DryRunBroker)
    service = DryRunService(first, FakeRest({}))
    assert asyncio.run(replay(service, [candle(0), candle(1)])) == 2
    assert first.status()["status"] == "ok"
    state_path = tmp_path / "state.json"
    raw = json.loads(state_path.read_text(encoding="utf-8"))
    assert raw["mode"] == "dry_run"
    assert raw["risk"]["circuit_open"] is False

    restored = engine(tmp_path)
    assert restored.state.cash == first.state.cash
    assert [item.open_time for item in restored.state.history] == [
        candle(0).open_time,
        candle(1).open_time,
    ]
    assert restored.status()["last_closed_candle"] == "2024-01-01T02:00:00Z"


def test_state_restores_position_and_risk_circuit_breaker(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    store = DryRunStateStore(path)
    position = Position(
        "BTC/USDT",
        Decimal("0.1"),
        Decimal("40000"),
        Decimal("39000"),
        Decimal("40500"),
        BASE,
        Decimal("4"),
    )
    state = DryRunState(
        Decimal("5996"),
        RiskState(
            Decimal("10000"),
            Decimal("10000"),
            daily_pnl=Decimal("-10"),
            daily_pnl_date=BASE.date(),
            circuit_open=True,
        ),
        position=position,
        history=[candle(0)],
    )
    store.save(state)
    restored = store.load(BotSettings(bot_mode="dry_run"))
    assert restored is not None
    assert restored.position == position
    assert restored.risk.circuit_open
    assert restored.cash == Decimal("5996")


def test_rejects_open_or_gapped_candles_without_processing(tmp_path: Path) -> None:
    runner = engine(tmp_path)
    runner.process(candle(0))
    with pytest.raises(DryRunError, match="open candle"):
        runner.process(candle(1, closed=False))
    with pytest.raises(DryRunError, match="gap"):
        runner.process(candle(2))
    assert [item.open_time for item in runner.state.history] == [candle(0).open_time]


def test_gap_is_recovered_through_rest_before_websocket_candle(tmp_path: Path) -> None:
    runner = engine(tmp_path)
    runner.process(candle(0))
    rest = FakeRest({(candle(1).open_time, candle(2).open_time): [candle(1)]})
    service = DryRunService(runner, rest)

    async def stream() -> AsyncIterator[Candle]:
        yield candle(2)

    assert asyncio.run(service.consume(stream())) == 1
    assert rest.calls == [(candle(1).open_time, candle(2).open_time)]
    assert [item.open_time for item in runner.state.history] == [
        candle(0).open_time,
        candle(1).open_time,
        candle(2).open_time,
    ]
    assert runner.status()["websocket_connected"] is True


def test_gap_recovery_fails_closed_when_rest_coverage_is_incomplete(tmp_path: Path) -> None:
    runner = engine(tmp_path)
    runner.process(candle(0))
    rest = FakeRest({(candle(1).open_time, candle(2).open_time): []})
    service = DryRunService(runner, rest)

    async def stream() -> AsyncIterator[Candle]:
        yield candle(2)

    with pytest.raises(DryRunError, match="returned no closed candles"):
        asyncio.run(service.consume(stream()))
    assert [item.open_time for item in runner.state.history] == [candle(0).open_time]


def test_bootstrap_warms_from_closed_rest_candles(tmp_path: Path) -> None:
    runner = engine(tmp_path)
    end = BASE + _hours(300)
    candles = [candle(index) for index in range(60, 300)]
    rest = FakeRest({(BASE + _hours(60), end): candles})
    service = DryRunService(runner, rest, now=lambda: end + timedelta(minutes=15))
    asyncio.run(service.bootstrap())
    assert len(runner.state.history) == 240
    assert runner.state.history[-1].close_time == end


def test_invalid_persisted_state_fails_closed(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text('{"state_version": 1, "mode": "live"}', encoding="utf-8")
    with pytest.raises(DryRunError, match="market or mode"):
        DryRunStateStore(state_path).load(BotSettings(bot_mode="dry_run"))


def test_cli_replay_runs_only_paper_state(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    replay_file = tmp_path / "replay.csv"
    state_file = tmp_path / "state.json"
    write_candles_atomic(replay_file, [candle(0), candle(1)])
    assert main(["dry-run", "--replay", str(replay_file), "--state", str(state_file)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["safety_locks"] == {
        "broker": "DryRunBroker",
        "live_trading_enabled": False,
        "binance_orders_sent": False,
        "ml_filter_enabled": False,
    }
    assert state_file.exists()


def _hours(value: int) -> timedelta:
    return timedelta(hours=value)
