from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import re
import sys
from dataclasses import asdict
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from trading_bot.backtest.engine import CandleBacktester
from trading_bot.data.binance import BinancePublicClient
from trading_bot.data.binance_historical import BinanceDataError, BinanceHistoricalDataClient
from trading_bot.data.binance_vision import BinanceVisionError, BinanceVisionHistoricalClient
from trading_bot.data.csv_store import (
    CsvDataError,
    contains_non_tradable_intervals,
    metadata_path,
    read_candles,
    verified_missing_open_times,
    verify_metadata_checksum,
    write_json_atomic,
)
from trading_bot.data.historical import (
    DataCoverageError,
    download_historical_csv,
    download_vision_historical_csv,
    summary_json,
)
from trading_bot.data.validation import CandleValidationError, validate_candles
from trading_bot.domain.models import Candle, Trade
from trading_bot.ml.baseline import BaselineTrainingError, train_logistic_baseline
from trading_bot.ml.dataset import DatasetBuildError, build_ml_dataset
from trading_bot.monitoring.health import create_app
from trading_bot.recommendations.engine import (
    RecommendationEngine,
    RecommendationError,
    RecommendationHistoryProvenance,
    RecommendationHistoryStore,
    accuracy_report,
    backfill_recommendations,
    evaluate_outcomes,
    merge_outcomes,
    merge_recommendations,
    outcome_json,
    recommendation_json,
    validate_strict_oos_history,
)
from trading_bot.recommendations.research import ResearchFreezeError, freeze_development_dataset
from trading_bot.runtime.dry_run import (
    DryRunEngine,
    DryRunError,
    DryRunService,
    DryRunStateStore,
    replay,
)
from trading_bot.settings import BotSettings, load_settings

_DATE_ONLY = re.compile(r"\d{4}-\d{2}-\d{2}\Z")


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
    if _DATE_ONLY.fullmatch(value):
        try:
            return datetime.combine(date.fromisoformat(value), time.min, tzinfo=UTC)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid UTC date: {value}") from exc
    try:
        parsed = datetime.fromisoformat(f"{value[:-1]}+00:00" if value.endswith("Z") else value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid UTC timestamp: {value}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError(f"timestamp must include a timezone offset: {value}")
    return parsed.astimezone(UTC)


def parse_strict_utc(value: str) -> datetime:
    """Parse an OOS boundary explicitly expressed in UTC."""

    if _DATE_ONLY.fullmatch(value):
        return parse_utc(value)
    try:
        parsed = datetime.fromisoformat(f"{value[:-1]}+00:00" if value.endswith("Z") else value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid UTC timestamp: {value}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise argparse.ArgumentTypeError(f"timestamp must be explicitly UTC: {value}")
    return parsed.astimezone(UTC)


def parse_download_end(value: str) -> datetime | None:
    return None if value == "now" else parse_utc(value)


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
    allowed = (
        "symbol",
        "timeframe",
        "starting_cash",
        "entry_fee_rate",
        "exit_fee_rate",
        "entry_slippage_rate",
        "exit_slippage_rate",
        "risk_per_trade",
        "max_exposure",
        "max_daily_loss",
        "max_drawdown",
        "consecutive_loss_limit",
        "cooldown_hours",
        "min_notional",
        "quantity_step",
        "ema_fast",
        "ema_slow",
        "ema_trend",
        "volume_window",
        "volume_multiplier",
        "atr_window",
        "stop_atr_multiple",
        "trailing_atr_multiple",
        "ml_filter_enabled",
        "model_version",
    )
    snapshot = {key: _setting_value(getattr(settings, key)) for key in allowed}
    if any(_is_sensitive_setting_name(key) for key in snapshot):
        raise RuntimeError("unsafe setting requested for backtest report")
    return snapshot


def _setting_value(value: Any) -> Any:
    return str(value) if isinstance(value, Decimal) else value


def _is_sensitive_setting_name(key: str) -> bool:
    return any(
        fragment in key.lower()
        for fragment in ("secret", "password", "token", "api_key", "credential")
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
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
    download.add_argument("--source", choices=("rest", "binance-vision"), default="rest")
    download.add_argument("--start", type=parse_utc, required=True)
    download.add_argument("--end", type=parse_download_end, required=True)
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
    dataset = subcommands.add_parser("build-dataset")
    dataset.add_argument("--input", type=Path, required=True)
    dataset.add_argument("--output-dir", type=Path, required=True)
    train = subcommands.add_parser("train")
    train.add_argument("--dataset-dir", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    for name in ("evaluate", "walk-forward", "report"):
        subcommands.add_parser(name)
    dry_run = subcommands.add_parser("dry-run")
    source = dry_run.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--replay", type=Path, help="validated closed-candle CSV for deterministic paper replay"
    )
    source.add_argument(
        "--public", action="store_true", help="public Binance REST/WebSocket market data"
    )
    dry_run.add_argument("--state", type=Path, default=Path("data/dry_run/btcusdt_1h.state.json"))
    dry_run.add_argument("--max-candles", type=int, default=None)
    dry_run.add_argument("--health-port", type=int, help="optional local health endpoint port")
    recommend = subcommands.add_parser("recommend")
    recommend.add_argument("--input", type=Path, required=True)
    recommend.add_argument("--output", type=Path, required=True)
    recommend.add_argument("--history", type=Path)
    backfill_recommendations_parser = subcommands.add_parser("backfill-recommendations")
    backfill_recommendations_parser.add_argument("--input", type=Path, required=True)
    backfill_recommendations_parser.add_argument(
        "--output", type=Path, required=True, help="atomic recommendation history JSON"
    )
    backfill_recommendations_parser.add_argument(
        "--evaluation-start",
        type=parse_strict_utc,
        help="UTC candle close time; earlier candles warm features but are not persisted",
    )
    evaluate_recommendations = subcommands.add_parser("evaluate-recommendations")
    evaluate_recommendations.add_argument("--input", type=Path, required=True)
    evaluate_recommendations.add_argument("--output", type=Path, required=True)
    freeze_research = subcommands.add_parser("freeze-recommendation-research")
    freeze_research.add_argument("--input", type=Path, required=True)
    freeze_research.add_argument("--output", type=Path, required=True)
    freeze_research.add_argument("--overwrite", action="store_true")
    return parser


def _ensure_market(symbol: str, timeframe: str) -> None:
    if symbol != "BTCUSDT" or timeframe != "1h":
        raise ValueError("only Binance Spot BTCUSDT 1h is supported")


def _download(args: argparse.Namespace, settings: BotSettings) -> None:
    _ensure_market(args.symbol, args.timeframe)
    end = args.end or datetime.now(UTC).replace(minute=0, second=0, microsecond=0) + timedelta(
        hours=1
    )
    if end <= args.start:
        raise ValueError("--end must be after --start")
    rest = BinanceHistoricalDataClient(
        base_url=settings.binance_public_base_url,
        timeout_seconds=settings.http_timeout_seconds,
        max_retries=settings.http_max_retries,
    )
    if args.source == "binance-vision":
        client = BinanceVisionHistoricalClient(
            Path("data/archive_cache"),
            rest,
            max_retries=settings.http_max_retries,
        )
        summary = asyncio.run(
            download_vision_historical_csv(client, args.start, end, args.output, args.overwrite)
        )
        payload = summary_json(summary)
        if not verify_metadata_checksum(args.output):
            raise DataCoverageError("Vision publisher did not create verified metadata")
        try:
            metadata = json.loads(metadata_path(args.output).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DataCoverageError("could not read verified Vision metadata") from exc
        payload["vision_audit"] = {
            key: metadata[key]
            for key in (
                "archive_candle_count",
                "exact_archive_timestamp_candle_count",
                "accepted_archive_anomaly_count",
                "market_interruption_event_count",
                "market_interruption_candle_count",
                "missing_candle_count",
                "contains_non_tradable_intervals",
                "rest_suffix_candle_count",
                "duplicate_candle_count",
                "conflicting_candle_count",
            )
        }
    else:
        summary = asyncio.run(
            download_historical_csv(rest, args.start, end, args.output, args.overwrite)
        )
        payload = summary_json(summary)
    print(json.dumps(payload, indent=2))


def _validate(args: argparse.Namespace) -> None:
    metadata_verified = verify_metadata_checksum(args.input)
    missing = verified_missing_open_times(args.input) if metadata_verified else set()
    candles = read_candles(args.input, allowed_missing_open_times=missing)
    if not metadata_verified:
        print("warning: metadata checksum sidecar is missing", file=sys.stderr)
    max_age = timedelta(hours=args.max_age_hours) if args.max_age_hours is not None else None
    validate_candles(candles, max_age=max_age, allowed_missing_open_times=missing)
    print(
        json.dumps(
            {
                "status": (
                    "valid_with_market_interruptions"
                    if contains_non_tradable_intervals(args.input)
                    else "valid"
                ),
                "candle_count": len(candles),
                "first_open": candles[0].open_time.isoformat(),
                "last_close": candles[-1].close_time.isoformat(),
                "input": str(args.input),
            },
            indent=2,
        )
    )


def _backtest(args: argparse.Namespace, settings: BotSettings) -> None:
    if not args.fixture and contains_non_tradable_intervals(args.input):
        raise DataCoverageError(
            "backtest refuses datasets containing non-tradable market interruption intervals"
        )
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


def _build_dataset(args: argparse.Namespace) -> None:
    summary = build_ml_dataset(args.input, args.output_dir)
    print(
        json.dumps(
            {
                "dataset_generation_id": summary.generation_id,
                "source_generation_id": summary.source_generation_id,
                "source_candle_count": summary.source_candle_count,
                "segment_count": summary.segment_count,
                "candidate_counts": summary.candidate_counts,
                "development_label_counts": summary.label_counts,
                "development_trainable": summary.trainable_splits,
                "candidate_policy": _json_safe(asdict(summary.candidate_policy)),
                "label_policy": _json_safe(asdict(summary.label_policy)),
                "excluded_row_counts": summary.exclusion_counts,
                "output_file_sha256": summary.output_checksums,
            },
            indent=2,
        )
    )


def _train(args: argparse.Namespace) -> None:
    summary = train_logistic_baseline(args.dataset_dir, args.output_dir)
    print(
        json.dumps(
            {
                "dataset_generation_id": summary.dataset_generation_id,
                "source_generation_id": summary.source_generation_id,
                "threshold": summary.threshold,
                "validation_metrics": summary.validation_metrics,
                "test_metrics": summary.test_metrics,
                "output_dir": str(summary.output_dir),
                "experimental_only": True,
                "production_eligible": False,
                "live_trading_enabled": False,
            },
            indent=2,
        )
    )


def _dry_run_settings(settings: BotSettings) -> BotSettings:
    if settings.ml_filter_enabled:
        raise ValueError("dry-run ML filtering is not enabled by this command")
    values = settings.model_dump()
    values["bot_mode"] = "dry_run"
    return BotSettings.model_validate(values)


async def _public_dry_run(
    service: DryRunService,
    settings: BotSettings,
    max_candles: int | None,
    health_port: int | None,
) -> int:
    server: Any | None = None
    server_task: asyncio.Task[None] | None = None
    if health_port is not None:
        try:
            import uvicorn
        except ImportError as exc:
            raise ValueError("install the api extra to serve dry-run health endpoints") from exc
        server = uvicorn.Server(
            uvicorn.Config(create_app(service.engine.status), host="127.0.0.1", port=health_port)
        )
        server_task = asyncio.create_task(server.serve())
    await service.bootstrap()
    client = BinancePublicClient(
        BinanceHistoricalDataClient(
            base_url=settings.binance_public_base_url,
            timeout_seconds=settings.http_timeout_seconds,
            max_retries=settings.http_max_retries,
        )
    )
    stream = client.closed_klines(
        on_connected=lambda: setattr(service.engine.health, "websocket_connected", True),
        on_disconnected=lambda: setattr(service.engine.health, "websocket_connected", False),
    )
    try:
        return await service.consume(stream, max_candles)
    finally:
        service.engine.health.websocket_connected = False
        service.engine.persist()
        if server is not None:
            server.should_exit = True
        if server_task is not None:
            await server_task


def _dry_run(args: argparse.Namespace, settings: BotSettings) -> None:
    if args.max_candles is not None and args.max_candles <= 0:
        raise ValueError("--max-candles must be positive")
    if args.health_port is not None and not 1 <= args.health_port <= 65_535:
        raise ValueError("--health-port must be between 1 and 65535")
    if args.replay is not None and args.health_port is not None:
        raise ValueError("--health-port is available only with --public")
    paper_settings = _dry_run_settings(settings)
    engine = DryRunEngine(paper_settings, DryRunStateStore(args.state))
    rest = BinanceHistoricalDataClient(
        base_url=paper_settings.binance_public_base_url,
        timeout_seconds=paper_settings.http_timeout_seconds,
        max_retries=paper_settings.http_max_retries,
    )
    service = DryRunService(engine, rest)
    if args.replay is not None:
        candles = read_candles(args.replay)
        validate_candles(candles)
        processed = asyncio.run(replay(service, candles))
    else:
        processed = asyncio.run(
            _public_dry_run(service, paper_settings, args.max_candles, args.health_port)
        )
    print(
        json.dumps(
            {
                "mode": "dry_run",
                "processed_closed_candles": processed,
                "state_file": str(args.state),
                "health": engine.status(),
                "safety_locks": {
                    "broker": "DryRunBroker",
                    "live_trading_enabled": False,
                    "binance_orders_sent": False,
                    "ml_filter_enabled": False,
                },
            },
            indent=2,
        )
    )


def _recommend(args: argparse.Namespace, settings: BotSettings) -> None:
    metadata_verified = verify_metadata_checksum(args.input)
    missing = verified_missing_open_times(args.input) if metadata_verified else set()
    candles = read_candles(args.input, allowed_missing_open_times=missing)
    validate_candles(candles, allowed_missing_open_times=missing)
    report = RecommendationEngine(settings).recommend(candles, missing)
    history_path = args.history or args.output.with_name("history.json")
    store = RecommendationHistoryStore(history_path)
    existing, existing_outcomes, provenance, _ = store.load_with_provenance()
    if provenance is not None and provenance.strict_oos:
        raise RecommendationError("locked OOS history only accepts backfill with its boundary")
    recommendations = merge_recommendations(existing, [report.recommendation])
    available_close_times = {candle.close_time for candle in candles}
    resolvable = [
        item for item in recommendations if item.signal_candle_time in available_close_times
    ]
    outcomes = merge_outcomes(
        existing_outcomes, evaluate_outcomes(resolvable, candles, settings, missing)
    )
    store.save(recommendations, outcomes, RecommendationHistoryProvenance(False))
    payload = {
        "schema_version": "1.0",
        "mode": "recommendation_only",
        "recommendation": recommendation_json(report.recommendation),
        "feature_count": report.feature_count,
        "history_file": str(history_path),
        "outcomes": [
            outcome_json(item)
            for item in outcomes
            if item.recommendation_id == report.recommendation.id
        ],
        "safety_locks": {
            "live_trading_enabled": False,
            "broker_used": False,
            "binance_orders_sent": False,
            "ml_probability_used": report.recommendation.probability_up is not None,
        },
    }
    write_json_atomic(args.output, payload)
    print(json.dumps(payload, indent=2))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _backfill_provenance(
    args: argparse.Namespace, candles: list[Candle]
) -> RecommendationHistoryProvenance:
    if args.evaluation_start is None:
        return RecommendationHistoryProvenance(False)
    return RecommendationHistoryProvenance(
        True,
        args.evaluation_start,
        _file_sha256(args.input),
        candles[0].close_time,
        candles[-1].close_time,
    )


def _backfill_recommendations(args: argparse.Namespace, settings: BotSettings) -> None:
    metadata_verified = verify_metadata_checksum(args.input)
    missing = verified_missing_open_times(args.input) if metadata_verified else set()
    candles = read_candles(args.input, allowed_missing_open_times=missing)
    validate_candles(candles, allowed_missing_open_times=missing)
    if args.evaluation_start is not None and args.evaluation_start not in {
        candle.close_time for candle in candles
    }:
        raise ValueError("--evaluation-start must equal a closed candle timestamp in --input")
    store = RecommendationHistoryStore(args.output)
    existing, existing_outcomes, persisted_provenance, legacy = store.load_with_provenance()
    requested_provenance = _backfill_provenance(args, candles)
    if persisted_provenance is not None and persisted_provenance.strict_oos:
        if args.evaluation_start is None:
            raise RecommendationError("strict OOS history requires --evaluation-start")
        if persisted_provenance != requested_provenance:
            raise RecommendationError("history provenance is locked; use a new output path")
        validate_strict_oos_history(existing, persisted_provenance)
    elif existing or existing_outcomes:
        if legacy and args.evaluation_start is not None:
            raise RecommendationError("legacy history cannot be used as strict OOS evidence")
        if persisted_provenance is not None and persisted_provenance != requested_provenance:
            raise RecommendationError("history provenance is locked; use a new output path")
    elif legacy and args.evaluation_start is not None:
        raise RecommendationError("legacy history cannot be used as strict OOS evidence")
    generated = backfill_recommendations(RecommendationEngine(settings), candles, missing)
    if args.evaluation_start is not None:
        generated = [item for item in generated if item.signal_candle_time >= args.evaluation_start]
    recommendations = merge_recommendations(existing, generated)
    available_close_times = {candle.close_time for candle in candles}
    resolvable = [
        item for item in recommendations if item.signal_candle_time in available_close_times
    ]
    outcomes = merge_outcomes(
        existing_outcomes, evaluate_outcomes(resolvable, candles, settings, missing)
    )
    store.save(recommendations, outcomes, requested_provenance)
    payload = {
        "schema_version": "1.1",
        "mode": "recommendation_backfill",
        "history_file": str(args.output),
        "generated_recommendations": len(generated),
        "stored_recommendations": len(recommendations),
        "stored_outcomes": len(outcomes),
        "provenance": {
            "strict_oos": requested_provenance.strict_oos,
            "evaluation_start": (
                args.evaluation_start.astimezone(UTC).isoformat().replace("+00:00", "Z")
                if args.evaluation_start is not None
                else None
            ),
            "input_sha256": requested_provenance.input_sha256,
        },
        "safety_locks": {
            "live_trading_enabled": False,
            "broker_used": False,
            "binance_orders_sent": False,
        },
    }
    print(json.dumps(payload, indent=2))


def _evaluate_recommendations(args: argparse.Namespace) -> None:
    recommendations, outcomes, provenance, legacy = RecommendationHistoryStore(
        args.input
    ).load_with_provenance()
    validate_strict_oos_history(recommendations, provenance)
    payload = accuracy_report(recommendations, outcomes, provenance)
    payload.update(
        {
            "schema_version": "1.2",
            "input": str(args.input),
            "strict_oos": bool(provenance is not None and provenance.strict_oos),
            "history_provenance": {
                "legacy": legacy,
                "evaluation_start": (
                    provenance.evaluation_start.astimezone(UTC).isoformat().replace("+00:00", "Z")
                    if provenance is not None and provenance.evaluation_start is not None
                    else None
                ),
                "input_sha256": provenance.input_sha256 if provenance is not None else None,
                "input_first_close": (
                    provenance.input_first_close.astimezone(UTC).isoformat().replace("+00:00", "Z")
                    if provenance is not None and provenance.input_first_close is not None
                    else None
                ),
                "input_last_close": (
                    provenance.input_last_close.astimezone(UTC).isoformat().replace("+00:00", "Z")
                    if provenance is not None and provenance.input_last_close is not None
                    else None
                ),
            },
        }
    )
    write_json_atomic(args.output, payload)
    print(json.dumps(payload, indent=2))


def _freeze_recommendation_research(args: argparse.Namespace) -> None:
    manifest = freeze_development_dataset(args.input, args.output, overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "schema_version": manifest["schema_version"],
                "research_role": manifest["research_role"],
                "candle_count": manifest["dataset"]["candle_count"],
                "validation_status": manifest["dataset"]["validation_status"],
                "market_interruption_count": len(manifest["market_interruptions"]),
                "safety_locks": manifest["safety_locks"],
            },
            indent=2,
        )
    )


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
        elif args.command == "build-dataset":
            _build_dataset(args)
        elif args.command == "train":
            _train(args)
        elif args.command == "dry-run":
            _dry_run(args, settings)
        elif args.command == "recommend":
            _recommend(args, settings)
        elif args.command == "backfill-recommendations":
            _backfill_recommendations(args, settings)
        elif args.command == "evaluate-recommendations":
            _evaluate_recommendations(args)
        elif args.command == "freeze-recommendation-research":
            _freeze_recommendation_research(args)
        else:
            print(f"{args.command}: no live activity performed.")
    except (
        BinanceDataError,
        BinanceVisionError,
        CandleValidationError,
        CsvDataError,
        DataCoverageError,
        ResearchFreezeError,
        DatasetBuildError,
        BaselineTrainingError,
        DryRunError,
        RecommendationError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0
