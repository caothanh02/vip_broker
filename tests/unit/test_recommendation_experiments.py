from __future__ import annotations

import asyncio
import inspect
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from trading_bot.cli import main
from trading_bot.data.binance_vision import _verify_anomaly_continuity, parse_verified_archive_kline
from trading_bot.data.historical import download_vision_historical_csv
from trading_bot.data.market_interruptions import KNOWN_MARKET_INTERRUPTIONS
from trading_bot.domain.models import Candle
from trading_bot.recommendations import experiments
from trading_bot.recommendations.engine import RecommendationEngine, backfill_recommendations
from trading_bot.recommendations.experiments import (
    RecommendationExperimentError,
    run_development_experiment,
)
from trading_bot.recommendations.research import ResearchContract, freeze_development_dataset
from trading_bot.settings import BotSettings

EVENT = KNOWN_MARKET_INTERRUPTIONS[0]
BASE = datetime(2023, 3, 24, 11, tzinfo=UTC)


def _manifest_path(name: str) -> Path:
    return Path("reports/research/manifests") / name


def _experiment_path(name: str) -> Path:
    return Path("reports/research/experiments") / name


def _row(opened: int, closed: int) -> list[str]:
    return [str(opened), "100", "101", "99", "100", "1", str(closed)]


def _parsed(hour: int):
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
        _row(opened, opened + 3_599_999),
        archive_name=EVENT.archive_name,
        archive_sha256=EVENT.archive_sha256,
        row_number=hour,
        checksum_verified=True,
    )


class _AuditedClient:
    def __init__(self) -> None:
        self.parsed = _verify_anomaly_continuity(
            [_parsed(0), _parsed(1), _parsed(3)],
            [_parsed(0).candle, _parsed(1).candle, _parsed(3).candle],
        )
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


@pytest.fixture
def frozen_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, ResearchContract]:
    monkeypatch.chdir(tmp_path)
    dataset = Path("data/raw/development.csv")
    asyncio.run(
        download_vision_historical_csv(
            _AuditedClient(), BASE, BASE + timedelta(hours=4), dataset, True
        )
    )
    contract = ResearchContract(
        development_start=BASE,
        development_end=BASE + timedelta(hours=4),
        strict_oos_start=BASE + timedelta(hours=4),
        expected_candle_count=3,
        expected_csv_sha256=experiments.csv_sha256(dataset),
        allowed_interruption_event_id=EVENT.event_id,
        allowed_missing_open_time=BASE + timedelta(hours=2),
    )
    manifest = _manifest_path("development.json")
    freeze_development_dataset(
        dataset,
        manifest,
        overwrite=False,
        contract=contract,
        now=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )
    return dataset, manifest, contract


def _run(manifest: Path, output: Path, *, overwrite: bool = False) -> dict[str, object]:
    return run_development_experiment(
        manifest,
        "baseline_ema_volume_atr_v1",
        output,
        BotSettings(),
        overwrite=overwrite,
        now=lambda: datetime(2026, 1, 2, tzinfo=UTC),
    )


def test_run_development_experiment_is_deterministic_and_never_oos_claim(
    frozen_input: tuple[Path, Path, ResearchContract],
) -> None:
    _, manifest, _ = frozen_input
    output = _experiment_path("baseline.json")

    first = _run(manifest, output)
    second = _run(manifest, output, overwrite=True)

    assert first == second == json.loads(output.read_text(encoding="utf-8"))
    assert first["candidate_id"] == "baseline_ema_volume_atr_v1"
    assert first["research_role"] == "development"
    assert first["strict_oos_evaluation_history"] is False
    assert first["research_claim_eligible"] is False
    assert first["candidate"]["cost_model"] == {
        "entry_fee_rate": "0.001",
        "exit_fee_rate": "0.001",
        "entry_slippage_rate": "0.0005",
        "exit_slippage_rate": "0.0005",
    }
    assert first["cost_model"] == first["candidate"]["cost_model"]
    assert first["safety_locks"] == {
        "live_trading_enabled": False,
        "broker_used": False,
        "orders_submitted": False,
        "ml_inference_used": False,
    }
    metrics = first["metrics"]
    assert isinstance(metrics, dict)
    assert metrics["research_claim_eligible"] is False
    assert all(
        horizon["research_claim_eligible"] is False for horizon in metrics["horizons"].values()
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("research_role", "strict_oos", "development role"),
        ("strict_oos_evaluation_history", True, "strict OOS"),
        ("schema_version", "9.9", "unsupported"),
    ],
)
def test_run_rejects_tampered_or_nondevelopment_manifest_before_dataset_read(
    frozen_input: tuple[Path, Path, ResearchContract],
    field: str,
    value: object,
    message: str,
) -> None:
    _, manifest, _ = frozen_input
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload[field] = value
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RecommendationExperimentError, match=message):
        _run(manifest, _experiment_path("rejected.json"))


def test_run_rejects_unknown_candidate_before_reading_manifest(
    frozen_input: tuple[Path, Path, ResearchContract],
) -> None:
    _, manifest, _ = frozen_input

    with pytest.raises(RecommendationExperimentError, match="unregistered"):
        run_development_experiment(
            manifest,
            "unknown",
            _experiment_path("unknown.json"),
            BotSettings(),
            overwrite=False,
        )


def test_run_rejects_settings_that_do_not_match_predeclared_candidate(
    frozen_input: tuple[Path, Path, ResearchContract],
) -> None:
    _, manifest, _ = frozen_input

    with pytest.raises(RecommendationExperimentError, match="predeclared"):
        run_development_experiment(
            manifest,
            "baseline_ema_volume_atr_v1",
            _experiment_path("settings.json"),
            BotSettings(volume_multiplier=1.3),
            overwrite=False,
        )


@pytest.mark.parametrize(
    "field",
    [
        "entry_fee_rate",
        "exit_fee_rate",
        "entry_slippage_rate",
        "exit_slippage_rate",
    ],
)
def test_run_rejects_cost_rate_changes_before_backfill(
    frozen_input: tuple[Path, Path, ResearchContract],
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    _, manifest, _ = frozen_input
    monkeypatch.setattr(
        experiments,
        "backfill_recommendations",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not backfill")),
    )

    with pytest.raises(RecommendationExperimentError, match="cost rate"):
        run_development_experiment(
            manifest,
            "baseline_ema_volume_atr_v1",
            _experiment_path(f"{field}.json"),
            BotSettings(**{field: Decimal("0.002")}),
            overwrite=False,
        )


def test_manifest_symlink_is_rejected_before_manifest_json_is_read(
    frozen_input: tuple[Path, Path, ResearchContract], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, manifest, _ = frozen_input
    original_is_symlink = Path.is_symlink
    original_read_object = experiments._read_object
    raw_manifest = Path.cwd() / manifest

    def is_symlink(path: Path) -> bool:
        return path == raw_manifest or original_is_symlink(path)

    def reject_manifest_read(path: Path, label: str) -> dict[str, object]:
        if path == raw_manifest:
            raise AssertionError("symlink target manifest must not be read")
        return original_read_object(path, label)

    monkeypatch.setattr(Path, "is_symlink", is_symlink)
    monkeypatch.setattr(experiments, "_read_object", reject_manifest_read)

    with pytest.raises(
        RecommendationExperimentError, match="research manifest must not be a symlink"
    ):
        _run(manifest, _experiment_path("manifest-symlink.json"))


def test_dataset_symlink_is_rejected_before_csv_is_read(
    frozen_input: tuple[Path, Path, ResearchContract], monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, manifest, _ = frozen_input
    original_is_symlink = Path.is_symlink
    original_sha256 = experiments.csv_sha256
    raw_dataset = Path.cwd() / dataset

    def is_symlink(path: Path) -> bool:
        return path == raw_dataset or original_is_symlink(path)

    def reject_dataset_hash(path: Path) -> str:
        if path == raw_dataset:
            raise AssertionError("symlink target CSV must not be read")
        return original_sha256(path)

    monkeypatch.setattr(Path, "is_symlink", is_symlink)
    monkeypatch.setattr(experiments, "csv_sha256", reject_dataset_hash)

    with pytest.raises(RecommendationExperimentError, match="dataset path must not be a symlink"):
        _run(manifest, _experiment_path("dataset-symlink.json"))


def test_absolute_manifest_dataset_path_is_rejected_before_csv_is_read(
    frozen_input: tuple[Path, Path, ResearchContract], monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, manifest, _ = frozen_input
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["dataset"]["path"] = str((Path.cwd() / dataset).resolve())
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    def reject_dataset_hash(path: Path) -> str:
        if path == Path.cwd() / dataset:
            raise AssertionError("absolute-path CSV must not be hashed")
        return "a" * 64

    def reject_dataset_read(*args: object, **kwargs: object) -> list[Candle]:
        raise AssertionError("absolute-path CSV must not be read")

    monkeypatch.setattr(experiments, "csv_sha256", reject_dataset_hash)
    monkeypatch.setattr(experiments, "read_candles", reject_dataset_read)

    with pytest.raises(RecommendationExperimentError, match="dataset path must remain"):
        _run(manifest, _experiment_path("absolute-dataset.json"))


@pytest.mark.parametrize("target", ["csv", "metadata", "anomaly"])
def test_run_rejects_changed_dataset_or_sidecar_checksums(
    frozen_input: tuple[Path, Path, ResearchContract], target: str
) -> None:
    dataset, manifest, _ = frozen_input
    target_path = {
        "csv": dataset,
        "metadata": dataset.with_name(f"{dataset.name}.metadata.json"),
        "anomaly": dataset.with_suffix(".anomalies.json"),
    }[target]
    target_path.write_bytes(target_path.read_bytes() + b"\n")

    with pytest.raises(RecommendationExperimentError, match="checksum"):
        _run(manifest, _experiment_path("changed.json"))


def test_run_rejects_generation_mismatch_after_matching_sidecar_hashes(
    frozen_input: tuple[Path, Path, ResearchContract],
) -> None:
    dataset, manifest, _ = frozen_input
    metadata_path = dataset.with_name(f"{dataset.name}.metadata.json")
    anomaly_path = dataset.with_suffix(".anomalies.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    anomaly = json.loads(anomaly_path.read_text(encoding="utf-8"))
    metadata["generation_id"] = "different-generation"
    anomaly["generation_id"] = "different-generation"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    anomaly_path.write_text(json.dumps(anomaly), encoding="utf-8")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["dataset"]["metadata_sha256"] = experiments.csv_sha256(metadata_path)
    payload["dataset"]["anomaly_sidecar_sha256"] = experiments.csv_sha256(anomaly_path)
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RecommendationExperimentError, match="provenance"):
        _run(manifest, _experiment_path("generation.json"))


@pytest.mark.parametrize(
    "output",
    [
        Path("reports/research/baseline.json"),
        Path("reports/research/experiments/../baseline.json"),
        Path("reports/research/experiments/baseline.txt"),
    ],
)
def test_run_rejects_unsafe_experiment_output(
    frozen_input: tuple[Path, Path, ResearchContract], output: Path
) -> None:
    _, manifest, _ = frozen_input

    with pytest.raises(RecommendationExperimentError, match="output"):
        _run(manifest, output)


def test_run_does_not_overwrite_existing_report_when_validation_fails(
    frozen_input: tuple[Path, Path, ResearchContract],
) -> None:
    dataset, manifest, _ = frozen_input
    output = _experiment_path("existing.json")
    _run(manifest, output)
    original = output.read_bytes()
    dataset.write_bytes(dataset.read_bytes() + b"\n")

    with pytest.raises(RecommendationExperimentError):
        _run(manifest, output, overwrite=True)

    assert output.read_bytes() == original


def test_run_rejects_existing_or_symlinked_experiment_output(
    frozen_input: tuple[Path, Path, ResearchContract], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, manifest, _ = frozen_input
    output = _experiment_path("existing.json")
    _run(manifest, output)

    with pytest.raises(RecommendationExperimentError, match="already exists"):
        _run(manifest, output)

    original_is_symlink = Path.is_symlink
    raw_output = Path.cwd() / output

    def is_symlink(path: Path) -> bool:
        return path == raw_output or original_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", is_symlink)
    with pytest.raises(RecommendationExperimentError, match="must not be a symlink"):
        _run(manifest, output, overwrite=True)


def _real_candidate_candles() -> list[Candle]:
    result: list[Candle] = []
    for index in range(270):
        close = (
            Decimal("120") - Decimal(index) * Decimal("0.1") + Decimal(index % 2) * Decimal("0.2")
            if index < 220
            else Decimal("150") + Decimal(index - 220) * Decimal("3")
        )
        result.append(
            Candle(
                BASE + timedelta(hours=index),
                BASE + timedelta(hours=index + 1),
                "BTC/USDT",
                "1h",
                close,
                close + Decimal("2"),
                close - Decimal("2"),
                close,
                Decimal("5000") if index >= 220 else Decimal("1000"),
                True,
            )
        )
    return result


def test_causal_backfill_recommendation_is_unchanged_by_future_mutation() -> None:
    original = _real_candidate_candles()
    engine = RecommendationEngine(BotSettings(), now=lambda: datetime(2026, 1, 2, tzinfo=UTC))
    baseline = backfill_recommendations(engine, original)
    candidate_index = next(
        index for index, item in enumerate(baseline) if item.rule_reason.startswith("ema_volume")
    )
    mutated = list(original)
    for index in range(candidate_index + 1, len(mutated)):
        candle = mutated[index]
        mutated[index] = Candle(
            candle.open_time,
            candle.close_time,
            candle.symbol,
            candle.timeframe,
            candle.open + Decimal("100"),
            candle.high + Decimal("100"),
            candle.low + Decimal("100"),
            candle.close + Decimal("100"),
            candle.volume * Decimal("2"),
            candle.is_closed,
        )

    changed = backfill_recommendations(engine, mutated)

    assert changed[candidate_index] == baseline[candidate_index]


def test_experiment_path_has_no_execution_or_model_dependencies() -> None:
    source = inspect.getsource(experiments)
    for forbidden in (
        "RiskEngine",
        "BinanceBroker",
        "DryRunBroker",
        "OrderRequest",
        "api_key",
        "api_secret",
        "ProbabilityModel",
    ):
        assert forbidden not in source


def test_cli_runs_research_only_experiment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = {
        "candidate_id": "baseline_ema_volume_atr_v1",
        "research_role": "development",
        "strict_oos_evaluation_history": False,
        "research_claim_eligible": False,
        "recommendation_count": 3,
        "outcome_count": 9,
        "safety_locks": {
            "live_trading_enabled": False,
            "broker_used": False,
            "orders_submitted": False,
            "ml_inference_used": False,
        },
    }
    monkeypatch.setattr(
        "trading_bot.cli.run_development_experiment", lambda *args, **kwargs: result
    )

    assert (
        main(
            [
                "run-recommendation-experiment",
                "--manifest",
                str(tmp_path / "manifest.json"),
                "--candidate",
                "baseline_ema_volume_atr_v1",
                "--output",
                str(tmp_path / "report.json"),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["research_role"] == "development"
    assert payload["research_claim_eligible"] is False
