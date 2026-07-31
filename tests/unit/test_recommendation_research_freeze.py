from __future__ import annotations

import asyncio
import inspect
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from trading_bot.cli import main
from trading_bot.data.binance_vision import _verify_anomaly_continuity, parse_verified_archive_kline
from trading_bot.data.historical import download_vision_historical_csv
from trading_bot.data.market_interruptions import KNOWN_MARKET_INTERRUPTIONS
from trading_bot.domain.models import Candle
from trading_bot.recommendations import research
from trading_bot.recommendations.research import (
    ResearchContract,
    ResearchFreezeError,
    freeze_development_dataset,
)

EVENT = KNOWN_MARKET_INTERRUPTIONS[0]
BASE = datetime(2023, 3, 24, 11, tzinfo=UTC)


def _manifest_path(name: str) -> Path:
    return Path("reports/research/manifests") / name


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
def audited_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, ResearchContract]:
    monkeypatch.chdir(tmp_path)
    output = Path("data/raw/development.csv")
    asyncio.run(
        download_vision_historical_csv(
            _AuditedClient(), BASE, BASE + timedelta(hours=4), output, True
        )
    )
    contract = ResearchContract(
        development_start=BASE,
        development_end=BASE + timedelta(hours=4),
        strict_oos_start=BASE + timedelta(hours=4),
        expected_candle_count=3,
        expected_csv_sha256=research.csv_sha256(output),
        allowed_interruption_event_id=EVENT.event_id,
        allowed_missing_open_time=BASE + timedelta(hours=2),
    )
    return output, contract


def test_freeze_verified_dataset_writes_deterministic_audit_manifest(
    audited_dataset: tuple[Path, ResearchContract], tmp_path: Path
) -> None:
    input_path, contract = audited_dataset
    output = tmp_path / "reports/research/manifests/development.json"

    first = freeze_development_dataset(
        input_path,
        output,
        overwrite=False,
        contract=contract,
        now=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )
    second = freeze_development_dataset(
        input_path,
        output,
        overwrite=True,
        contract=contract,
        now=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert first == second == json.loads(output.read_text(encoding="utf-8"))
    assert first["research_role"] == "development"
    assert first["strict_oos_evaluation_history"] is False
    assert first["dataset"]["validation_status"] == "valid_with_market_interruptions"
    assert first["dataset"]["csv_sha256"] == contract.expected_csv_sha256
    assert first["market_interruptions"] == [
        {
            "event_id": EVENT.event_id,
            "missing_open_times": ["2023-03-24T13:00:00Z"],
            "official_source_urls": list(EVENT.official_source_urls),
            "tradable": False,
        }
    ]
    assert first["safety_locks"] == {
        "live_trading_enabled": False,
        "broker_used": False,
        "orders_submitted": False,
        "ml_used": False,
    }
    serialized = json.dumps(first).lower()
    assert "api_key" not in serialized
    assert "secret" not in serialized


@pytest.mark.parametrize("sidecar", ["metadata", "anomaly"])
def test_freeze_rejects_missing_required_sidecars(
    audited_dataset: tuple[Path, ResearchContract], sidecar: str
) -> None:
    input_path, contract = audited_dataset
    path = (
        input_path.with_name(f"{input_path.name}.metadata.json")
        if sidecar == "metadata"
        else input_path.with_suffix(".anomalies.json")
    )
    path.unlink()

    with pytest.raises(ResearchFreezeError, match="sidecar"):
        freeze_development_dataset(
            input_path, _manifest_path("missing-sidecar.json"), overwrite=False, contract=contract
        )


@pytest.mark.parametrize("report_name", ["../external-anomalies.json", "wrong-anomalies.json"])
def test_freeze_rejects_noncanonical_anomaly_name_before_reading_external_json(
    audited_dataset: tuple[Path, ResearchContract],
    monkeypatch: pytest.MonkeyPatch,
    report_name: str,
) -> None:
    input_path, contract = audited_dataset
    metadata_path = input_path.with_name(f"{input_path.name}.metadata.json")
    external_path = input_path.parent.parent / "external-anomalies.json"
    external_path.parent.mkdir(parents=True, exist_ok=True)
    external_path.write_text('{"untrusted": true}', encoding="utf-8")
    if report_name == "wrong-anomalies.json":
        external_path = input_path.parent / report_name
        external_path.write_text('{"untrusted": true}', encoding="utf-8")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["anomaly_report"] = report_name
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    original_read_json = research._read_json
    read_paths: list[Path] = []

    def guarded_read_json(path: Path, label: str) -> dict[str, object]:
        read_paths.append(path)
        if path == external_path:
            raise AssertionError("noncanonical anomaly JSON must not be read")
        return original_read_json(path, label)

    monkeypatch.setattr(research, "_read_json", guarded_read_json)

    with pytest.raises(ResearchFreezeError, match="identity"):
        freeze_development_dataset(
            input_path, _manifest_path("identity.json"), overwrite=False, contract=contract
        )

    assert external_path not in read_paths


def test_freeze_rejects_absolute_anomaly_name_before_reading_external_json(
    audited_dataset: tuple[Path, ResearchContract], monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path, contract = audited_dataset
    metadata_path = input_path.with_name(f"{input_path.name}.metadata.json")
    external_path = input_path.parent.parent / "external-anomalies.json"
    external_path.parent.mkdir(parents=True, exist_ok=True)
    external_path.write_text('{"untrusted": true}', encoding="utf-8")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["anomaly_report"] = str(external_path.resolve())
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    original_read_json = research._read_json

    def guarded_read_json(path: Path, label: str) -> dict[str, object]:
        if path == external_path:
            raise AssertionError("external anomaly JSON must not be read")
        return original_read_json(path, label)

    monkeypatch.setattr(research, "_read_json", guarded_read_json)

    with pytest.raises(ResearchFreezeError, match="identity"):
        freeze_development_dataset(
            input_path, _manifest_path("absolute.json"), overwrite=False, contract=contract
        )


@pytest.mark.parametrize("target", ["csv", "metadata", "anomaly"])
def test_freeze_rejects_symlinked_dataset_artifacts(
    audited_dataset: tuple[Path, ResearchContract],
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    input_path, contract = audited_dataset
    target_path = {
        "csv": input_path,
        "metadata": input_path.with_name(f"{input_path.name}.metadata.json"),
        "anomaly": input_path.with_suffix(".anomalies.json"),
    }[target]
    original_is_symlink = Path.is_symlink

    def is_symlink(path: Path) -> bool:
        return path == target_path or original_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", is_symlink)

    with pytest.raises(ResearchFreezeError, match="must not be a symlink"):
        freeze_development_dataset(
            input_path, _manifest_path("symlink.json"), overwrite=False, contract=contract
        )


@pytest.mark.parametrize("target", ["csv", "metadata", "anomaly"])
def test_freeze_rejects_checksum_tampering(
    audited_dataset: tuple[Path, ResearchContract], target: str
) -> None:
    input_path, contract = audited_dataset
    path = {
        "csv": input_path,
        "metadata": input_path.with_name(f"{input_path.name}.metadata.json"),
        "anomaly": input_path.with_suffix(".anomalies.json"),
    }[target]
    if target == "metadata":
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata["csv_sha256"] = "0" * 64
        path.write_text(json.dumps(metadata), encoding="utf-8")
    else:
        path.write_bytes(path.read_bytes() + b"\n")

    with pytest.raises(ResearchFreezeError):
        freeze_development_dataset(
            input_path, _manifest_path("tampered.json"), overwrite=False, contract=contract
        )


@pytest.mark.parametrize("mutation", ["duplicate", "conflict", "gap", "open"])
def test_freeze_rejects_invalid_candle_data(
    audited_dataset: tuple[Path, ResearchContract], mutation: str
) -> None:
    input_path, contract = audited_dataset
    lines = input_path.read_text(encoding="utf-8").splitlines()
    if mutation == "duplicate":
        lines.append(lines[1])
    elif mutation == "conflict":
        fields = lines[1].split(",")
        fields[7] = "999"
        lines.append(",".join(fields))
    elif mutation == "gap":
        lines.pop(1)
    else:
        fields = lines[1].split(",")
        fields[-1] = "false"
        lines[1] = ",".join(fields)
    input_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ResearchFreezeError):
        freeze_development_dataset(
            input_path, _manifest_path("invalid.csv.json"), overwrite=False, contract=contract
        )


def test_freeze_rejects_wrong_range_or_oos_boundary(
    audited_dataset: tuple[Path, ResearchContract],
) -> None:
    input_path, contract = audited_dataset
    wrong_range = ResearchContract(
        contract.development_start + timedelta(hours=1),
        contract.development_end,
        contract.strict_oos_start,
        contract.expected_candle_count,
        contract.expected_csv_sha256,
        contract.allowed_interruption_event_id,
        contract.allowed_missing_open_time,
    )
    with pytest.raises(ResearchFreezeError, match="range"):
        freeze_development_dataset(
            input_path, _manifest_path("range.json"), overwrite=False, contract=wrong_range
        )
    overlapping_oos = ResearchContract(
        contract.development_start,
        contract.development_end,
        contract.development_end - timedelta(hours=1),
        contract.expected_candle_count,
        contract.expected_csv_sha256,
        contract.allowed_interruption_event_id,
        contract.allowed_missing_open_time,
    )
    with pytest.raises(ResearchFreezeError, match="strict OOS"):
        freeze_development_dataset(
            input_path, _manifest_path("oos.json"), overwrite=False, contract=overlapping_oos
        )


def test_freeze_does_not_overwrite_or_publish_partial_manifest_after_validation_error(
    audited_dataset: tuple[Path, ResearchContract],
) -> None:
    input_path, contract = audited_dataset
    output = Path("reports/research/manifests/development.json")
    freeze_development_dataset(input_path, output, overwrite=False, contract=contract)
    original = output.read_bytes()
    with pytest.raises(ResearchFreezeError, match="already exists"):
        freeze_development_dataset(input_path, output, overwrite=False, contract=contract)
    assert output.read_bytes() == original

    input_path.write_bytes(input_path.read_bytes() + b"\n")
    with pytest.raises(ResearchFreezeError):
        freeze_development_dataset(input_path, output, overwrite=True, contract=contract)
    assert output.read_bytes() == original


def test_freeze_accepts_only_json_in_manifest_directory(
    audited_dataset: tuple[Path, ResearchContract],
) -> None:
    input_path, contract = audited_dataset

    manifest = freeze_development_dataset(
        input_path, _manifest_path("development.json"), overwrite=False, contract=contract
    )

    assert manifest["dataset"]["path"] == "data/raw/development.csv"


@pytest.mark.parametrize(
    ("output", "message"),
    [
        (Path("reports/research/development.json"), "inside reports/research/manifests"),
        (Path("reports/research/manifests/../development.json"), "path traversal"),
        (Path("reports/research/manifests/development.txt"), ".json"),
    ],
)
def test_freeze_rejects_output_outside_manifest_contract(
    audited_dataset: tuple[Path, ResearchContract], output: Path, message: str
) -> None:
    input_path, contract = audited_dataset

    with pytest.raises(ResearchFreezeError, match=message):
        freeze_development_dataset(input_path, output, overwrite=False, contract=contract)


@pytest.mark.parametrize("target", ["input", "metadata", "anomaly"])
def test_freeze_rejects_output_collision_with_dataset_or_sidecar(
    audited_dataset: tuple[Path, ResearchContract], target: str
) -> None:
    input_path, contract = audited_dataset
    output = {
        "input": input_path,
        "metadata": input_path.with_name(f"{input_path.name}.metadata.json"),
        "anomaly": input_path.with_suffix(".anomalies.json"),
    }[target]

    with pytest.raises(ResearchFreezeError, match="must not replace dataset input or sidecars"):
        freeze_development_dataset(input_path, output, overwrite=False, contract=contract)


def test_freeze_path_has_no_execution_dependencies_or_credentials() -> None:
    source = inspect.getsource(research)
    for forbidden in (
        "RiskEngine",
        "BinanceBroker",
        "DryRunBroker",
        "OrderRequest",
        "api_key",
        "api_secret",
    ):
        assert forbidden not in source


def test_cli_freeze_command_is_research_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = {
        "schema_version": "1.0",
        "research_role": "development",
        "dataset": {"candle_count": 3, "validation_status": "valid_with_market_interruptions"},
        "market_interruptions": [{}],
        "safety_locks": {
            "live_trading_enabled": False,
            "broker_used": False,
            "orders_submitted": False,
            "ml_used": False,
        },
    }
    monkeypatch.setattr(
        "trading_bot.cli.freeze_development_dataset", lambda *args, **kwargs: manifest
    )

    assert (
        main(
            [
                "freeze-recommendation-research",
                "--input",
                str(tmp_path / "development.csv"),
                "--output",
                str(tmp_path / "manifest.json"),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["research_role"] == "development"
    assert payload["safety_locks"]["broker_used"] is False
