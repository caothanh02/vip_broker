from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from trading_bot import operations
from trading_bot.cli import build_parser, main
from trading_bot.data.binance_vision import _verify_anomaly_continuity, parse_verified_archive_kline
from trading_bot.data.csv_store import csv_sha256
from trading_bot.data.historical import download_vision_historical_csv
from trading_bot.data.market_interruptions import KNOWN_MARKET_INTERRUPTIONS
from trading_bot.domain.models import Candle
from trading_bot.operations import OperationalSafetyError, audit_safety, operational_status

EVENT = KNOWN_MARKET_INTERRUPTIONS[0]
BASE = datetime(2023, 3, 24, 11, tzinfo=UTC)


def _prepare_audit_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repository = Path(__file__).resolve().parents[2]
    for relative in (
        Path("src/trading_bot/settings.py"),
        Path("src/trading_bot/cli.py"),
        Path("src/trading_bot/operations.py"),
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            (repository / relative).read_text(encoding="utf-8"), encoding="utf-8"
        )
    (tmp_path / ".gitignore").write_text(
        (repository / ".gitignore").read_text(encoding="utf-8"), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)


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
        self.archive_urls = ["https://example.invalid/BTCUSDT-1h-2023-03.zip"]
        self.monthly_archives = [EVENT.archive_name]
        self.daily_archives: list[str] = []
        self.archive_checksums = {EVENT.archive_name: EVENT.archive_sha256}
        self.rest_suffix = None
        self.rest_suffix_candle_count = 0

    async def fetch_closed(self, start: datetime, end: datetime) -> list[Candle]:
        return [item.candle for item in self.parsed if start <= item.candle.open_time < end]


@pytest.fixture
def audited_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    input_path = Path("data/raw/btcusdt_1h.csv")
    asyncio.run(
        download_vision_historical_csv(
            _AuditedClient(), BASE, BASE + timedelta(hours=4), input_path, True
        )
    )
    return input_path


def test_operational_status_verifies_identity_interruptions_and_neutral_locks(
    audited_dataset: Path,
) -> None:
    payload = operational_status(
        audited_dataset,
        Path("reports/operations/status.json"),
        overwrite=False,
        now=lambda: BASE + timedelta(hours=6),
    )

    assert payload["dataset"]["symbol"] == "BTC/USDT"
    assert payload["dataset"]["timeframe"] == "1h"
    assert payload["dataset"]["candle_count"] == 3
    assert payload["dataset"]["source"].startswith("Binance Vision")
    assert payload["dataset"]["requested_range"] == {
        "start": "2023-03-24T11:00:00Z",
        "end": "2023-03-24T15:00:00Z",
        "candle_count": 4,
    }
    assert payload["dataset"]["stored_range"] == {
        "start": "2023-03-24T11:00:00Z",
        "end": "2023-03-24T15:00:00Z",
    }
    assert payload["dataset"]["checksums"]["metadata_csv_checksum_verified"] is True
    assert payload["dataset"]["checksums"]["anomaly_sidecar_checksum_verified"] is True
    assert payload["dataset"]["checksums"]["verification_mode"] == "official_online"
    assert payload["tradability"] == {
        "continuous_for_trading": False,
        "contains_audited_non_tradable_interruption": True,
        "audited_non_tradable_interruptions": [
            {
                "event_id": EVENT.event_id,
                "missing_open_times": ["2023-03-24T13:00:00Z"],
                "tradable": False,
            }
        ],
    }
    assert payload["safety_locks"]["default_recommendation"] == "NEUTRAL"
    assert payload["safety_locks"]["development_protocols"] == {
        "v1": "no_policy_selected",
        "v2": "no_policy_selected",
    }
    assert payload["safety_locks"]["strict_oos_2025"] == {"sealed": True, "executed": False}
    assert payload["safety_locks"]["strict_oos_sealed"] is True
    assert payload["safety_locks"]["strict_oos_evaluated"] is False
    assert payload["safety_locks"]["ml_inference_used"] is False
    assert json.loads(Path("reports/operations/status.json").read_text(encoding="utf-8")) == payload


@pytest.mark.parametrize("artifact", ["csv", "metadata", "anomaly"])
def test_operational_status_rejects_tampered_or_missing_artifacts_before_publish(
    audited_dataset: Path, artifact: str
) -> None:
    output = Path("reports/operations/status.json")
    target = {
        "csv": audited_dataset,
        "metadata": audited_dataset.with_name(f"{audited_dataset.name}.metadata.json"),
        "anomaly": audited_dataset.with_suffix(".anomalies.json"),
    }[artifact]
    if artifact == "csv":
        target.write_text("tampered", encoding="utf-8")
    else:
        target.unlink()

    with pytest.raises(OperationalSafetyError):
        operational_status(audited_dataset, output, overwrite=False)

    assert not output.exists()


def test_operational_status_rejects_duplicate_raw_timestamp_before_publish(
    audited_dataset: Path,
) -> None:
    output = Path("reports/operations/status.json")
    lines = audited_dataset.read_text(encoding="utf-8").splitlines()
    audited_dataset.write_text("\n".join([*lines, lines[-1]]) + "\n", encoding="utf-8")

    with pytest.raises(OperationalSafetyError, match="duplicate source timestamp"):
        operational_status(audited_dataset, output, overwrite=False)

    assert not output.exists()


def test_operational_status_rejects_anomaly_checksum_and_generation_mismatches(
    audited_dataset: Path,
) -> None:
    output = Path("reports/operations/status.json")
    metadata_path = audited_dataset.with_name(f"{audited_dataset.name}.metadata.json")
    anomaly_path = audited_dataset.with_suffix(".anomalies.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    anomaly = json.loads(anomaly_path.read_text(encoding="utf-8"))
    anomaly["generation_id"] = "different-generation"
    anomaly_path.write_text(json.dumps(anomaly), encoding="utf-8")
    metadata["anomaly_report_sha256"] = csv_sha256(anomaly_path)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(OperationalSafetyError, match="dataset validation failed"):
        operational_status(audited_dataset, output, overwrite=False)

    assert not output.exists()


@pytest.mark.parametrize("sidecar", ["metadata", "anomaly"])
def test_operational_status_rejects_malformed_or_duplicate_key_json(
    audited_dataset: Path, sidecar: str
) -> None:
    output = Path("reports/operations/status.json")
    path = (
        audited_dataset.with_name(f"{audited_dataset.name}.metadata.json")
        if sidecar == "metadata"
        else audited_dataset.with_suffix(".anomalies.json")
    )
    path.write_text('{"generation_id":"first","generation_id":"second"}', encoding="utf-8")

    with pytest.raises(OperationalSafetyError, match="could not read"):
        operational_status(audited_dataset, output, overwrite=False)

    assert not output.exists()


def test_operational_status_rejects_metadata_path_traversal_without_reading_target(
    audited_dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = Path("reports/operations/status.json")
    metadata_path = audited_dataset.with_name(f"{audited_dataset.name}.metadata.json")
    external = Path("external-anomalies.json").resolve()
    external.write_text('{"untrusted": true}', encoding="utf-8")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["anomaly_report"] = "../external-anomalies.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    original_read_bytes = Path.read_bytes

    def guarded_read_bytes(path: Path) -> bytes:
        if path.resolve() == external:
            raise AssertionError("metadata-selected external sidecar must not be read")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    with pytest.raises(OperationalSafetyError, match="identity"):
        operational_status(audited_dataset, output, overwrite=False)

    assert not output.exists()


@pytest.mark.parametrize("mutation", ["open_candle", "un_audited_gap"])
def test_operational_status_rejects_invalid_closed_candle_or_gap(
    audited_dataset: Path, mutation: str
) -> None:
    output = Path("reports/operations/status.json")
    metadata_path = audited_dataset.with_name(f"{audited_dataset.name}.metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    lines = audited_dataset.read_text(encoding="utf-8").splitlines()
    if mutation == "open_candle":
        lines[1] = lines[1].removesuffix(",true") + ",false"
    else:
        del lines[1]
    audited_dataset.write_text("\n".join(lines) + "\n", encoding="utf-8")
    metadata["csv_sha256"] = csv_sha256(audited_dataset)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(OperationalSafetyError, match="dataset validation failed"):
        operational_status(audited_dataset, output, overwrite=False)

    assert not output.exists()


def test_operational_status_rechecks_original_artifacts_before_publish(
    audited_dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = Path("reports/operations/status.json")
    original_recheck = operations._recheck_snapshot

    def mutate_then_recheck(paths: tuple[Path, Path, Path], digests: tuple[str, str, str]) -> None:
        paths[0].write_bytes(paths[0].read_bytes() + b"\n")
        original_recheck(paths, digests)

    monkeypatch.setattr(operations, "_recheck_snapshot", mutate_then_recheck)
    with pytest.raises(OperationalSafetyError, match="changed after validation"):
        operational_status(audited_dataset, output, overwrite=False)

    assert not output.exists()


@pytest.mark.parametrize(
    "output",
    [
        Path("/outside.json"),
        Path("reports/operations/../outside.json"),
        Path("reports/recommendations/status.json"),
        Path("reports/operations/status.txt"),
    ],
)
def test_status_rejects_unsafe_output_before_any_input_read(
    audited_dataset: Path, output: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_input_read(path: Path) -> tuple[Path, Path, Path]:
        raise AssertionError(f"unsafe output must be rejected before reading {path}")

    monkeypatch.setattr(operations, "_canonical_artifacts", unexpected_input_read)
    with pytest.raises(OperationalSafetyError):
        operational_status(audited_dataset, output, overwrite=False)


def test_status_rejects_existing_output_directory_before_any_input_read(
    audited_dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = Path("reports/operations/status.json")
    output.mkdir(parents=True)

    def unexpected_input_read(path: Path) -> tuple[Path, Path, Path]:
        raise AssertionError(f"unsafe output must be rejected before reading {path}")

    monkeypatch.setattr(operations, "_canonical_artifacts", unexpected_input_read)
    with pytest.raises(OperationalSafetyError, match="regular file"):
        operational_status(audited_dataset, output, overwrite=False)


@pytest.mark.parametrize("artifact", ["input", "metadata", "anomaly", "output"])
def test_status_rejects_symlinked_artifacts(
    audited_dataset: Path, artifact: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = Path("reports/operations/status.json")
    target = {
        "input": (Path.cwd() / audited_dataset).resolve(),
        "metadata": (
            Path.cwd() / audited_dataset.with_name(f"{audited_dataset.name}.metadata.json")
        ).resolve(),
        "anomaly": (Path.cwd() / audited_dataset.with_suffix(".anomalies.json")).resolve(),
        "output": (Path.cwd() / output).resolve(),
    }[artifact]
    original_is_symlink = Path.is_symlink

    def is_symlink(path: Path) -> bool:
        return path == target or original_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", is_symlink)
    with pytest.raises(OperationalSafetyError, match="symlink"):
        operational_status(audited_dataset, output, overwrite=False)


def test_status_refuses_overwrite_after_validation(audited_dataset: Path) -> None:
    output = Path("reports/operations/status.json")
    output.parent.mkdir(parents=True)
    output.write_text('{"existing": true}\n', encoding="utf-8")

    with pytest.raises(OperationalSafetyError, match="already exists"):
        operational_status(audited_dataset, output, overwrite=False)

    assert output.read_text(encoding="utf-8") == '{"existing": true}\n'


def test_status_atomic_failure_preserves_existing_output(
    audited_dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = Path("reports/operations/status.json")
    output.parent.mkdir(parents=True)
    output.write_bytes(b"old-status")

    def fail_write(path: Path, payload: dict[str, object]) -> None:
        raise OSError("simulated publication failure")

    monkeypatch.setattr(operations, "write_json_atomic", fail_write)
    with pytest.raises(OperationalSafetyError, match="atomically publish"):
        operational_status(audited_dataset, output, overwrite=True)

    assert output.read_bytes() == b"old-status"
    assert not list(output.parent.glob(".*.tmp"))


def test_status_replace_failure_cleans_staging_and_preserves_old_output(
    audited_dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trading_bot.data import csv_store

    output = Path("reports/operations/status.json")
    output.parent.mkdir(parents=True)
    output.write_bytes(b"old-status")

    def fail_replace(temporary: Path, target: Path) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(csv_store, "_replace_with_retry", fail_replace)
    with pytest.raises(OperationalSafetyError, match="atomically publish"):
        operational_status(audited_dataset, output, overwrite=True)

    assert output.read_bytes() == b"old-status"
    assert not list(output.parent.glob(".*.tmp"))


def test_audit_is_safe_by_default_and_redacts_secret_like_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_audit_workspace(tmp_path, monkeypatch)
    payload = audit_safety(
        Path("reports/operations/audit.json"),
        overwrite=False,
        now=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert payload["passed"] is True
    assert {item["code"] for item in payload["findings"]} >= {
        "LIVE_MODE_FAIL_CLOSED",
        "CLI_RUNTIME_IMPORT_CLOSURE",
        "NO_CREDENTIALS_REQUIRED",
        "GENERATED_ARTIFACTS_IGNORED",
    }
    serialized = json.dumps(payload).lower()
    assert "api_key" not in serialized
    assert "api_secret" not in serialized


def test_audit_fails_when_a_prohibited_dependency_is_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_audit_workspace(tmp_path, monkeypatch)
    cli = tmp_path / "src/trading_bot/cli.py"
    cli.write_text(
        cli.read_text(encoding="utf-8") + "\nfrom trading_bot.risk.engine import RiskEngine\n",
        encoding="utf-8",
    )
    payload = audit_safety(Path("reports/operations/audit.json"), overwrite=False)

    assert payload["passed"] is False
    finding = next(
        item for item in payload["findings"] if item["code"] == "CLI_RUNTIME_IMPORT_CLOSURE"
    )
    assert finding["status"] == "fail"


def test_audit_semantically_rejects_dead_live_lock_strings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_audit_workspace(tmp_path, monkeypatch)
    settings = Path("src/trading_bot/settings.py")
    settings.write_text(
        "class BotSettings:\n"
        "    def unrelated(self):\n"
        '        # if self.bot_mode == \\"live\\":\n'
        '        #     raise ValueError(\\"live mode is deliberately disabled in this build\\")\n'
        "        return None\n",
        encoding="utf-8",
    )

    payload = audit_safety(Path("reports/operations/audit.json"), overwrite=False)

    finding = next(item for item in payload["findings"] if item["code"] == "LIVE_MODE_FAIL_CLOSED")
    assert finding["status"] == "fail"


def test_audit_does_not_read_env_or_emit_secret_like_fixture_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_audit_workspace(tmp_path, monkeypatch)
    secret = "test-secret-value-7ef3c9"
    Path(".env").write_text(f"BINANCE_API_SECRET={secret}\n", encoding="utf-8")
    original_read_text = Path.read_text

    def guarded_read_text(path: Path, *args: object, **kwargs: object) -> str:
        if path.name == ".env":
            raise AssertionError("audit must not read .env")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    payload = audit_safety(Path("reports/operations/audit.json"), overwrite=False)

    assert payload["passed"] is True
    assert secret not in json.dumps(payload)


def test_audit_fails_when_a_required_ignore_rule_is_later_negated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_audit_workspace(tmp_path, monkeypatch)
    with Path(".gitignore").open("a", encoding="utf-8") as handle:
        handle.write("\n!reports/operations/status.json\n")

    payload = audit_safety(Path("reports/operations/audit.json"), overwrite=False)

    finding = next(
        item for item in payload["findings"] if item["code"] == "GENERATED_ARTIFACTS_IGNORED"
    )
    assert finding["status"] == "fail"


def test_audit_fails_when_live_lock_or_ignore_coverage_cannot_be_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_audit_workspace(tmp_path, monkeypatch)
    monkeypatch.setattr(operations, "_live_mode_fail_closed", lambda: False)
    monkeypatch.setattr(operations, "_ignored_artifact_rules", lambda: (False, ["models/*"]))
    payload = audit_safety(Path("reports/operations/audit.json"), overwrite=False)

    failed = {item["code"] for item in payload["findings"] if item["status"] == "fail"}
    assert failed == {"LIVE_MODE_FAIL_CLOSED", "GENERATED_ARTIFACTS_IGNORED"}


def test_operational_cli_commands_bypass_non_operational_dependency_loader(
    audited_dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import trading_bot.cli as cli

    _prepare_audit_workspace(tmp_path, monkeypatch)
    monkeypatch.setattr(
        cli,
        "_load_non_operational_dependencies",
        lambda: (_ for _ in ()).throw(
            AssertionError("must not load research/broker/ML dependencies")
        ),
    )
    assert (
        main(
            [
                "operational-status",
                "--input",
                str(audited_dataset),
                "--output",
                "reports/operations/cli-status.json",
            ]
        )
        == 0
    )
    assert main(["audit-safety", "--output", "reports/operations/cli-audit.json"]) == 0
    assert "operational-status" in build_parser().format_help()
    assert "audit-safety" in build_parser().format_help()
