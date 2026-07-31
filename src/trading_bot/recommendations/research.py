"""Freeze a verified development dataset without creating recommendations or trades."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trading_bot.data.csv_store import (
    CsvDataError,
    contains_non_tradable_intervals,
    csv_sha256,
    metadata_path,
    read_candles,
    verified_missing_open_times,
    verify_metadata_checksum,
    write_json_atomic,
)
from trading_bot.data.validation import CandleValidationError, validate_candles
from trading_bot.domain.models import Candle

_MANIFEST_SCHEMA_VERSION = "1.0"
_MANIFEST_DIRECTORY = Path("reports/research/manifests")


class ResearchFreezeError(ValueError):
    """The requested development dataset cannot be frozen as an audited contract."""


@dataclass(frozen=True, slots=True)
class ResearchContract:
    development_start: datetime
    development_end: datetime
    strict_oos_start: datetime
    expected_candle_count: int
    expected_csv_sha256: str
    allowed_interruption_event_id: str
    allowed_missing_open_time: datetime
    checksum_verification_mode: str = "official_online"


DEFAULT_RESEARCH_CONTRACT = ResearchContract(
    development_start=datetime(2022, 1, 1, tzinfo=UTC),
    development_end=datetime(2025, 1, 1, tzinfo=UTC),
    strict_oos_start=datetime(2025, 1, 1, tzinfo=UTC),
    expected_candle_count=26_303,
    expected_csv_sha256="4381359aa96f45dc985ee80af6806e1ba05352948e261cbb519dc7478cd3636a",
    allowed_interruption_event_id="binance-spot-2023-03-24-trailing-stop-maintenance",
    allowed_missing_open_time=datetime(2023, 3, 24, 13, tzinfo=UTC),
)


def _iso_utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResearchFreezeError(f"could not read {label}") from exc
    if not isinstance(value, dict):
        raise ResearchFreezeError(f"{label} must be a JSON object")
    return value


def _relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError as exc:
        raise ResearchFreezeError("dataset input must be inside the workspace") from exc


def _require_regular_non_symlink(path: Path, label: str) -> None:
    """Reject filesystem indirections before reading a trusted research artifact."""

    try:
        if path.is_symlink():
            raise ResearchFreezeError(f"{label} must not be a symlink")
        if not path.is_file():
            raise ResearchFreezeError(f"{label} must be a regular file")
    except OSError as exc:
        raise ResearchFreezeError(f"could not inspect {label}") from exc


def _canonical_anomaly_path(input_path: Path) -> Path:
    """Return the only anomaly sidecar identity accepted by the freeze contract."""

    anomaly_path = input_path.with_suffix(".anomalies.json")
    try:
        anomaly_path.resolve().relative_to(input_path.parent.resolve())
    except (OSError, ValueError) as exc:
        raise ResearchFreezeError(
            "development dataset anomaly sidecar must remain beside the CSV"
        ) from exc
    return anomaly_path


def _validated_manifest_output_path(input_path: Path, output_path: Path) -> Path:
    """Restrict manifests to their ignored workspace namespace before any data access."""

    if ".." in output_path.parts:
        raise ResearchFreezeError("research manifest output must not contain path traversal")
    workspace = Path.cwd().resolve()
    manifest_directory = (workspace / _MANIFEST_DIRECTORY).resolve()
    try:
        manifest_directory.relative_to(workspace)
    except ValueError as exc:
        raise ResearchFreezeError(
            "research manifest directory must remain inside the workspace"
        ) from exc
    resolved_output = output_path.resolve()
    protected_paths = {
        input_path.resolve(),
        metadata_path(input_path).resolve(),
        input_path.with_suffix(".anomalies.json").resolve(),
    }
    if resolved_output in protected_paths:
        raise ResearchFreezeError(
            "research manifest output must not replace dataset input or sidecars"
        )
    if output_path.suffix != ".json":
        raise ResearchFreezeError("research manifest output must be a .json file")
    try:
        resolved_output.relative_to(manifest_directory)
    except ValueError as exc:
        raise ResearchFreezeError(
            "research manifest output must be inside reports/research/manifests"
        ) from exc
    return resolved_output


def _verified_dataset(
    input_path: Path, contract: ResearchContract
) -> tuple[list[Candle], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    sidecar = metadata_path(input_path)
    anomaly_path = _canonical_anomaly_path(input_path)
    _require_regular_non_symlink(input_path, "development dataset CSV")
    _require_regular_non_symlink(sidecar, "development dataset metadata sidecar")
    _require_regular_non_symlink(anomaly_path, "development dataset anomaly sidecar")
    metadata = _read_json(sidecar, "metadata sidecar")
    report_name = metadata.get("anomaly_report")
    if report_name != anomaly_path.name:
        raise ResearchFreezeError(
            "development dataset anomaly sidecar identity does not match the CSV"
        )
    report = _read_json(anomaly_path, "anomaly sidecar")
    try:
        if not verify_metadata_checksum(input_path):
            raise ResearchFreezeError("development dataset metadata checksum is required")
        missing = verified_missing_open_times(input_path)
        candles = read_candles(input_path, allowed_missing_open_times=missing)
        validate_candles(candles, allowed_missing_open_times=missing)
    except (CsvDataError, CandleValidationError) as exc:
        raise ResearchFreezeError("development dataset validation failed") from exc

    actual_csv_sha256 = csv_sha256(input_path)
    if actual_csv_sha256 != contract.expected_csv_sha256:
        raise ResearchFreezeError("development dataset CSV checksum does not match the contract")
    if len(candles) != contract.expected_candle_count:
        raise ResearchFreezeError("development dataset candle count does not match the contract")
    if (
        candles[0].open_time != contract.development_start
        or candles[-1].close_time != contract.development_end
    ):
        raise ResearchFreezeError("development dataset range does not match the contract")
    if contract.strict_oos_start <= contract.development_start or contract.strict_oos_start < (
        contract.development_end
    ):
        raise ResearchFreezeError("strict OOS boundary must begin at or after development end")
    if metadata.get("checksum_verification_mode") != contract.checksum_verification_mode:
        raise ResearchFreezeError("development dataset checksum verification mode is invalid")
    policy = report.get("policy")
    if not isinstance(policy, dict) or policy.get("checksum_verification_mode") != (
        contract.checksum_verification_mode
    ):
        raise ResearchFreezeError("development dataset anomaly verification mode is invalid")

    interruptions = report.get("market_interruptions")
    if not isinstance(interruptions, list):
        raise ResearchFreezeError("development dataset market interruption records are invalid")
    if contains_non_tradable_intervals(input_path) != bool(interruptions):
        raise ResearchFreezeError("development dataset interruption status is inconsistent")
    if len(interruptions) != 1:
        raise ResearchFreezeError(
            "development dataset must contain exactly one audited interruption"
        )
    interruption = interruptions[0]
    if not isinstance(interruption, dict):
        raise ResearchFreezeError("development dataset interruption record is invalid")
    expected_missing = [_iso_utc(contract.allowed_missing_open_time)]
    if (
        interruption.get("event_id") != contract.allowed_interruption_event_id
        or interruption.get("missing_open_times") != expected_missing
        or interruption.get("tradable") is not False
        or interruption.get("official_source_urls") is None
    ):
        raise ResearchFreezeError(
            "development dataset interruption is not the audited contract event"
        )
    sources = interruption["official_source_urls"]
    if not isinstance(sources, list) or not all(isinstance(source, str) for source in sources):
        raise ResearchFreezeError("development dataset interruption source URLs are invalid")
    if missing != {contract.allowed_missing_open_time}:
        raise ResearchFreezeError("development dataset audited gaps do not match the contract")
    return candles, metadata, report, [interruption]


def freeze_development_dataset(
    input_path: Path,
    output_path: Path,
    *,
    overwrite: bool,
    contract: ResearchContract = DEFAULT_RESEARCH_CONTRACT,
    now: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Validate the fixed development dataset and atomically publish its audit manifest."""

    resolved_output = _validated_manifest_output_path(input_path, output_path)
    candles, metadata, report, interruptions = _verified_dataset(input_path, contract)
    if resolved_output.exists() and not overwrite:
        raise ResearchFreezeError("manifest already exists; pass --overwrite after validation")
    anomaly_path = _canonical_anomaly_path(input_path)
    created_at = (now or (lambda: datetime.now(UTC)))().astimezone(UTC)
    status = "valid_with_market_interruptions" if interruptions else "valid"
    manifest: dict[str, Any] = {
        "schema_version": _MANIFEST_SCHEMA_VERSION,
        "created_at": _iso_utc(created_at),
        "research_role": "development",
        "strict_oos_evaluation_history": False,
        "strict_oos_start": _iso_utc(contract.strict_oos_start),
        "dataset": {
            "path": _relative_path(input_path),
            "symbol": candles[0].symbol,
            "timeframe": candles[0].timeframe,
            "range": {
                "start": _iso_utc(candles[0].open_time),
                "end": _iso_utc(candles[-1].close_time),
            },
            "candle_count": len(candles),
            "csv_sha256": csv_sha256(input_path),
            "metadata_sha256": csv_sha256(metadata_path(input_path)),
            "anomaly_sidecar_sha256": csv_sha256(anomaly_path),
            "generation_id": metadata.get("generation_id"),
            "validation_status": status,
            "checksum_verification_mode": metadata.get("checksum_verification_mode"),
        },
        "market_interruptions": [
            {
                "event_id": item["event_id"],
                "missing_open_times": item["missing_open_times"],
                "official_source_urls": item["official_source_urls"],
                "tradable": False,
            }
            for item in interruptions
        ],
        "safety_locks": {
            "live_trading_enabled": False,
            "broker_used": False,
            "orders_submitted": False,
            "ml_used": False,
        },
    }
    if not isinstance(manifest["dataset"]["generation_id"], str):
        raise ResearchFreezeError("development dataset generation ID is invalid")
    write_json_atomic(resolved_output, manifest)
    return manifest
