from __future__ import annotations

import csv
import hashlib
import json
import pickle
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from trading_bot.cli import main
from trading_bot.features.pipeline import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION
from trading_bot.ml import baseline
from trading_bot.ml.artifact import ModelArtifactError, load_sealed_baseline_artifact
from trading_bot.ml.baseline import BaselineTrainingError, train_logistic_baseline
from trading_bot.ml.dataset import (
    CANDIDATE_POLICY_VERSION,
    DATASET_SCHEMA_VERSION,
    DEFAULT_CANDIDATE_POLICY,
    DEFAULT_LABEL_POLICY,
    FINAL_HOLDOUT,
    HOLDOUT_POLICY_VERSION,
    LABEL_POLICY_VERSION,
    SEGMENTATION_POLICY_VERSION,
    SPLITS,
    DatasetBuildError,
    _file_sha256,
    _generation_id,
    _output_times,
    _validate_generation,
    _write_csv,
    validate_development_dataset_generation,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _canonical_splits() -> dict[str, tuple[datetime, datetime]]:
    return {**SPLITS, FINAL_HOLDOUT: (SPLITS["test"][1], datetime(2026, 5, 3, tzinfo=UTC))}


def _row(split: str, index: int, target: int, start: datetime) -> dict[str, Any]:
    signal_time = start + timedelta(hours=index + 1)
    entry_time = signal_time + timedelta(hours=1)
    row: dict[str, Any] = {
        "split": split,
        "segment_id": f"{split}-0",
        "signal_time": _iso(signal_time),
        "entry_time": _iso(entry_time),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        **{column: float(index + target * 10) for column in FEATURE_COLUMNS},
    }
    if split != FINAL_HOLDOUT:
        row.update(
            {
                "label_end_time": _iso(entry_time),
                "target": target,
                "outcome": "profit" if target else "stop",
                "net_return_after_costs": "0.02" if target else "-0.01",
            }
        )
    return row


def _label_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "positive": sum(str(row["target"]) == "1" for row in rows),
        "negative": sum(str(row["target"]) == "0" for row in rows),
        "timeout": 0,
    }


def _write_manifest(directory: Path, manifest: dict[str, Any]) -> None:
    (directory / "dataset.manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def _read_manifest(directory: Path) -> dict[str, Any]:
    return json.loads((directory / "dataset.manifest.json").read_text(encoding="utf-8"))


def _refresh_checksum(directory: Path, split: str) -> None:
    manifest = _read_manifest(directory)
    manifest["output_file_sha256"][f"{split}.csv"] = _file_sha256(directory / f"{split}.csv")
    _write_manifest(directory, manifest)


def _canonical_dataset(
    tmp_path: Path,
    *,
    validation_count: int = 6,
    train_labels: list[int] | None = None,
) -> Path:
    """Create a production-valid generation; no baseline validator is mocked."""
    directory = tmp_path / "dataset"
    directory.mkdir(parents=True)
    splits = _canonical_splits()
    labels = train_labels or [index % 2 for index in range(8)]
    rows: dict[str, list[dict[str, Any]]] = {
        "train": [
            _row("train", index, label, splits["train"][0]) for index, label in enumerate(labels)
        ],
        "validation": [
            _row("validation", index, index % 2, splits["validation"][0])
            for index in range(validation_count)
        ],
        "test": [_row("test", index, index % 2, splits["test"][0]) for index in range(6)],
        FINAL_HOLDOUT: [_row(FINAL_HOLDOUT, 0, 0, splits[FINAL_HOLDOUT][0])],
    }
    checksums: dict[str, str] = {}
    for split, split_rows in rows.items():
        path = directory / f"{split}.csv"
        _write_csv(path, split_rows, split)
        checksums[path.name] = _file_sha256(path)
    source_checksums = {
        "source_csv_sha256": "a" * 64,
        "source_metadata_sha256": "b" * 64,
        "anomaly_report_sha256": "c" * 64,
    }
    label_counts = {split: _label_counts(rows[split]) for split in SPLITS}
    manifest: dict[str, Any] = {
        "dataset_generation_id": "",
        **source_checksums,
        "anomaly_report": "btc.anomalies.json",
        "source_generation_id": "verified-source",
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "splits": {
            split: {"start": _iso(start), "end": _iso(end)}
            for split, (start, end) in splits.items()
        },
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "candidate_policy_version": CANDIDATE_POLICY_VERSION,
        "label_policy_version": LABEL_POLICY_VERSION,
        "segmentation_policy_version": SEGMENTATION_POLICY_VERSION,
        "holdout_policy_version": HOLDOUT_POLICY_VERSION,
        "candidate_policy": asdict(DEFAULT_CANDIDATE_POLICY),
        "label_policy": {
            key: str(value) if isinstance(value, Decimal) else value
            for key, value in asdict(DEFAULT_LABEL_POLICY).items()
        },
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_columns": FEATURE_COLUMNS,
        "code_commit": None,
        "row_counts": {split: len(split_rows) for split, split_rows in rows.items()},
        "candidate_counts": {split: len(split_rows) for split, split_rows in rows.items()},
        "development_label_counts": label_counts,
        "development_trainable": {
            split: counts["positive"] > 0 and counts["negative"] > 0
            for split, counts in label_counts.items()
        },
        "excluded_row_counts": {},
        "split_signal_coverage": {
            split: _output_times(split_rows) for split, split_rows in rows.items()
        },
        "output_file_sha256": checksums,
        "interruption_summary": {"policy_version": SEGMENTATION_POLICY_VERSION, "segment_count": 4},
        "segments": {
            f"{split}-0": {"split": split, "start": _iso(start), "end": _iso(end)}
            for split, (start, end) in splits.items()
        },
    }
    manifest["dataset_generation_id"] = _generation_id(
        source_checksums,
        manifest["source_generation_id"],
        splits,
        DEFAULT_CANDIDATE_POLICY,
        DEFAULT_LABEL_POLICY,
    )
    _write_manifest(directory, manifest)
    _validate_generation(directory)
    return directory


def _rewrite_split(directory: Path, split: str, rows: list[dict[str, Any]]) -> None:
    _write_csv(directory / f"{split}.csv", rows, split)
    _refresh_checksum(directory, split)


def _refresh_test_metadata(directory: Path) -> None:
    manifest = _read_manifest(directory)
    rows = list(csv.DictReader((directory / "test.csv").open(encoding="utf-8", newline="")))
    counts = _label_counts(rows)
    manifest["row_counts"]["test"] = len(rows)
    manifest["candidate_counts"]["test"] = len(rows)
    manifest["development_label_counts"]["test"] = counts
    manifest["development_trainable"]["test"] = counts["positive"] > 0 and counts["negative"] > 0
    manifest["split_signal_coverage"]["test"] = _output_times(rows)
    _write_manifest(directory, manifest)


def test_training_never_opens_final_holdout_and_writes_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _canonical_dataset(tmp_path)
    final_path = dataset / "final_holdout.csv"
    final_path.write_text("sentinel,not,a,valid,csv\n", encoding="utf-8")
    original_open, original_stat = Path.open, Path.stat

    def reject_final_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path == final_path:
            raise AssertionError("trainer accessed final holdout")
        return original_open(path, *args, **kwargs)

    def reject_final_stat(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path == final_path:
            raise AssertionError("trainer accessed final holdout")
        return original_stat(path, *args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "open", reject_final_open)
        scoped.setattr(Path, "stat", reject_final_stat)
        output = tmp_path / "model"
        summary = train_logistic_baseline(dataset, output)
    metadata = json.loads((output / "model.metadata.json").read_text(encoding="utf-8"))
    assert summary.threshold > 0
    assert metadata["ordered_feature_schema"] == FEATURE_COLUMNS
    assert metadata["experimental_only"] is True
    assert metadata["production_eligible"] is False
    assert metadata["live_trading_enabled"] is False
    assert set(metadata["input_file_sha256"]) == {"train.csv", "validation.csv", "test.csv"}
    artifact = load_sealed_baseline_artifact(
        output,
        dataset_generation_id=summary.dataset_generation_id,
        source_generation_id=summary.source_generation_id,
    )
    assert artifact.threshold == summary.threshold
    with pytest.raises(DatasetBuildError):
        _validate_generation(dataset)


@pytest.mark.parametrize("mutation", ["missing", "extra", "checksum", "duplicate", "threshold"])
def test_artifact_loader_fails_closed_before_unpickle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    dataset = _canonical_dataset(tmp_path)
    output = tmp_path / "model"
    summary = train_logistic_baseline(dataset, output)
    if mutation == "missing":
        (output / "model.pkl").unlink()
    elif mutation == "extra":
        (output / "extra").write_text("x", encoding="utf-8")
    elif mutation == "checksum":
        (output / "model.pkl").write_bytes(b"tampered")
    elif mutation == "duplicate":
        (output / "artifact.manifest.json").write_text('{"files":{},"files":{}}', encoding="utf-8")
    else:
        manifest = json.loads((output / "artifact.manifest.json").read_text(encoding="utf-8"))
        manifest["threshold"] = float("nan")
        (output / "artifact.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def reject_unpickle(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("pickle.loads must not be called")

    monkeypatch.setattr(pickle, "loads", reject_unpickle)
    with pytest.raises(ModelArtifactError):
        load_sealed_baseline_artifact(
            output,
            dataset_generation_id=summary.dataset_generation_id,
            source_generation_id=summary.source_generation_id,
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda manifest: manifest.update({"threshold": -0.1}),
        lambda manifest: manifest.update({"threshold": 1.1}),
        lambda manifest: manifest.update({"threshold": float("inf")}),
        lambda manifest: manifest.update(
            {"ordered_feature_schema": list(reversed(FEATURE_COLUMNS))}
        ),
        lambda manifest: manifest.update({"experimental_only": False}),
        lambda manifest: manifest.update({"production_eligible": True}),
        lambda manifest: manifest.update({"live_trading_enabled": True}),
    ],
    ids=[
        "threshold-negative",
        "threshold-high",
        "threshold-infinite",
        "feature-order",
        "not-experimental",
        "production",
        "live",
    ],
)
def test_artifact_manifest_contract_tampering_is_rejected(
    tmp_path: Path, mutate: Callable[[dict[str, Any]], None]
) -> None:
    dataset = _canonical_dataset(tmp_path)
    output = tmp_path / "model"
    summary = train_logistic_baseline(dataset, output)
    path = output / "artifact.manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mutate(manifest)
    path.write_text(json.dumps(manifest, allow_nan=True), encoding="utf-8")
    with pytest.raises(ModelArtifactError):
        load_sealed_baseline_artifact(
            output,
            dataset_generation_id=summary.dataset_generation_id,
            source_generation_id=summary.source_generation_id,
        )


@pytest.mark.parametrize("mutation", ["dimension", "classes", "nonfinite", "type"])
def test_artifact_pickle_contract_tampering_is_rejected(tmp_path: Path, mutation: str) -> None:
    dataset = _canonical_dataset(tmp_path)
    output = tmp_path / "model"
    summary = train_logistic_baseline(dataset, output)
    model_path = output / "model.pkl"
    with model_path.open("rb") as handle:
        payload = pickle.load(handle)
    if mutation == "dimension":
        payload["scaler"].mean_ = payload["scaler"].mean_[:-1]
    elif mutation == "classes":
        payload["model"].classes_ = payload["model"].classes_[::-1]
    elif mutation == "nonfinite":
        payload["model"].coef_[0, 0] = float("nan")
    else:
        payload["model"] = object()
    with model_path.open("wb") as handle:
        pickle.dump(payload, handle)
    manifest_path = output / "artifact.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["model.pkl"] = _sha256(model_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ModelArtifactError):
        load_sealed_baseline_artifact(
            output,
            dataset_generation_id=summary.dataset_generation_id,
            source_generation_id=summary.source_generation_id,
        )


@pytest.mark.parametrize("bad_policy", [None, [], "policy", {"version": "wrong"}])
def test_artifact_rejects_malformed_threshold_policy(tmp_path: Path, bad_policy: object) -> None:
    dataset = _canonical_dataset(tmp_path)
    output = tmp_path / "model"
    summary = train_logistic_baseline(dataset, output)
    metadata_path = output / "model.metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["threshold_policy"] = bad_policy
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    manifest_path = output / "artifact.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["model.metadata.json"] = _sha256(metadata_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ModelArtifactError):
        load_sealed_baseline_artifact(
            output,
            dataset_generation_id=summary.dataset_generation_id,
            source_generation_id=summary.source_generation_id,
        )


@pytest.mark.parametrize(
    "error",
    [ValueError("bad"), TypeError("bad"), RecursionError("bad"), pickle.UnpicklingError("bad")],
)
def test_artifact_wraps_pickle_loads_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    dataset = _canonical_dataset(tmp_path)
    output = tmp_path / "model"
    summary = train_logistic_baseline(dataset, output)
    monkeypatch.setattr(pickle, "loads", lambda _: (_ for _ in ()).throw(error))
    with pytest.raises(ModelArtifactError) as raised:
        load_sealed_baseline_artifact(
            output,
            dataset_generation_id=summary.dataset_generation_id,
            source_generation_id=summary.source_generation_id,
        )
    assert raised.value.__cause__ is error


def test_valid_test_labels_do_not_change_model_threshold_or_validation_metrics(
    tmp_path: Path,
) -> None:
    first = _canonical_dataset(tmp_path / "first")
    second = _canonical_dataset(tmp_path / "second")
    test_rows = list(csv.DictReader((second / "test.csv").open(encoding="utf-8", newline="")))
    for row in test_rows:
        target = 1 - int(row["target"])
        row["target"] = target
        row["outcome"] = "profit" if target else "stop"
        row["net_return_after_costs"] = "0.04" if target else "-0.03"
    _rewrite_split(second, "test", test_rows)
    _refresh_test_metadata(second)
    validate_development_dataset_generation(first)
    validate_development_dataset_generation(second)

    first_summary = train_logistic_baseline(first, tmp_path / "first-model")
    second_summary = train_logistic_baseline(second, tmp_path / "second-model")
    with (tmp_path / "first-model" / "model.pkl").open("rb") as handle:
        first_model = pickle.load(handle)
    with (tmp_path / "second-model" / "model.pkl").open("rb") as handle:
        second_model = pickle.load(handle)
    assert first_model["scaler"].mean_.tolist() == second_model["scaler"].mean_.tolist()
    assert first_model["scaler"].scale_.tolist() == second_model["scaler"].scale_.tolist()
    assert first_model["model"].coef_.tolist() == second_model["model"].coef_.tolist()
    assert first_model["model"].intercept_.tolist() == second_model["model"].intercept_.tolist()
    assert first_summary.threshold == second_summary.threshold
    assert first_summary.validation_metrics == second_summary.validation_metrics
    assert first_summary.test_metrics != second_summary.test_metrics


def test_semantic_test_label_tampering_is_rejected_before_model_output(tmp_path: Path) -> None:
    dataset = _canonical_dataset(tmp_path)
    rows = list(csv.DictReader((dataset / "test.csv").open(encoding="utf-8", newline="")))
    rows[0]["target"] = "1" if rows[0]["target"] == "0" else "0"
    _rewrite_split(dataset, "test", rows)
    with pytest.raises(BaselineTrainingError, match="development dataset generation"):
        train_logistic_baseline(dataset, tmp_path / "model")
    assert not (tmp_path / "model").exists()


def test_test_split_is_loaded_only_after_validation_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _canonical_dataset(tmp_path)
    selected = False
    original_select = baseline._select_threshold
    original_load = baseline._load_split

    def select_after_validation(*args: Any) -> Any:
        nonlocal selected
        result = original_select(*args)
        selected = True
        return result

    def load_after_selection(*args: Any) -> Any:
        if args[2] == "test":
            assert selected
        return original_load(*args)

    monkeypatch.setattr(baseline, "_select_threshold", select_after_validation)
    monkeypatch.setattr(baseline, "_load_split", load_after_selection)
    train_logistic_baseline(dataset, tmp_path / "model")


@pytest.mark.parametrize("bad_feature", ["nan", "inf"])
def test_training_rejects_nonfinite_features(tmp_path: Path, bad_feature: str) -> None:
    dataset = _canonical_dataset(tmp_path)
    rows = list(csv.DictReader((dataset / "train.csv").open(encoding="utf-8", newline="")))
    rows[0][FEATURE_COLUMNS[0]] = bad_feature
    _rewrite_split(dataset, "train", rows)
    with pytest.raises(BaselineTrainingError, match="development dataset generation"):
        train_logistic_baseline(dataset, tmp_path / "model")


def test_training_rejects_single_class_insufficient_validation_and_overwrite(
    tmp_path: Path,
) -> None:
    single_class = _canonical_dataset(tmp_path / "single", train_labels=[1] * 8)
    with pytest.raises(BaselineTrainingError, match="both binary"):
        train_logistic_baseline(single_class, tmp_path / "single-model")
    too_few = _canonical_dataset(tmp_path / "few", validation_count=4)
    with pytest.raises(BaselineTrainingError, match="minimum five"):
        train_logistic_baseline(too_few, tmp_path / "few-model")

    valid = _canonical_dataset(tmp_path / "valid")
    output = tmp_path / "valid-model"
    train_logistic_baseline(valid, output)
    with pytest.raises(BaselineTrainingError, match="overwrite"):
        train_logistic_baseline(valid, output)


def test_training_rejects_checksum_mismatch(tmp_path: Path) -> None:
    dataset = _canonical_dataset(tmp_path)
    path = dataset / "validation.csv"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(BaselineTrainingError, match="development dataset generation"):
        train_logistic_baseline(dataset, tmp_path / "model")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda manifest, directory: manifest["candidate_policy"].update({"ema_fast": 21}),
        lambda manifest, directory: manifest["splits"]["train"].update(
            {"start": "2022-01-01T01:00:00Z"}
        ),
        lambda manifest, directory: manifest.update({"dataset_generation_id": "0" * 32}),
        lambda manifest, directory: manifest["development_label_counts"]["train"].update(
            {"positive": 99}
        ),
    ],
    ids=[
        "unsupported-policy",
        "noncanonical-split",
        "forged-identity",
        "count-mismatch",
    ],
)
def test_training_rejects_provenance_tampering(
    tmp_path: Path, mutate: Callable[[dict[str, Any], Path], None]
) -> None:
    dataset = _canonical_dataset(tmp_path)
    manifest = _read_manifest(dataset)
    mutate(manifest, dataset)
    _write_manifest(dataset, manifest)
    with pytest.raises(BaselineTrainingError, match="development dataset generation"):
        train_logistic_baseline(dataset, tmp_path / "model")
    assert not (tmp_path / "model").exists()


def _tamper_train_timestamp(manifest: dict[str, Any], directory: Path) -> None:
    rows = list(csv.DictReader((directory / "train.csv").open(encoding="utf-8", newline="")))
    rows[0]["signal_time"] = "2021-12-31T23:00:00Z"
    _write_csv(directory / "train.csv", rows, "train")
    manifest["output_file_sha256"]["train.csv"] = _sha256(directory / "train.csv")


def test_training_rejects_timestamp_tampering_for_the_semantic_reason(tmp_path: Path) -> None:
    dataset = _canonical_dataset(tmp_path)
    manifest = _read_manifest(dataset)
    _tamper_train_timestamp(manifest, dataset)
    _write_manifest(dataset, manifest)
    with pytest.raises(BaselineTrainingError, match="development dataset generation") as raised:
        train_logistic_baseline(dataset, tmp_path / "model")
    assert isinstance(raised.value.__cause__, DatasetBuildError)
    assert "staged signal/entry leaves its split" in str(raised.value.__cause__)
    assert not (tmp_path / "model").exists()


def _validate_development_without_final_filesystem_access(
    dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> Any:
    final_path = dataset / "final_holdout.csv"
    original_open, original_stat = Path.open, Path.stat

    def reject_final_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path == final_path:
            raise AssertionError("development validator accessed final holdout")
        return original_open(path, *args, **kwargs)

    def reject_final_stat(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path == final_path:
            raise AssertionError("development validator accessed final holdout")
        return original_stat(path, *args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "open", reject_final_open)
        scoped.setattr(Path, "stat", reject_final_stat)
        return validate_development_dataset_generation(dataset)


@pytest.mark.parametrize(
    ("mutate", "accepted"),
    [
        (
            lambda manifest: (
                manifest["row_counts"].update({FINAL_HOLDOUT: 0}),
                manifest["candidate_counts"].update({FINAL_HOLDOUT: 1}),
                manifest["excluded_row_counts"].update({"missing_next_open": 1}),
                manifest["split_signal_coverage"].update(
                    {FINAL_HOLDOUT: {"first_signal_time": None, "last_signal_time": None}}
                ),
            ),
            True,
        ),
        (lambda manifest: manifest["candidate_counts"].update({FINAL_HOLDOUT: 0}), False),
        (
            lambda manifest: manifest["candidate_counts"].update({FINAL_HOLDOUT: 2}),
            False,
        ),
        (lambda manifest: manifest["candidate_counts"].update({FINAL_HOLDOUT: True}), False),
        (lambda manifest: manifest["candidate_counts"].update({FINAL_HOLDOUT: -1}), False),
        (lambda manifest: manifest["row_counts"].update({FINAL_HOLDOUT: -1}), False),
        (
            lambda manifest: manifest["excluded_row_counts"].update({"missing_next_open": True}),
            False,
        ),
        (
            lambda manifest: manifest["excluded_row_counts"].update({"missing_next_open": -1}),
            False,
        ),
        (lambda manifest: manifest.update({"excluded_row_counts": []}), False),
    ],
    ids=[
        "missing-next-open-exclusion",
        "candidate-less-than-rows",
        "unexplained-candidate-difference",
        "candidate-bool",
        "candidate-negative",
        "row-negative",
        "missing-next-open-bool",
        "missing-next-open-negative",
        "excluded-row-counts-not-dict",
    ],
)
def test_development_validator_checks_final_candidate_row_invariant_without_opening_final_holdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate: Callable[[dict[str, Any]], object],
    accepted: bool,
) -> None:
    dataset = _canonical_dataset(tmp_path)
    manifest = _read_manifest(dataset)
    mutate(manifest)
    _write_manifest(dataset, manifest)
    if accepted:
        _validate_development_without_final_filesystem_access(dataset, monkeypatch)
    else:
        with pytest.raises(DatasetBuildError):
            _validate_development_without_final_filesystem_access(dataset, monkeypatch)


def _set_two_final_holdout_segments(manifest: dict[str, Any]) -> None:
    manifest["segments"][f"{FINAL_HOLDOUT}-0"]["end"] = "2026-05-01T02:00:00Z"
    manifest["segments"][f"{FINAL_HOLDOUT}-1"] = {
        "split": FINAL_HOLDOUT,
        "start": "2026-05-01T04:00:00Z",
        "end": "2026-05-03T00:00:00Z",
    }


def test_development_validator_accepts_final_coverage_across_multiple_segments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _canonical_dataset(tmp_path)
    manifest = _read_manifest(dataset)
    _set_two_final_holdout_segments(manifest)
    manifest["split_signal_coverage"][FINAL_HOLDOUT] = {
        "first_signal_time": "2026-05-01T01:00:00Z",
        "last_signal_time": "2026-05-01T04:00:00Z",
    }
    _write_manifest(dataset, manifest)
    _validate_development_without_final_filesystem_access(dataset, monkeypatch)


@pytest.mark.parametrize(
    "coverage",
    [
        {"first_signal_time": "2026-05-01T03:00:00Z", "last_signal_time": "2026-05-01T04:00:00Z"},
        {"first_signal_time": "2026-05-01T01:00:00Z", "last_signal_time": "2026-05-01T03:00:00Z"},
    ],
    ids=["first-in-interruption-gap", "last-in-interruption-gap"],
)
def test_development_validator_rejects_final_coverage_inside_interruption_gap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, coverage: dict[str, str]
) -> None:
    dataset = _canonical_dataset(tmp_path)
    manifest = _read_manifest(dataset)
    _set_two_final_holdout_segments(manifest)
    manifest["split_signal_coverage"][FINAL_HOLDOUT] = coverage
    _write_manifest(dataset, manifest)
    with pytest.raises(DatasetBuildError):
        _validate_development_without_final_filesystem_access(dataset, monkeypatch)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda manifest: manifest["segments"][f"{FINAL_HOLDOUT}-0"].update(
            {"end": "2026-05-02T00:00:00Z"}
        ),
        lambda manifest: manifest["segments"][f"{FINAL_HOLDOUT}-0"].update(
            {"start": "2026-04-30T23:00:00Z"}
        ),
    ],
    ids=["overlap", "outside-split"],
)
def test_development_validator_still_rejects_invalid_final_segments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutate: Callable[[dict[str, Any]], None]
) -> None:
    dataset = _canonical_dataset(tmp_path)
    manifest = _read_manifest(dataset)
    _set_two_final_holdout_segments(manifest)
    mutate(manifest)
    _write_manifest(dataset, manifest)
    with pytest.raises(DatasetBuildError):
        _validate_development_without_final_filesystem_access(dataset, monkeypatch)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda manifest: manifest["output_file_sha256"].pop("final_holdout.csv"),
        lambda manifest: manifest["output_file_sha256"].update({"final_holdout.csv": "x" * 64}),
        lambda manifest: manifest["row_counts"].pop(FINAL_HOLDOUT),
        lambda manifest: manifest["row_counts"].update({FINAL_HOLDOUT: True}),
        lambda manifest: manifest["candidate_counts"].pop(FINAL_HOLDOUT),
        lambda manifest: manifest["candidate_counts"].update({FINAL_HOLDOUT: 2}),
        lambda manifest: manifest["split_signal_coverage"].pop(FINAL_HOLDOUT),
        lambda manifest: manifest["split_signal_coverage"].update(
            {FINAL_HOLDOUT: {"first_signal_time": None, "last_signal_time": "2026-05-01T01:00:00Z"}}
        ),
        lambda manifest: manifest["segments"][f"{FINAL_HOLDOUT}-0"].update(
            {"start": "2026-05-01T02:00:00Z"}
        ),
    ],
    ids=[
        "checksum-key",
        "checksum-format",
        "row-key",
        "row-bool",
        "candidate-key",
        "candidate-mismatch",
        "coverage-key",
        "coverage-invalid",
        "coverage-outside-final-segment",
    ],
)
def test_development_validator_rejects_bad_final_metadata_without_opening_final_holdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutate: Callable[[dict[str, Any]], None]
) -> None:
    dataset = _canonical_dataset(tmp_path)
    manifest = _read_manifest(dataset)
    mutate(manifest)
    _write_manifest(dataset, manifest)
    with pytest.raises(DatasetBuildError):
        _validate_development_without_final_filesystem_access(dataset, monkeypatch)


def test_train_cli_fails_closed_for_invalid_dataset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status = main(
        [
            "train",
            "--dataset-dir",
            str(tmp_path / "missing"),
            "--output-dir",
            str(tmp_path / "model"),
        ]
    )
    assert status == 1
    assert "development dataset generation" in capsys.readouterr().err
