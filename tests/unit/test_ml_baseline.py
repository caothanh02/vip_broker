from __future__ import annotations

import csv
import hashlib
import json
import pickle
from pathlib import Path

import pytest

from trading_bot.cli import main
from trading_bot.features.pipeline import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION
from trading_bot.ml import baseline
from trading_bot.ml.baseline import BaselineTrainingError, train_logistic_baseline
from trading_bot.ml.dataset import AUDIT_COLUMNS, LABEL_COLUMNS, DatasetBuildError


@pytest.fixture(autouse=True)
def _validated_fixture_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    def validate(directory: Path) -> dict[str, object]:
        try:
            return json.loads((directory / "dataset.manifest.json").read_text(encoding="utf-8"))
        except OSError as exc:
            raise DatasetBuildError("dataset manifest is missing") from exc

    monkeypatch.setattr(baseline, "validate_development_dataset_generation", validate)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(split: str, index: int, target: int) -> dict[str, str | float | int]:
    base = f"2025-01-{index + 1:02d}T"
    return {
        "split": split,
        "segment_id": f"{split}-0",
        "signal_time": f"{base}00:00:00Z",
        "entry_time": f"{base}01:00:00Z",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        **{column: float(index + target * 10) for column in FEATURE_COLUMNS},
        "label_end_time": f"{base}02:00:00Z",
        "target": target,
        "outcome": "profit" if target else "stop",
        "net_return_after_costs": "0.02" if target else "-0.01",
    }


def _write_split(path: Path, split: str, count: int, labels: list[int] | None = None) -> None:
    labels = labels or [index % 2 for index in range(count)]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=[*AUDIT_COLUMNS, *FEATURE_COLUMNS, *LABEL_COLUMNS]
        )
        writer.writeheader()
        for index, label in enumerate(labels):
            writer.writerow(_row(split, index, label))


def _dataset(
    tmp_path: Path, *, validation_count: int = 6, train_labels: list[int] | None = None
) -> Path:
    directory = tmp_path / "dataset"
    directory.mkdir(parents=True)
    _write_split(directory / "train.csv", "train", 8, train_labels)
    _write_split(directory / "validation.csv", "validation", validation_count)
    _write_split(directory / "test.csv", "test", 6)
    checksums = {
        f"{split}.csv": _sha256(directory / f"{split}.csv")
        for split in ("train", "validation", "test")
    }
    (directory / "dataset.manifest.json").write_text(
        json.dumps(
            {
                "dataset_generation_id": "dataset-generation",
                "source_generation_id": "source-generation",
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "feature_columns": FEATURE_COLUMNS,
                "output_file_sha256": checksums,
            }
        ),
        encoding="utf-8",
    )
    return directory


def test_training_never_opens_final_holdout_and_writes_provenance(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    assert not (dataset / "final_holdout.csv").exists()
    output = tmp_path / "model"
    summary = train_logistic_baseline(dataset, output)
    metadata = json.loads((output / "model.metadata.json").read_text(encoding="utf-8"))
    assert summary.threshold > 0
    assert metadata["ordered_feature_schema"] == FEATURE_COLUMNS
    assert metadata["dataset_generation_id"] == "dataset-generation"
    assert metadata["experimental_only"] is True
    assert metadata["production_eligible"] is False
    assert metadata["live_trading_enabled"] is False
    assert set(metadata["input_file_sha256"]) == {"train.csv", "validation.csv", "test.csv"}


def test_scaler_fits_train_only_and_test_labels_do_not_change_model_or_threshold(
    tmp_path: Path,
) -> None:
    first = _dataset(tmp_path / "first")
    second = _dataset(tmp_path / "second")
    second_test = second / "test.csv"
    rows = list(csv.DictReader(second_test.open(encoding="utf-8", newline="")))
    for row in rows:
        row["target"] = "1" if row["target"] == "0" else "0"
    with second_test.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=[*AUDIT_COLUMNS, *FEATURE_COLUMNS, *LABEL_COLUMNS]
        )
        writer.writeheader()
        writer.writerows(rows)
    manifest = json.loads((second / "dataset.manifest.json").read_text(encoding="utf-8"))
    manifest["output_file_sha256"]["test.csv"] = _sha256(second_test)
    (second / "dataset.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    train_logistic_baseline(first, tmp_path / "first-model")
    train_logistic_baseline(second, tmp_path / "second-model")
    with (tmp_path / "first-model" / "model.pkl").open("rb") as handle:
        first_model = pickle.load(handle)
    with (tmp_path / "second-model" / "model.pkl").open("rb") as handle:
        second_model = pickle.load(handle)
    assert first_model["scaler"].mean_[0] == pytest.approx(8.5)
    assert first_model["scaler"].mean_.tolist() == second_model["scaler"].mean_.tolist()
    assert first_model["model"].coef_.tolist() == second_model["model"].coef_.tolist()
    assert (tmp_path / "first-model" / "threshold.selection.json").read_bytes() == (
        tmp_path / "second-model" / "threshold.selection.json"
    ).read_bytes()


def test_test_split_is_loaded_only_after_validation_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _dataset(tmp_path)
    selected = False
    original_select = baseline._select_threshold
    original_load = baseline._load_split

    def select_after_validation(*args: object) -> object:
        nonlocal selected
        result = original_select(*args)  # type: ignore[arg-type]
        selected = True
        return result

    def load_after_selection(*args: object) -> object:
        if args[2] == "test":
            assert selected
        return original_load(*args)  # type: ignore[arg-type]

    monkeypatch.setattr(baseline, "_select_threshold", select_after_validation)
    monkeypatch.setattr(baseline, "_load_split", load_after_selection)
    train_logistic_baseline(dataset, tmp_path / "model")


@pytest.mark.parametrize("bad_feature", ["nan", "inf"])
def test_training_rejects_nonfinite_features(tmp_path: Path, bad_feature: str) -> None:
    dataset = _dataset(tmp_path)
    path = dataset / "train.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    rows[0][FEATURE_COLUMNS[0]] = bad_feature
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=[*AUDIT_COLUMNS, *FEATURE_COLUMNS, *LABEL_COLUMNS]
        )
        writer.writeheader()
        writer.writerows(rows)
    manifest = json.loads((dataset / "dataset.manifest.json").read_text(encoding="utf-8"))
    manifest["output_file_sha256"]["train.csv"] = _sha256(path)
    (dataset / "dataset.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BaselineTrainingError, match="non-finite"):
        train_logistic_baseline(dataset, tmp_path / "model")


def test_training_rejects_schema_single_class_insufficient_validation_and_overwrite(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path / "schema")
    path = dataset / "train.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text(
        ",".join(reversed(lines[0].split(","))) + "\n" + "\n".join(lines[1:]) + "\n",
        encoding="utf-8",
    )
    manifest = json.loads((dataset / "dataset.manifest.json").read_text(encoding="utf-8"))
    manifest["output_file_sha256"]["train.csv"] = _sha256(path)
    (dataset / "dataset.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BaselineTrainingError, match="schema or order"):
        train_logistic_baseline(dataset, tmp_path / "schema-model")

    single_class = _dataset(tmp_path / "single", train_labels=[1] * 8)
    with pytest.raises(BaselineTrainingError, match="both binary"):
        train_logistic_baseline(single_class, tmp_path / "single-model")
    too_few = _dataset(tmp_path / "few", validation_count=4)
    with pytest.raises(BaselineTrainingError, match="minimum five"):
        train_logistic_baseline(too_few, tmp_path / "few-model")

    valid = _dataset(tmp_path / "valid")
    output = tmp_path / "valid-model"
    train_logistic_baseline(valid, output)
    with pytest.raises(BaselineTrainingError, match="overwrite"):
        train_logistic_baseline(valid, output)


def test_training_rejects_checksum_mismatch(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    path = dataset / "validation.csv"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(BaselineTrainingError, match="checksum"):
        train_logistic_baseline(dataset, tmp_path / "model")


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
