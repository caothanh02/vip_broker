"""Sealed, deterministic Logistic Regression baseline for candidate filtering only."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import pickle
import platform
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Final

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from trading_bot.features.pipeline import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION
from trading_bot.ml.dataset import (
    AUDIT_COLUMNS,
    LABEL_COLUMNS,
    DatasetBuildError,
    validate_development_dataset_generation,
)

DEVELOPMENT_SPLITS: Final = ("train", "validation", "test")
THRESHOLD_POLICY_VERSION: Final = "1.0.0"
BASELINE_MODEL_VERSION: Final = "1.0.0"
MIN_VALIDATION_TRADES: Final = 5


class BaselineTrainingError(ValueError):
    """The sealed development dataset or baseline artifact contract is invalid."""


@dataclass(frozen=True, slots=True)
class SplitData:
    features: np.ndarray
    targets: np.ndarray
    returns: tuple[Decimal, ...]
    checksum: str


@dataclass(frozen=True, slots=True)
class ThresholdSelection:
    threshold: float
    selected_trade_count: int
    cumulative_net_return: Decimal
    mean_net_return: Decimal
    precision: float


@dataclass(frozen=True, slots=True)
class BaselineTrainingSummary:
    dataset_generation_id: str
    source_generation_id: str
    threshold: float
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    output_dir: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_split(dataset_dir: Path, manifest: Mapping[str, Any], split: str) -> SplitData:
    if split not in DEVELOPMENT_SPLITS:
        raise BaselineTrainingError("final holdout is sealed from baseline training")
    path = dataset_dir / f"{split}.csv"
    checksums = manifest["output_file_sha256"]
    expected_checksum = checksums.get(path.name)
    if (
        not isinstance(expected_checksum, str)
        or not path.is_file()
        or _sha256(path) != expected_checksum
    ):
        raise BaselineTrainingError(f"{split} dataset checksum is invalid")
    features: list[list[float]] = []
    targets: list[int] = []
    returns: list[Decimal] = []
    expected_columns = [*AUDIT_COLUMNS, *FEATURE_COLUMNS, *LABEL_COLUMNS]
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != expected_columns:
                raise BaselineTrainingError(f"{split} feature schema or order is invalid")
            for row in reader:
                if set(row) != set(expected_columns) or row.get("split") != split:
                    raise BaselineTrainingError(f"{split} dataset row schema is invalid")
                values = [float(row[column]) for column in FEATURE_COLUMNS]
                if not all(math.isfinite(value) for value in values):
                    raise BaselineTrainingError(f"{split} contains non-finite features")
                target = row.get("target")
                if target not in {"0", "1"}:
                    raise BaselineTrainingError(f"{split} target is invalid")
                net_return = Decimal(str(row.get("net_return_after_costs")))
                if not net_return.is_finite():
                    raise BaselineTrainingError(f"{split} contains non-finite returns")
                features.append(values)
                targets.append(int(target))
                returns.append(net_return)
    except BaselineTrainingError:
        raise
    except (OSError, ValueError, TypeError, KeyError, InvalidOperation, csv.Error) as exc:
        raise BaselineTrainingError(f"could not read {split} dataset") from exc
    if not features:
        raise BaselineTrainingError(f"{split} dataset is empty")
    return SplitData(
        np.asarray(features, dtype=float),
        np.asarray(targets, dtype=int),
        tuple(returns),
        expected_checksum,
    )


def _load_training_data(dataset_dir: Path) -> tuple[dict[str, SplitData], Mapping[str, Any]]:
    try:
        manifest = validate_development_dataset_generation(dataset_dir)
    except DatasetBuildError as exc:
        raise BaselineTrainingError("development dataset generation is invalid") from exc
    splits = {split: _load_split(dataset_dir, manifest, split) for split in ("train", "validation")}
    return splits, manifest


def _precision(targets: np.ndarray, selected: np.ndarray) -> float:
    selected_count = int(selected.sum())
    return float(((targets == 1) & selected).sum() / selected_count) if selected_count else 0.0


def _trade_metrics(
    data: SplitData, selected: np.ndarray, probabilities: np.ndarray | None
) -> dict[str, Any]:
    selected_count = int(selected.sum())
    count = len(data.targets)
    true_positive = int(((data.targets == 1) & selected).sum())
    true_negative = int(((data.targets == 0) & ~selected).sum())
    precision = true_positive / selected_count if selected_count else 0.0
    positives = int((data.targets == 1).sum())
    negatives = int((data.targets == 0).sum())
    recall = true_positive / positives if positives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    specificity = true_negative / negatives if negatives else 0.0
    selected_returns = [
        value for value, include in zip(data.returns, selected, strict=True) if include
    ]
    cumulative = sum(selected_returns, Decimal())
    mean = cumulative / Decimal(selected_count) if selected_count else Decimal()
    gross_profit = sum((value for value in selected_returns if value > 0), Decimal())
    gross_loss = -sum((value for value in selected_returns if value < 0), Decimal())
    profit_factor: str | None = str(gross_profit / gross_loss) if gross_loss else None
    equity, peak, drawdown = Decimal("1"), Decimal("1"), Decimal()
    for value in selected_returns:
        equity *= Decimal("1") + value
        peak = max(peak, equity)
        drawdown = max(drawdown, (peak - equity) / peak)
    metrics: dict[str, Any] = {
        "candidate_count": count,
        "selected_trade_count": selected_count,
        "selection_rate": selected_count / count,
        "positive_count": positives,
        "negative_count": negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": (recall + specificity) / 2,
        "cumulative_net_return_after_costs": str(cumulative),
        "mean_net_return_after_costs": str(mean),
        "win_rate": sum(value > 0 for value in selected_returns) / selected_count
        if selected_count
        else 0.0,
        "profit_factor": profit_factor,
        "maximum_drawdown": str(drawdown),
        "roc_auc": None,
        "pr_auc": None,
        "brier_score": None,
    }
    if probabilities is not None and positives and negatives:
        metrics.update(
            {
                "roc_auc": float(roc_auc_score(data.targets, probabilities)),
                "pr_auc": float(average_precision_score(data.targets, probabilities)),
                "brier_score": float(brier_score_loss(data.targets, probabilities)),
            }
        )
    return metrics


def _select_threshold(data: SplitData, probabilities: np.ndarray) -> ThresholdSelection:
    candidates: list[ThresholdSelection] = []
    for threshold in sorted({float(value) for value in probabilities}):
        selected = probabilities >= threshold
        count = int(selected.sum())
        if count < MIN_VALIDATION_TRADES:
            continue
        selected_returns = [
            value for value, include in zip(data.returns, selected, strict=True) if include
        ]
        cumulative = sum(selected_returns, Decimal())
        mean = cumulative / Decimal(count)
        candidates.append(
            ThresholdSelection(
                threshold, count, cumulative, mean, _precision(data.targets, selected)
            )
        )
    if not candidates:
        raise BaselineTrainingError("validation does not provide the minimum five threshold trades")
    return max(
        candidates,
        key=lambda choice: (
            choice.cumulative_net_return,
            choice.mean_net_return,
            choice.precision,
            choice.selected_trade_count,
            choice.threshold,
        ),
    )


def _code_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, check=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value if len(value) == 40 else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    def normalize(value: Any) -> Any:
        if isinstance(value, Decimal):
            return str(value)
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return value

    path.write_text(
        json.dumps(normalize(payload), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def train_logistic_baseline(dataset_dir: Path, output_dir: Path) -> BaselineTrainingSummary:
    """Fit on train only, select a threshold on validation only, then evaluate test once."""
    if output_dir.exists():
        raise BaselineTrainingError("refusing to overwrite an existing model artifact directory")
    data, manifest = _load_training_data(dataset_dir)
    train, validation = (data[split] for split in ("train", "validation"))
    if len(np.unique(train.targets)) != 2:
        raise BaselineTrainingError("train dataset must contain both binary classes")
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train.features)
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
    model.fit(train_features, train.targets)
    validation_probabilities = model.predict_proba(scaler.transform(validation.features))[:, 1]
    selection = _select_threshold(validation, validation_probabilities)
    validation_metrics = _trade_metrics(
        validation, validation_probabilities >= selection.threshold, validation_probabilities
    )
    validation_metrics["accept_all"] = _trade_metrics(
        validation, np.ones(len(validation.targets), dtype=bool), None
    )
    # Test is evaluated only after the validation-derived threshold is immutable.
    test = _load_split(dataset_dir, manifest, "test")
    data["test"] = test
    test_probabilities = model.predict_proba(scaler.transform(test.features))[:, 1]
    test_metrics = _trade_metrics(
        test, test_probabilities >= selection.threshold, test_probabilities
    )
    test_metrics["accept_all"] = _trade_metrics(test, np.ones(len(test.targets), dtype=bool), None)
    metadata: dict[str, Any] = {
        "model_version": BASELINE_MODEL_VERSION,
        "model_type": "StandardScaler + LogisticRegression",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "ordered_feature_schema": FEATURE_COLUMNS,
        "dataset_generation_id": manifest["dataset_generation_id"],
        "source_generation_id": manifest["source_generation_id"],
        "input_file_sha256": {f"{split}.csv": data[split].checksum for split in DEVELOPMENT_SPLITS},
        "training_configuration": {
            "random_state": 42,
            "regularization_c": 1.0,
            "solver": "lbfgs",
            "max_iter": 1000,
            "fit_split": "train",
            "threshold_selection_split": "validation",
            "test_evaluation_split": "test",
        },
        "threshold_policy": {
            "version": THRESHOLD_POLICY_VERSION,
            "minimum_validation_trades": MIN_VALIDATION_TRADES,
            "objective": "cumulative_net_return_after_costs",
            "tie_breaks": ["mean_net_return", "precision", "trade_count", "higher_threshold"],
        },
        "code_commit": _code_commit(),
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "experimental_only": True,
        "production_eligible": False,
        "live_trading_enabled": False,
    }
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=parent))
    try:
        with (staging / "model.pkl").open("wb") as handle:
            pickle.dump(
                {"scaler": scaler, "model": model}, handle, protocol=pickle.HIGHEST_PROTOCOL
            )
            handle.flush()
            os.fsync(handle.fileno())
        _write_json(staging / "model.metadata.json", metadata)
        _write_json(staging / "validation.metrics.json", validation_metrics)
        _write_json(staging / "test.metrics.json", test_metrics)
        _write_json(staging / "threshold.selection.json", asdict(selection))
        os.replace(staging, output_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return BaselineTrainingSummary(
        manifest["dataset_generation_id"],
        manifest["source_generation_id"],
        selection.threshold,
        validation_metrics,
        test_metrics,
        output_dir,
    )
