"""Fail-closed loader for the experimental offline baseline artifact."""

from __future__ import annotations

import hashlib
import json
import math
import pickle
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from trading_bot.features.pipeline import FEATURE_COLUMNS
from trading_bot.ml.baseline import (
    ARTIFACT_MANIFEST_FILENAME,
    BASELINE_MODEL_VERSION,
    THRESHOLD_POLICY_VERSION,
)

_ARTIFACT_FILES: Final = {
    "model.pkl",
    "model.metadata.json",
    "validation.metrics.json",
    "test.metrics.json",
    "threshold.selection.json",
}


class ModelArtifactError(ValueError):
    """The sealed experimental artifact is absent, partial, or tampered."""


@dataclass(frozen=True, slots=True)
class SealedBaselineArtifact:
    scaler: StandardScaler
    model: LogisticRegression
    threshold: float
    dataset_generation_id: str
    source_generation_id: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ModelArtifactError("model artifact JSON has duplicate keys")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelArtifactError("model artifact JSON is invalid") from exc
    if not isinstance(value, dict):
        raise ModelArtifactError("model artifact JSON is invalid")
    return value


def load_sealed_baseline_artifact(
    directory: Path, *, dataset_generation_id: str, source_generation_id: str
) -> SealedBaselineArtifact:
    """Verify every byte and contract field before deserializing model.pkl."""
    manifest_path = directory / ARTIFACT_MANIFEST_FILENAME
    manifest = _json(manifest_path)
    if set(path.name for path in directory.iterdir()) != {
        *_ARTIFACT_FILES,
        ARTIFACT_MANIFEST_FILENAME,
    }:
        raise ModelArtifactError("model artifact files are incomplete or unexpected")
    files = manifest.get("files")
    if (
        manifest.get("artifact_schema_version") != "1.0.0"
        or manifest.get("model_artifact_version") != BASELINE_MODEL_VERSION
        or manifest.get("dataset_generation_id") != dataset_generation_id
        or manifest.get("source_generation_id") != source_generation_id
        or manifest.get("ordered_feature_schema") != FEATURE_COLUMNS
        or manifest.get("threshold_policy_version") != THRESHOLD_POLICY_VERSION
        or manifest.get("model_type") != "StandardScaler + LogisticRegression"
        or manifest.get("experimental_only") is not True
        or manifest.get("production_eligible") is not False
        or manifest.get("live_trading_enabled") is not False
        or not isinstance(files, dict)
        or set(files) != _ARTIFACT_FILES
    ):
        raise ModelArtifactError("model artifact manifest is invalid")
    if any(
        not isinstance(value, str) or _sha256(directory / name) != value
        for name, value in files.items()
    ):
        raise ModelArtifactError("model artifact checksum mismatch")
    metadata = _json(directory / "model.metadata.json")
    selection = _json(directory / "threshold.selection.json")
    threshold = selection.get("threshold")
    if (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or not math.isfinite(threshold)
    ):
        raise ModelArtifactError("model artifact threshold is invalid")
    if (
        manifest.get("threshold") != threshold
        or not 0 <= float(threshold) <= 1
        or metadata.get("ordered_feature_schema") != FEATURE_COLUMNS
        or metadata.get("dataset_generation_id") != dataset_generation_id
        or metadata.get("source_generation_id") != source_generation_id
    ):
        raise ModelArtifactError("model artifact contract is invalid")
    try:
        with (directory / "model.pkl").open("rb") as handle:
            payload = pickle.load(handle)
    except (OSError, pickle.UnpicklingError, EOFError, AttributeError, ImportError) as exc:
        raise ModelArtifactError("model artifact pickle is invalid") from exc
    if not isinstance(payload, Mapping):
        raise ModelArtifactError("model artifact pickle is invalid")
    scaler, model = payload.get("scaler"), payload.get("model")
    if not isinstance(scaler, StandardScaler) or not isinstance(model, LogisticRegression):
        raise ModelArtifactError("model artifact type is invalid")
    feature_count = len(FEATURE_COLUMNS)
    if (
        getattr(scaler, "n_features_in_", None) != feature_count
        or getattr(model, "n_features_in_", None) != feature_count
        or not isinstance(getattr(model, "coef_", None), np.ndarray)
        or model.coef_.shape[1] != feature_count
        or not isinstance(getattr(model, "intercept_", None), np.ndarray)
        or not isinstance(getattr(model, "classes_", None), np.ndarray)
        or model.classes_.tolist() != [0, 1]
        or not all(
            isinstance(getattr(scaler, name, None), np.ndarray)
            and getattr(scaler, name).shape == (feature_count,)
            and np.isfinite(getattr(scaler, name)).all()
            for name in ("mean_", "scale_", "var_")
        )
        or not np.isfinite(model.coef_).all()
        or not np.isfinite(model.intercept_).all()
    ):
        raise ModelArtifactError("model artifact feature dimensions are invalid")
    return SealedBaselineArtifact(
        scaler, model, float(threshold), dataset_generation_id, source_generation_id
    )
