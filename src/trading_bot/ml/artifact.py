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


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json(payload: bytes) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ModelArtifactError("model artifact JSON has duplicate keys")
            result[key] = value
        return result

    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelArtifactError("model artifact JSON is invalid") from exc
    if not isinstance(value, dict):
        raise ModelArtifactError("model artifact JSON is invalid")
    return value


def _read_regular(path: Path) -> bytes:
    try:
        if not path.is_file() or path.is_symlink():
            raise ModelArtifactError("model artifact must be a regular file")
        return path.read_bytes()
    except ModelArtifactError:
        raise
    except OSError as exc:
        raise ModelArtifactError("model artifact is unreadable") from exc


def load_sealed_baseline_artifact(
    directory: Path, *, dataset_generation_id: str, source_generation_id: str
) -> SealedBaselineArtifact:
    """Verify every byte and contract field before deserializing model.pkl."""
    try:
        entries = {path.name: path for path in directory.iterdir()}
    except OSError as exc:
        raise ModelArtifactError("model artifact directory is unreadable") from exc
    if set(entries) != {
        *_ARTIFACT_FILES,
        ARTIFACT_MANIFEST_FILENAME,
    }:
        raise ModelArtifactError("model artifact files are incomplete or unexpected")
    snapshots = {name: _read_regular(path) for name, path in entries.items()}
    manifest = _json(snapshots[ARTIFACT_MANIFEST_FILENAME])
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
        not isinstance(value, str) or _sha256(snapshots[name]) != value
        for name, value in files.items()
    ):
        raise ModelArtifactError("model artifact checksum mismatch")
    metadata = _json(snapshots["model.metadata.json"])
    selection = _json(snapshots["threshold.selection.json"])
    _json(snapshots["validation.metrics.json"])
    _json(snapshots["test.metrics.json"])
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
        or any(
            metadata.get(key) != manifest.get(key)
            for key in (
                "model_version",
                "model_type",
                "dataset_generation_id",
                "source_generation_id",
                "ordered_feature_schema",
                "feature_schema_version",
                "experimental_only",
                "production_eligible",
                "live_trading_enabled",
            )
        )
        or metadata.get("model_version") != BASELINE_MODEL_VERSION
        or metadata.get("ordered_feature_schema") != FEATURE_COLUMNS
        or metadata.get("dataset_generation_id") != dataset_generation_id
        or metadata.get("source_generation_id") != source_generation_id
        or metadata.get("threshold_policy", {}).get("version") != THRESHOLD_POLICY_VERSION
        or metadata.get("training_configuration") != manifest.get("training_configuration")
    ):
        raise ModelArtifactError("model artifact contract is invalid")
    try:
        payload = pickle.loads(snapshots["model.pkl"])
    except (
        pickle.UnpicklingError,
        EOFError,
        AttributeError,
        ImportError,
        IndexError,
        KeyError,
    ) as exc:
        raise ModelArtifactError("model artifact pickle is invalid") from exc
    if not isinstance(payload, Mapping) or set(payload) != {"scaler", "model"}:
        raise ModelArtifactError("model artifact pickle is invalid")
    scaler, model = payload.get("scaler"), payload.get("model")
    if not isinstance(scaler, StandardScaler) or not isinstance(model, LogisticRegression):
        raise ModelArtifactError("model artifact type is invalid")
    feature_count = len(FEATURE_COLUMNS)
    if (
        scaler.with_mean is not True
        or scaler.with_std is not True
        or getattr(scaler, "n_features_in_", None) != feature_count
        or getattr(model, "n_features_in_", None) != feature_count
        or not isinstance(getattr(model, "coef_", None), np.ndarray)
        or model.coef_.shape != (1, feature_count)
        or not isinstance(getattr(model, "intercept_", None), np.ndarray)
        or model.intercept_.shape != (1,)
        or not isinstance(getattr(model, "classes_", None), np.ndarray)
        or model.classes_.shape != (2,)
        or model.classes_.tolist() != [0, 1]
        or not all(
            isinstance(getattr(scaler, name, None), np.ndarray)
            and getattr(scaler, name).shape == (feature_count,)
            and np.isfinite(getattr(scaler, name)).all()
            for name in ("mean_", "scale_", "var_")
        )
        or not np.isfinite(model.coef_).all()
        or not np.isfinite(model.intercept_).all()
        or not np.all(scaler.scale_ > 0)
        or model.C != 1.0
        or model.solver != "lbfgs"
        or model.max_iter != 1000
        or model.random_state != 42
    ):
        raise ModelArtifactError("model artifact feature dimensions are invalid")
    return SealedBaselineArtifact(
        scaler, model, float(threshold), dataset_generation_id, source_generation_id
    )
