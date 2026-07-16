from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from trading_bot.domain.models import MLScore, ModelMetadata
from trading_bot.features.pipeline import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION


def triple_barrier_labels(
    frame: pd.DataFrame, candidate: pd.Series, max_holding: int = 48
) -> pd.Series:
    labels = pd.Series(index=frame.index, dtype="object")
    for i in frame.index[candidate.fillna(False)]:
        idx = frame.index.get_loc(i)
        if idx + 1 >= len(frame) or pd.isna(frame.iloc[idx].atr14):
            continue
        entry = frame.iloc[idx + 1].open
        atr = frame.iloc[idx].atr14
        stop, target = entry - 2 * atr, entry + 4 * atr
        outcome: int | str = "timeout"
        for _, row in frame.iloc[idx + 1 : idx + 1 + max_holding].iterrows():
            if row.low <= stop:
                outcome = 0
                break
            if row.high >= target:
                outcome = 1
                break
        labels.loc[i] = outcome
    return labels


def build_dataset(frame: pd.DataFrame, candidate: pd.Series) -> pd.DataFrame:
    data = frame.loc[:, FEATURE_COLUMNS].copy()
    data["label"] = triple_barrier_labels(frame, candidate)
    return data.dropna().query("label != 'timeout'").astype({"label": "int"})


def train_logistic(
    dataset: pd.DataFrame, threshold: float = 0.65
) -> tuple[Pipeline, dict[str, float]]:
    x, y = dataset[FEATURE_COLUMNS], dataset.label
    split = max(1, int(len(dataset) * 0.7))
    model = Pipeline(
        [("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1000, random_state=7))]
    )
    model.fit(x.iloc[:split], y.iloc[:split])
    prob = model.predict_proba(x.iloc[split:])[:, 1]
    truth = y.iloc[split:]
    metrics = {
        "precision": precision_score(truth, prob >= threshold, zero_division=0),
        "recall": recall_score(truth, prob >= threshold, zero_division=0),
        "f1": f1_score(truth, prob >= threshold, zero_division=0),
        "roc_auc": roc_auc_score(truth, prob) if truth.nunique() > 1 else 0.0,
        "pr_auc": average_precision_score(truth, prob),
        "brier": brier_score_loss(truth, prob),
    }
    return model, metrics


def score(model: Pipeline, values: pd.DataFrame, version: str, threshold: float) -> MLScore:
    probability = float(model.predict_proba(values[FEATURE_COLUMNS])[-1, 1])
    return MLScore(
        version, probability, threshold, probability >= threshold, FEATURE_SCHEMA_VERSION
    )


def save_metadata(
    directory: Path, version: str, metrics: dict[str, float], threshold: float
) -> ModelMetadata:
    directory.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {"version": version, "features": FEATURE_COLUMNS, "metrics": metrics}, sort_keys=True
    ).encode()
    checksum = hashlib.sha256(payload).hexdigest()
    meta = ModelMetadata(
        version,
        "logistic_regression",
        pd.Timestamp.now(tz="UTC").isoformat(),
        FEATURE_SCHEMA_VERSION,
        FEATURE_COLUMNS,
        threshold,
        {},
        metrics,
        checksum,
    )
    (directory / f"{version}.metadata.json").write_text(
        json.dumps(asdict(meta), default=str, indent=2), encoding="utf-8"
    )
    return meta


def dummy_baseline(dataset: pd.DataFrame) -> float:
    split = max(1, int(len(dataset) * 0.7))
    model = DummyClassifier(strategy="prior").fit(
        dataset[FEATURE_COLUMNS].iloc[:split], dataset.label.iloc[:split]
    )
    return float(model.score(dataset[FEATURE_COLUMNS].iloc[split:], dataset.label.iloc[split:]))
