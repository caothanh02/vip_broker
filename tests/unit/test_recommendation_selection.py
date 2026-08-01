from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from trading_bot.data.csv_store import csv_sha256
from trading_bot.recommendations import experiments, selection


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _development_report(manifest_path: Path, *, decision: str = "selected") -> dict[str, Any]:
    candidate_id = "baseline_ema_volume_atr_v1"
    return {
        "schema_version": "1.0",
        "protocol_version": "development_walk_forward_v1",
        "code_revision": selection.source_revision(),
        "candidate_id": candidate_id,
        "candidate": experiments._CANDIDATES[candidate_id],
        "research_role": "development",
        "strict_oos_evaluation_history": False,
        "research_claim_eligible": False,
        "selection_decision": {
            "decision": decision,
            "selected_candidate_id": candidate_id if decision == "selected" else None,
        },
        "source_manifest": {
            "path": manifest_path.relative_to(Path.cwd()).as_posix(),
            "sha256": csv_sha256(manifest_path),
        },
    }


def _paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    manifest = tmp_path / "reports/research/manifests/development.json"
    _write_json(manifest, {"schema_version": "1.0"})
    report = tmp_path / "reports/research/walk-forward/baseline.json"
    _write_json(report, _development_report(manifest))
    artifact = tmp_path / "reports/research/selections/baseline.json"
    return manifest, report, artifact


def test_selected_development_report_seals_and_validates_exact_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest, report, artifact = _paths(tmp_path)

    created = selection.create_development_selection(
        report.relative_to(tmp_path),
        artifact.relative_to(tmp_path),
        overwrite=False,
        now=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )
    loaded, checksum = selection.validate_development_selection(
        artifact.relative_to(tmp_path), "baseline_ema_volume_atr_v1"
    )

    assert created == loaded
    assert checksum == csv_sha256(artifact)
    assert loaded["selection_decision"] == "selected"
    assert loaded["candidate"] == experiments._CANDIDATES["baseline_ema_volume_atr_v1"]
    assert loaded["development_report"]["sha256"] == csv_sha256(report)
    assert loaded["development_manifest"]["sha256"] == csv_sha256(manifest)


def test_no_policy_selected_report_cannot_create_strict_oos_authorization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest, report, artifact = _paths(tmp_path)
    _write_json(report, _development_report(manifest, decision="no_policy_selected"))

    with pytest.raises(selection.DevelopmentSelectionError, match="did not select"):
        selection.create_development_selection(
            report.relative_to(tmp_path), artifact.relative_to(tmp_path), overwrite=False
        )
    assert not artifact.exists()


def test_fabricated_selected_artifact_cannot_authorize_no_policy_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest, report, artifact = _paths(tmp_path)
    selection.create_development_selection(
        report.relative_to(tmp_path), artifact.relative_to(tmp_path), overwrite=False
    )
    _write_json(report, _development_report(manifest, decision="no_policy_selected"))
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["development_report"]["sha256"] = csv_sha256(report)
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        selection.DevelopmentSelectionError, match="development report is inconsistent"
    ):
        selection.validate_development_selection(
            artifact.relative_to(tmp_path), "baseline_ema_volume_atr_v1"
        )


@pytest.mark.parametrize(
    ("relative_path", "message"),
    [
        (Path("reports/research/selections/../selection.json"), "traversal"),
        (None, "relative"),
    ],
)
def test_unsafe_selection_path_is_rejected_before_json_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    relative_path: Path | None,
    message: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        selection,
        "_object",
        lambda *_: (_ for _ in ()).throw(AssertionError("selection must not be read")),
    )

    path = relative_path if relative_path is not None else tmp_path.parent / "selection.json"
    with pytest.raises(selection.DevelopmentSelectionError, match=message):
        selection.validate_development_selection(path, "baseline_ema_volume_atr_v1")


def test_symlink_selection_path_is_rejected_before_json_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    artifact = Path("reports/research/selections/selection.json")
    original_is_symlink = Path.is_symlink

    def is_symlink(path: Path) -> bool:
        return path == tmp_path / artifact or original_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", is_symlink)
    monkeypatch.setattr(
        selection,
        "_object",
        lambda *_: (_ for _ in ()).throw(AssertionError("selection must not be read")),
    )

    with pytest.raises(selection.DevelopmentSelectionError, match="symlink"):
        selection.validate_development_selection(artifact, "baseline_ema_volume_atr_v1")


@pytest.mark.parametrize("mutation", ["checksum", "candidate", "revision"])
def test_tampered_selection_artifact_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mutation: str
) -> None:
    monkeypatch.chdir(tmp_path)
    _, report, artifact = _paths(tmp_path)
    selection.create_development_selection(
        report.relative_to(tmp_path), artifact.relative_to(tmp_path), overwrite=False
    )
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    if mutation == "checksum":
        payload["development_report"]["sha256"] = "0" * 64
    elif mutation == "candidate":
        payload["candidate"]["cost_model"]["entry_fee_rate"] = "0.002"
    else:
        payload["code_revision"] = "0" * 40
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(selection.DevelopmentSelectionError):
        selection.validate_development_selection(
            artifact.relative_to(tmp_path), "baseline_ema_volume_atr_v1"
        )


def test_selection_module_has_no_execution_or_credential_dependencies() -> None:
    source = Path(selection.__file__).read_text(encoding="utf-8")
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
