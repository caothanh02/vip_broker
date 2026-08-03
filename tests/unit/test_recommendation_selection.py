from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from trading_bot.data.csv_store import csv_sha256
from trading_bot.recommendations import experiments, selection, walk_forward
from trading_bot.settings import BotSettings

_IDENTITY: dict[str, Any] = {
    "schema_version": "1.0",
    "revision": "a" * 40,
    "tracked_objects": {
        "src/trading_bot": "b" * 40,
        "pyproject.toml": "c" * 40,
        "uv.lock": "d" * 40,
    },
}


def _identity() -> dict[str, Any]:
    return _IDENTITY


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _metrics(*, passed: bool = True) -> dict[str, Any]:
    sample = 30 if passed else 29
    accuracy = 0.60 if passed else 0.50
    return {
        "horizons": {
            horizon: {
                "applicable_resolved_count": sample,
                "non_neutral_coverage": 0.10,
                "after_cost_directional_accuracy": accuracy,
                "neutral_rate": 0.50,
                "statistical_result": {
                    "two_sided_95_percent_exact_lower_bound": 0.55 if passed else 0.49
                },
            }
            for horizon in ("1h", "4h", "24h")
        }
    }


def _development_report(
    manifest_path: Path, *, gate_passed: bool = True, decision: str = "selected"
) -> dict[str, Any]:
    candidate_id = "baseline_ema_volume_atr_v1"
    folds = [
        {"fold_id": fold.identifier, "metrics": _metrics(passed=gate_passed)}
        for fold in walk_forward.FOLDS
    ]
    pooled = _metrics(passed=gate_passed)
    if gate_passed:
        for horizon in pooled["horizons"].values():
            horizon["applicable_resolved_count"] = 100
    gate = walk_forward.development_selection_gate([fold["metrics"] for fold in folds], pooled)
    selection_decision = walk_forward.select_candidate(
        {
            candidate_id: {
                "fold_metrics": folds,
                "pooled_metrics": pooled,
                "selection_gate": gate,
            }
        }
    )
    if decision != selection_decision["decision"]:
        selection_decision = {
            "decision": decision,
            "selected_candidate_id": candidate_id if decision == "selected" else None,
        }
    return {
        "schema_version": "1.1",
        "protocol_version": "development_walk_forward_v1",
        "code_revision": _IDENTITY["revision"],
        "source_identity": _IDENTITY,
        "run_at": "2026-01-01T00:00:00Z",
        "candidate_id": candidate_id,
        "candidate": experiments._CANDIDATES[candidate_id],
        "research_role": "development",
        "strict_oos_evaluation_history": False,
        "research_claim_eligible": False,
        "research_claim_eligibility_reason": "development_dataset_not_strict_oos",
        "folds": folds,
        "pooled_metrics": pooled,
        "selection_gate": gate,
        "selection_decision": selection_decision,
        "source_manifest": {
            "path": manifest_path.relative_to(Path.cwd()).as_posix(),
            "sha256": csv_sha256(manifest_path),
        },
        "dataset": {
            "path": "data/raw/development.csv",
            "csv_sha256": "a" * 64,
            "generation_id": "generation",
            "range": {"start": "2022-01-01T00:00:00Z", "end": "2025-01-01T00:00:00Z"},
            "candle_count": 1,
        },
        "disclaimer": "development only",
        "safety_locks": {
            "live_trading_enabled": False,
            "broker_used": False,
            "orders_submitted": False,
            "ml_inference_used": False,
        },
    }


def _paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    manifest = tmp_path / "reports/research/manifests/development.json"
    _write_json(manifest, {"schema_version": "1.0"})
    report = tmp_path / "reports/research/walk-forward/baseline.json"
    _write_json(report, _development_report(manifest))
    artifact = tmp_path / "reports/research/selections/baseline.json"
    return manifest, report, artifact


def _replay(
    expected: dict[str, Any],
) -> Callable[[dict[str, Any], str, Callable[[], dict[str, Any]]], dict[str, Any]]:
    return lambda *_: deepcopy(expected)


def _create(report: Path, artifact: Path, tmp_path: Path) -> dict[str, Any]:
    expected = json.loads(report.read_text(encoding="utf-8"))
    return selection.create_development_selection(
        report.relative_to(tmp_path),
        artifact.relative_to(tmp_path),
        overwrite=False,
        identity=_identity,
        replay=_replay(expected),
    )


def _validate(
    artifact: Path, tmp_path: Path, expected: dict[str, Any] | None = None
) -> tuple[dict[str, Any], str]:
    expected_report = (
        expected
        if expected is not None
        else json.loads(
            (tmp_path / "reports/research/walk-forward/baseline.json").read_text(encoding="utf-8")
        )
    )
    return selection.validate_development_selection(
        artifact.relative_to(tmp_path),
        "baseline_ema_volume_atr_v1",
        identity=_identity,
        replay=_replay(expected_report),
    )


def test_selected_development_report_seals_and_validates_exact_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest, report, artifact = _paths(tmp_path)

    created = _create(report, artifact, tmp_path)
    loaded, checksum = _validate(artifact, tmp_path)

    assert created == loaded
    assert checksum == csv_sha256(artifact)
    assert loaded["selection_decision"] == "selected"
    assert loaded["source_identity"] == _IDENTITY
    assert loaded["candidate"] == experiments._CANDIDATES["baseline_ema_volume_atr_v1"]
    assert loaded["development_report"]["sha256"] == csv_sha256(report)
    assert loaded["development_manifest"]["sha256"] == csv_sha256(manifest)


@pytest.mark.parametrize(
    ("gate_passed", "decision"),
    [
        (False, "selected"),
        (True, "no_policy_selected"),
    ],
)
def test_untrusted_selection_decision_cannot_bypass_fold_or_pooled_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, gate_passed: bool, decision: str
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest, report, artifact = _paths(tmp_path)
    _write_json(report, _development_report(manifest, gate_passed=gate_passed, decision=decision))

    with pytest.raises(selection.DevelopmentSelectionError, match="did not select"):
        _create(report, artifact, tmp_path)
    assert not artifact.exists()


def test_fabricated_selected_artifact_cannot_authorize_no_policy_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest, report, artifact = _paths(tmp_path)
    expected = json.loads(report.read_text(encoding="utf-8"))
    _create(report, artifact, tmp_path)
    _write_json(report, _development_report(manifest, gate_passed=False, decision="selected"))
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["development_report"]["sha256"] = csv_sha256(report)
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(selection.DevelopmentSelectionError, match="did not select"):
        _validate(artifact, tmp_path, expected)


def test_report_missing_source_identity_is_rejected_before_artifact_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    _, report, artifact = _paths(tmp_path)
    payload = json.loads(report.read_text(encoding="utf-8"))
    del payload["source_identity"]
    report.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(selection.DevelopmentSelectionError, match="schema"):
        _create(report, artifact, tmp_path)
    assert not artifact.exists()


def test_synchronously_tampered_metrics_and_selection_are_rejected_by_replay(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest, report, artifact = _paths(tmp_path)
    expected = json.loads(report.read_text(encoding="utf-8"))
    _create(report, artifact, tmp_path)
    tampered = _development_report(manifest)
    for fold in tampered["folds"]:
        for horizon in fold["metrics"]["horizons"].values():
            horizon["after_cost_directional_accuracy"] = 0.61
    for horizon in tampered["pooled_metrics"]["horizons"].values():
        horizon["statistical_result"]["two_sided_95_percent_exact_lower_bound"] = 0.56
    tampered["selection_gate"] = walk_forward.development_selection_gate(
        [fold["metrics"] for fold in tampered["folds"]], tampered["pooled_metrics"]
    )
    tampered["selection_decision"] = walk_forward.select_candidate(
        {
            "baseline_ema_volume_atr_v1": {
                "fold_metrics": tampered["folds"],
                "pooled_metrics": tampered["pooled_metrics"],
                "selection_gate": tampered["selection_gate"],
            }
        }
    )
    _write_json(report, tampered)
    artifact_payload = json.loads(artifact.read_text(encoding="utf-8"))
    artifact_payload["development_report"]["sha256"] = csv_sha256(report)
    artifact.write_text(json.dumps(artifact_payload), encoding="utf-8")

    with pytest.raises(selection.DevelopmentSelectionError, match="deterministic replay"):
        _validate(artifact, tmp_path, expected)


def test_replay_uses_in_memory_walk_forward_builder_with_locked_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    _, report, _ = _paths(tmp_path)
    expected = json.loads(report.read_text(encoding="utf-8"))
    observed: dict[str, Any] = {}

    def build(
        manifest_path: Path,
        candidate_id: str,
        settings: BotSettings,
        *,
        identity: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        observed["manifest_path"] = manifest_path
        observed["candidate_id"] = candidate_id
        observed["settings"] = settings
        observed["identity"] = identity
        return expected

    from trading_bot.recommendations import walk_forward

    monkeypatch.setattr(walk_forward, "build_development_walk_forward_report", build)

    assert (
        selection._replay_development_report(expected, "baseline_ema_volume_atr_v1", _identity)
        == expected
    )
    assert observed["manifest_path"] == Path("reports/research/manifests/development.json")
    assert observed["candidate_id"] == "baseline_ema_volume_atr_v1"
    assert observed["settings"] == BotSettings()
    assert observed["identity"] is _identity


@pytest.mark.parametrize("tamper", ["candidate", "cost", "report_sha", "manifest_sha"])
def test_candidate_cost_and_checksum_tampering_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tamper: str
) -> None:
    monkeypatch.chdir(tmp_path)
    _, report, artifact = _paths(tmp_path)
    _create(report, artifact, tmp_path)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    if tamper == "candidate":
        payload["candidate_id"] = "unknown_candidate"
    elif tamper == "cost":
        payload["candidate"]["cost_model"]["entry_fee_rate"] = "0.002"
    elif tamper == "report_sha":
        payload["development_report"]["sha256"] = "0" * 64
    else:
        payload["development_manifest"]["sha256"] = "0" * 64
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(selection.DevelopmentSelectionError):
        _validate(artifact, tmp_path)


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
        selection.validate_development_selection(
            path, "baseline_ema_volume_atr_v1", identity=_identity
        )


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
        selection.validate_development_selection(
            artifact, "baseline_ema_volume_atr_v1", identity=_identity
        )


@pytest.mark.parametrize(
    "status", [" M src/trading_bot/recommendations/engine.py", "M  src/trading_bot/cli.py"]
)
def test_staged_and_unstaged_executable_source_changes_block_seal_and_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, status: str
) -> None:
    monkeypatch.chdir(tmp_path)
    _, report, artifact = _paths(tmp_path)
    monkeypatch.setattr(selection, "_source_worktree_entries", lambda: [status])

    with pytest.raises(selection.DevelopmentSelectionError, match="not clean"):
        selection.create_development_selection(
            report.relative_to(tmp_path), artifact.relative_to(tmp_path), overwrite=False
        )
    _create(report, artifact, tmp_path)
    with pytest.raises(selection.DevelopmentSelectionError, match="not clean"):
        selection.validate_development_selection(
            artifact.relative_to(tmp_path), "baseline_ema_volume_atr_v1"
        )


def test_source_identity_ignores_docs_and_generated_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(selection, "_source_worktree_entries", lambda: [])
    monkeypatch.setattr(selection, "source_revision", lambda: "a" * 40)
    monkeypatch.setattr(selection, "_git_output", lambda *_: "b" * 40)

    identity = selection.source_identity()

    assert identity["revision"] == "a" * 40
    assert set(identity["tracked_objects"]) == {"src/trading_bot", "pyproject.toml", "uv.lock"}


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
