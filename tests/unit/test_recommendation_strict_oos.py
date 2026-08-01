from __future__ import annotations

import inspect
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from trading_bot.domain.models import Candle, Recommendation, RecommendationType
from trading_bot.recommendations import strict_oos
from trading_bot.recommendations.selection import DevelopmentSelectionError
from trading_bot.settings import BotSettings


def _candle(open_time: datetime) -> Candle:
    return Candle(
        open_time,
        open_time + timedelta(hours=1),
        "BTC/USDT",
        "1h",
        Decimal("100"),
        Decimal("101"),
        Decimal("99"),
        Decimal("100"),
        Decimal("1000"),
        True,
    )


def _recommendation(signal_time: datetime) -> Recommendation:
    return Recommendation(
        "oos-recommendation",
        signal_time,
        signal_time,
        "BTC/USDT",
        "1h",
        ("1h", "4h", "24h"),
        RecommendationType.BUY_BIAS,
        None,
        0.55,
        None,
        "test_feature_schema",
        "ema_volume_atr_rule_candidate_rule_only",
        "validated_closed_contiguous",
        Decimal("100"),
        Decimal("98"),
        Decimal("104"),
    )


def _selection() -> dict[str, Any]:
    return {
        "candidate_id": "baseline_ema_volume_atr_v1",
        "candidate": strict_oos.experiments._CANDIDATES["baseline_ema_volume_atr_v1"],
        "development_report": {
            "path": "reports/research/walk-forward/selected.json",
            "sha256": "a" * 64,
        },
        "development_manifest": {
            "path": "reports/research/manifests/development.json",
            "sha256": "b" * 64,
        },
    }


def _allow_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        strict_oos,
        "validate_development_selection",
        lambda *_args, **_kwargs: (_selection(), "c" * 64),
    )


def test_strict_oos_output_rejects_traversal_absolute_and_symlink_before_manifest_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    _allow_selection(monkeypatch)
    monkeypatch.setattr(
        strict_oos,
        "_manifest_snapshot",
        lambda *_: (_ for _ in ()).throw(AssertionError("manifest must not be read")),
    )
    for output in (
        Path("reports/research/strict-oos/../report.json"),
        tmp_path / "reports/research/strict-oos/report.json",
        Path("reports/research/report.json"),
    ):
        with pytest.raises(strict_oos.StrictOosError, match="output"):
            strict_oos.run_strict_oos_evaluation(
                Path("reports/research/manifests/oos.json"),
                "baseline_ema_volume_atr_v1",
                output,
                Path("reports/research/selections/selected.json"),
                BotSettings(),
                overwrite=False,
            )


def test_strict_oos_rejects_candidate_and_cost_mismatch_before_manifest_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        strict_oos,
        "_manifest_snapshot",
        lambda *_: (_ for _ in ()).throw(AssertionError("manifest must not be read")),
    )
    output = Path("reports/research/strict-oos/report.json")
    selection = Path("reports/research/selections/selected.json")
    with pytest.raises(strict_oos.StrictOosError, match="unregistered"):
        strict_oos.run_strict_oos_evaluation(
            Path("reports/research/manifests/oos.json"),
            "unknown",
            output,
            selection,
            BotSettings(),
            overwrite=False,
        )
    with pytest.raises(strict_oos.StrictOosError, match="cost rate"):
        strict_oos.run_strict_oos_evaluation(
            Path("reports/research/manifests/oos.json"),
            "baseline_ema_volume_atr_v1",
            output,
            selection,
            BotSettings(entry_fee_rate=Decimal("0.002")),
            overwrite=False,
        )


def test_no_policy_selection_blocks_strict_freeze_and_evaluation_before_oos_reads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)

    def reject_selection(*_args: object, **_kwargs: object) -> tuple[dict[str, Any], str]:
        raise DevelopmentSelectionError("development report did not select a policy")

    monkeypatch.setattr(strict_oos, "validate_development_selection", reject_selection)
    monkeypatch.setattr(
        strict_oos,
        "_inside_workspace",
        lambda *_: (_ for _ in ()).throw(AssertionError("OOS dataset must not be read")),
    )
    with pytest.raises(strict_oos.StrictOosError, match="did not select"):
        strict_oos.freeze_strict_oos_dataset(
            Path("data/raw/oos.csv"),
            Path("reports/research/manifests/oos.json"),
            Path("reports/research/selections/selection.json"),
            "baseline_ema_volume_atr_v1",
            overwrite=False,
        )

    monkeypatch.setattr(
        strict_oos,
        "_manifest_snapshot",
        lambda *_: (_ for _ in ()).throw(AssertionError("OOS manifest must not be read")),
    )
    with pytest.raises(strict_oos.StrictOosError, match="did not select"):
        strict_oos.run_strict_oos_evaluation(
            Path("reports/research/manifests/oos.json"),
            "baseline_ema_volume_atr_v1",
            Path("reports/research/strict-oos/result.json"),
            Path("reports/research/selections/selection.json"),
            BotSettings(),
            overwrite=False,
        )


def test_manifest_selection_checksum_mismatch_rejects_before_oos_dataset_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    selection = _selection()
    manifest = {
        "schema_version": "1.1",
        "created_at": "2026-01-01T00:00:00Z",
        "research_role": "strict_oos",
        "strict_oos_evaluation_history": True,
        "strict_oos_start": "2025-01-01T00:00:00Z",
        "code_revision": "d" * 40,
        "selection_artifact": {"sha256": "0" * 64, **selection},
        "dataset": {},
        "market_interruptions": [],
        "safety_locks": {},
    }
    monkeypatch.setattr(strict_oos, "_manifest_path", lambda _: tmp_path / "oos.json")
    monkeypatch.setattr(strict_oos, "_read_object", lambda *_: manifest)
    monkeypatch.setattr(strict_oos, "source_revision", lambda: "d" * 40)
    monkeypatch.setattr(
        strict_oos,
        "_verify_oos_dataset",
        lambda *_: (_ for _ in ()).throw(AssertionError("OOS CSV must not be read")),
    )

    with pytest.raises(strict_oos.StrictOosError, match="provenance"):
        strict_oos._manifest_snapshot(tmp_path / "oos.json", selection, "c" * 64)


def test_strict_oos_report_binds_selected_artifact_revision_and_incomplete_outcomes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    _allow_selection(monkeypatch)
    monkeypatch.setattr(strict_oos, "source_revision", lambda: "d" * 40)
    candles = [_candle(strict_oos._OOS_START + timedelta(hours=index)) for index in range(25)]
    dataset = {
        "path": "data/raw/oos.csv",
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "range": {"start": "2025-01-01T00:00:00Z", "end": "2026-01-01T00:00:00Z"},
        "candle_count": 8760,
        "csv_sha256": "a" * 64,
        "metadata_sha256": "b" * 64,
        "anomaly_sidecar_sha256": "c" * 64,
        "generation_id": "generation",
        "validation_status": "valid",
        "checksum_verification_mode": "official_online",
    }
    verified = strict_oos._VerifiedOosInput(
        tmp_path / "data/raw/oos.csv", candles, set(), dataset, []
    )
    manifest_path = Path("reports/research/manifests/oos.json")
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(strict_oos, "_manifest_snapshot", lambda *_: ({}, verified, "d" * 64))
    monkeypatch.setattr(
        strict_oos,
        "backfill_recommendations",
        lambda *_: [_recommendation(candles[-1].close_time)],
    )
    output = Path("reports/research/strict-oos/result.json")

    result = strict_oos.run_strict_oos_evaluation(
        manifest_path,
        "baseline_ema_volume_atr_v1",
        output,
        Path("reports/research/selections/selected.json"),
        BotSettings(),
        overwrite=False,
        now=lambda: datetime(2026, 2, 1, tzinfo=UTC),
    )

    assert result["strict_oos_evaluation_history"] is True
    assert result["research_role"] == "strict_oos"
    assert result["code_revision"] == "d" * 40
    assert result["selection_artifact"]["sha256"] == "c" * 64
    assert result["selection_artifact"]["candidate"] == _selection()["candidate"]
    assert result["metrics"]["strict_oos"] is True
    assert result["research_claim_eligible"] is False
    assert result["metrics"]["horizons"]["24h"]["resolved_recommendations"] == 0
    assert result["metrics"]["horizons"]["24h"]["applicable_resolved_count"] == 0
    assert result["metrics"]["horizons"]["24h"]["outcome_count"] == 1
    assert result["metrics"]["horizons"]["24h"]["insufficient_future_data_count"] == 1
    assert (tmp_path / output).is_file()


def test_strict_oos_path_has_no_execution_or_credential_dependencies() -> None:
    source = inspect.getsource(strict_oos)
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
