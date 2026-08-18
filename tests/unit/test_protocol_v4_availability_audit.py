"""Regression coverage for the mechanical-only Protocol V4 availability audit."""

from __future__ import annotations

import hashlib
import inspect
import io
import json
import zipfile
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from trading_bot.cli import build_parser, main
from trading_bot.recommendations import v4_availability_audit


def _month_rows(
    start: datetime,
    *,
    missing: set[datetime] | None = None,
    early_close: set[datetime] | None = None,
    invalid_early_close: set[datetime] | None = None,
) -> str:
    values: list[str] = []
    cursor = start
    stop = (
        start.replace(month=start.month % 12 + 1, day=1)
        if start.month != 12
        else start.replace(year=start.year + 1, month=1, day=1)
    )
    while cursor < stop:
        if missing is None or cursor not in missing:
            open_ms = int(cursor.timestamp()) * 1000
            close_ms = open_ms + 3_599_999
            if early_close is not None and cursor in early_close:
                close_ms -= 1_000
            if invalid_early_close is not None and cursor in invalid_early_close:
                close_ms -= 61_000
            values.append(f"{open_ms},1,1,1,1,1,{close_ms},1,1,1,1,0\n")
        cursor += timedelta(hours=1)
    return "".join(values)


def _zip(archive_name: str, rows: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(archive_name.removesuffix(".zip") + ".csv", rows)
    return buffer.getvalue()


def _transport(payload: bytes, archive_name: str) -> httpx.MockTransport:
    checksum = hashlib.sha256(payload).hexdigest()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith(".CHECKSUM"):
            return httpx.Response(200, text=f"{checksum}  {archive_name}\n")
        if request.url.path.endswith(archive_name):
            return httpx.Response(200, content=payload)
        return httpx.Response(404)

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_v4_audit_verifies_continuous_month_without_selecting_anything() -> None:
    start = datetime(2021, 1, 1, tzinfo=UTC)
    archive_name = "BTCUSDT-1h-2021-01.zip"
    payload = _zip(archive_name, _month_rows(start))

    report = await v4_availability_audit.audit_protocol_v4_availability(
        start,
        datetime(2021, 2, 1, tzinfo=UTC),
        transport=_transport(payload, archive_name),
    )

    assert report["result"] == "availability_verified_not_selected"
    assert report["absolute_continuity"] is True
    assert report["selection_authorized"] is False
    assert report["execution_authorized"] is False
    assert report["strict_oos_authorized"] is False
    assert report["safety_locks"] == {
        "default_recommendation": "NEUTRAL",
        "recommendation_engine_used": False,
        "signal_or_feature_computed": False,
        "performance_metric_computed": False,
        "data_persisted": False,
        "broker_used": False,
        "orders_submitted": False,
        "ml_used": False,
        "authenticated_api_used": False,
        "strict_oos_read": False,
        "network_used": True,
    }
    archive = report["archives"][0]
    assert isinstance(archive, dict)
    assert archive["expected_candle_count"] == 744
    assert archive["observed_candle_count"] == 744
    assert archive["accepted_timestamp_anomaly_count"] == 0


@pytest.mark.asyncio
async def test_v4_audit_accepts_the_existing_verified_early_close_policy() -> None:
    start = datetime(2021, 1, 1, tzinfo=UTC)
    archive_name = "BTCUSDT-1h-2021-01.zip"
    payload = _zip(
        archive_name,
        _month_rows(start, early_close={datetime(2021, 1, 2, 3, tzinfo=UTC)}),
    )

    report = await v4_availability_audit.audit_protocol_v4_availability(
        start,
        datetime(2021, 2, 1, tzinfo=UTC),
        transport=_transport(payload, archive_name),
    )

    assert report["result"] == "availability_verified_not_selected"
    archive = report["archives"][0]
    assert isinstance(archive, dict)
    assert archive["accepted_timestamp_anomaly_count"] == 1


@pytest.mark.asyncio
async def test_v4_audit_keeps_checksum_evidence_when_timestamp_policy_fails() -> None:
    start = datetime(2021, 1, 1, tzinfo=UTC)
    archive_name = "BTCUSDT-1h-2021-01.zip"
    payload = _zip(
        archive_name,
        _month_rows(start, invalid_early_close={datetime(2021, 1, 2, 3, tzinfo=UTC)}),
    )

    report = await v4_availability_audit.audit_protocol_v4_availability(
        start,
        datetime(2021, 2, 1, tzinfo=UTC),
        transport=_transport(payload, archive_name),
    )

    assert report["official_checksums_verified"] is True
    archive = report["archives"][0]
    assert isinstance(archive, dict)
    assert archive["checksum_verified"] is True
    assert "timestamp policy" in str(archive["error"])


@pytest.mark.asyncio
async def test_v4_audit_reports_a_gap_without_fallback_or_authorization() -> None:
    start = datetime(2021, 1, 1, tzinfo=UTC)
    missing = {datetime(2021, 1, 12, 2, tzinfo=UTC)}
    archive_name = "BTCUSDT-1h-2021-01.zip"
    payload = _zip(archive_name, _month_rows(start, missing=missing))

    report = await v4_availability_audit.audit_protocol_v4_availability(
        start,
        datetime(2021, 2, 1, tzinfo=UTC),
        transport=_transport(payload, archive_name),
    )

    assert report["result"] == "availability_not_verified"
    assert report["absolute_continuity"] is False
    assert report["selection_authorized"] is False
    archive = report["archives"][0]
    assert isinstance(archive, dict)
    assert archive["missing_open_times"] == ["2021-01-12T02:00:00Z"]
    assert archive["observed_candle_count"] == 743


@pytest.mark.asyncio
async def test_v4_audit_rejects_strict_oos_before_any_network_access() -> None:
    with pytest.raises(v4_availability_audit.ProtocolV4AvailabilityAuditError, match="strict OOS"):
        await v4_availability_audit.audit_protocol_v4_availability(
            datetime(2025, 1, 1, tzinfo=UTC),
            datetime(2025, 2, 1, tzinfo=UTC),
            transport=httpx.MockTransport(
                lambda request: (_ for _ in ()).throw(AssertionError("must not request"))
            ),
        )


def test_v4_audit_rejects_performance_selector_terms() -> None:
    with pytest.raises(v4_availability_audit.ProtocolV4AvailabilityAuditError, match="prohibited"):
        v4_availability_audit.assert_no_performance_inputs(["accuracy"])


def test_v4_audit_has_no_persistence_or_trading_dependencies() -> None:
    source = inspect.getsource(v4_availability_audit)
    for forbidden in (
        "RecommendationEngine",
        "RiskEngine",
        "BinanceBroker",
        "DryRunBroker",
        "OrderRequest",
        "ProbabilityModel",
        "trading_bot.settings",
        "BinanceHistoricalDataClient",
        "BinancePublicClient",
        "write_json",
        "write_text",
        "write_bytes",
    ):
        assert forbidden not in source


def test_v4_availability_cli_stays_isolated_from_runtime_dependencies(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def audit(start: datetime, end: datetime) -> dict[str, object]:
        assert start == datetime(2021, 1, 1, tzinfo=UTC)
        assert end == datetime(2021, 2, 1, tzinfo=UTC)
        return {"result": "availability_not_verified", "selection_authorized": False}

    monkeypatch.setattr(v4_availability_audit, "audit_protocol_v4_availability", audit)
    monkeypatch.setattr(
        "trading_bot.cli._load_non_operational_dependencies",
        lambda: (_ for _ in ()).throw(AssertionError("must not load runtime dependencies")),
    )

    assert (
        main(
            [
                "audit-protocol-v4-availability",
                "--start",
                "2021-01-01T00:00:00Z",
                "--end",
                "2021-02-01T00:00:00Z",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == {
        "result": "availability_not_verified",
        "selection_authorized": False,
    }
    assert "audit-protocol-v4-availability" in build_parser().format_help()
