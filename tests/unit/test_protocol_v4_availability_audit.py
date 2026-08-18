"""The closed V4 audit command must fail before network or runtime imports."""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest

from trading_bot.cli import main
from trading_bot.recommendations import v4_availability_audit


@pytest.mark.asyncio
async def test_closed_v4_rejects_audit_before_a_public_request() -> None:
    with pytest.raises(v4_availability_audit.ProtocolV4AvailabilityAuditError, match="closed"):
        await v4_availability_audit.audit_protocol_v4_availability(
            datetime(2021, 1, 1, tzinfo=UTC),
            datetime(2021, 2, 1, tzinfo=UTC),
            transport=httpx.MockTransport(
                lambda request: (_ for _ in ()).throw(AssertionError("must not request"))
            ),
        )


def test_closed_v4_cli_rejects_before_runtime_dependency_load(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
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
        == 1
    )
    assert "closed" in capsys.readouterr().err
