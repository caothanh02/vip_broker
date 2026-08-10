"""Regression coverage for the Windows-only, read-only operational runbook.

The behavioural checks execute the real PowerShell runner with fake ``git`` and
``uv`` commands.  On non-Windows hosts without PowerShell, the same tests retain
the static contract assertions so the source remains reviewable in Linux CI.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "run-operational-check.ps1"
INSTALLER = ROOT / "scripts" / "install-operational-check-task.ps1"


def _powershell() -> str | None:
    return shutil.which("powershell.exe") or shutil.which("powershell") or shutil.which("pwsh")


def _source_contract() -> tuple[str, str]:
    runner = RUNNER.read_text(encoding="utf-8")
    installer = INSTALLER.read_text(encoding="utf-8")
    assert '"run", "--offline", "trading-bot", $Subcommand' in runner
    assert '[ValidateSet("audit-safety", "operational-status")]' in runner
    assert "ConvertFrom-Json -ErrorAction Stop" in runner
    assert "Assert-AuditPayload -Payload $auditPayload" in runner
    assert "Assert-StatusPayload -Payload $statusPayload" in runner
    assert "ReparsePoint" in runner
    assert "Get-Item -LiteralPath $current" in runner
    assert "--overwrite" not in runner
    assert "Register-ScheduledTask" in installer
    assert "if (-not $Install.IsPresent)" in installer
    assert "New-ScheduledTaskTrigger -Daily -At 09:00AM" in installer
    assert "-LogonType Interactive -RunLevel Limited" in installer
    for prohibited in (
        "-ExecutionPolicy",
        "-EncodedCommand",
        "-Bypass",
        "Unregister-ScheduledTask",
        "schtasks",
        "Remove-Item",
    ):
        assert prohibited not in installer
    return runner, installer


def _safety_locks(*, live: bool = False, omit: str | None = None) -> str:
    values: dict[str, str] = {
        "default_recommendation": '"NEUTRAL"',
        "development_protocols": '{"v1":"no_policy_selected","v2":"no_policy_selected"}',
        "strict_oos_2025": '{"sealed":true,"executed":false}',
        "strict_oos_sealed": "true",
        "strict_oos_evaluated": "false",
        "live_trading_enabled": "true" if live else "false",
        "broker_used": "false",
        "orders_submitted": "false",
        "ml_used": "false",
        "ml_inference_used": "false",
        "recommendation_engine_used": "false",
        "risk_engine_used": "false",
        "dry_run_broker_used": "false",
        "authenticated_binance_api_used": "false",
        "network_used": "false",
    }
    if omit is not None:
        del values[omit]
    return "{" + ",".join(f'"{key}":{value}' for key, value in values.items()) + "}"


def _audit_json(mode: str) -> str:
    if mode == "malformed":
        return "not-json"
    locks = _safety_locks(
        live=mode == "wrong_lock", omit="network_used" if mode == "missing_lock" else None
    )
    passed = "false" if mode == "passed_false" else "true"
    return f'{{"passed":{passed},"not_a_certification":true,"safety_locks":{locks}}}'


def _status_json(mode: str) -> str:
    if mode == "malformed":
        return "not-json"
    locks = _safety_locks(omit="network_used" if mode == "missing_lock" else None)
    return (
        '{"not_research_or_oos_evidence":true,'
        '"status_kind":"read_only_operational_observability",'
        '"dataset":{"path":"data/raw/btcusdt_1h_development_2022_2024.csv",'
        '"symbol":"BTC/USDT","timeframe":"1h","generation_id":"generation",'
        '"checksums":{"csv_sha256":"csv","metadata_sha256":"metadata",'
        '"anomaly_sidecar_sha256":"anomaly","verification_mode":"official_online",'
        '"metadata_csv_checksum_verified":true,"anomaly_sidecar_checksum_verified":true},'
        '"requested_range":{},"stored_range":{},"candle_count":1,"freshness":{}},'
        '"tradability":{"audited_non_tradable_interruptions":[]},'
        f'"safety_locks":{locks}'
        "}"
    )


def _write_windows_fake_commands(bin_dir: Path, *, audit_mode: str, status_mode: str) -> None:
    (bin_dir / "git.cmd").write_text(
        "@echo off\n"
        'if /I "%1"=="status" exit /b 0\n'
        'if /I "%1"=="branch" (echo master & exit /b 0)\n'
        'if /I "%1"=="rev-parse" (echo 0123456789abcdef & exit /b 0)\n'
        'if /I "%1"=="check-ignore" exit /b 0\n'
        'if /I "%1"=="diff" exit /b 0\n'
        'if /I "%1"=="ls-files" exit /b 0\n'
        "exit /b 1\n",
        encoding="utf-8",
    )
    audit = _audit_json(audit_mode)
    status = _status_json(status_mode)
    (bin_dir / "uv.cmd").write_text(
        "".join(
            [
                "@echo off\n",
                'echo %*>> "%FAKE_UV_LOG%"\n',
                'if /I "%4"=="audit-safety" (\n',
                '  if /I "' + audit_mode + '"=="missing_report" (echo ' + audit + ") else (\n",
                '    > "%~6" echo ' + audit + "\n",
                "    echo " + audit + "\n",
                "  )\n",
                "  exit /b 0\n",
                ")\n",
                'if /I "%4"=="operational-status" (\n',
                '  > "%~8" echo ' + status + "\n",
                "  echo " + status + "\n",
                "  exit /b 0\n",
                ")\n",
                "exit /b 1\n",
            ]
        ),
        encoding="utf-8",
    )


def _make_junction(link: Path, target: Path) -> None:
    shell = _powershell()
    assert shell is not None
    environment = os.environ.copy()
    environment["JUNCTION_LINK"] = str(link)
    environment["JUNCTION_TARGET"] = str(target)
    command = (
        "New-Item -ItemType Junction -Path $env:JUNCTION_LINK "
        "-Target $env:JUNCTION_TARGET | Out-Null"
    )
    completed = subprocess.run(
        [
            shell,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            command,
        ],
        text=True,
        capture_output=True,
        check=False,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr


def _run_runner(
    tmp_path: Path,
    *,
    audit_mode: str = "pass",
    status_mode: str = "pass",
    reparse_ancestor: str | None = None,
) -> subprocess.CompletedProcess[str] | None:
    shell = _powershell()
    if shell is None or os.name != "nt":
        return None
    workspace = tmp_path / "workspace"
    scripts = workspace / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(RUNNER, scripts / RUNNER.name)
    if reparse_ancestor == "report":
        report_target = tmp_path / "report-target"
        reports = report_target / "operations"
        reports.mkdir(parents=True)
        _make_junction(workspace / "reports", report_target)
    else:
        reports = workspace / "reports" / "operations"
        reports.mkdir(parents=True)
    if reparse_ancestor == "dataset":
        data_target = tmp_path / "data-target"
        artifacts = data_target / "raw"
        artifacts.mkdir(parents=True)
        _make_junction(workspace / "data", data_target)
    else:
        artifacts = workspace / "data" / "raw"
        artifacts.mkdir(parents=True)
    for name in (
        "btcusdt_1h_development_2022_2024.csv",
        "btcusdt_1h_development_2022_2024.csv.metadata.json",
        "btcusdt_1h_development_2022_2024.anomalies.json",
    ):
        (artifacts / name).write_text("fixture", encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_windows_fake_commands(bin_dir, audit_mode=audit_mode, status_mode=status_mode)
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["FAKE_AUDIT_MODE"] = audit_mode
    env["FAKE_STATUS_MODE"] = status_mode
    env["FAKE_UV_LOG"] = str(tmp_path / "uv.log")
    return subprocess.run(
        [shell, "-NoLogo", "-NoProfile", "-NonInteractive", "-File", str(scripts / RUNNER.name)],
        cwd=workspace,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize("mode", ["passed_false", "malformed", "missing_lock", "wrong_lock"])
def test_runner_rejects_bad_audit_json_without_success(tmp_path: Path, mode: str) -> None:
    _source_contract()
    completed = _run_runner(tmp_path, audit_mode=mode)
    if completed is None:
        return
    assert completed.returncode != 0
    assert "pass safety-audit" not in completed.stdout


def test_runner_requires_parseable_reports_and_only_allows_offline_commands(tmp_path: Path) -> None:
    runner, _ = _source_contract()
    completed = _run_runner(tmp_path)
    if completed is None:
        assert "Assert-WorkspaceRegularFile" in runner
        return
    assert completed.returncode == 0, completed.stderr
    assert "pass safety-audit" in completed.stdout
    assert "pass operational-status" in completed.stdout
    log = (tmp_path / "uv.log").read_text(encoding="utf-8")
    assert "run --offline trading-bot audit-safety" in log
    assert "run --offline trading-bot operational-status" in log
    assert "recommend" not in log


@pytest.mark.parametrize(
    "audit_mode,status_mode", [("missing_report", "pass"), ("pass", "malformed")]
)
def test_runner_rejects_missing_or_malformed_report(
    tmp_path: Path, audit_mode: str, status_mode: str
) -> None:
    _source_contract()
    completed = _run_runner(tmp_path, audit_mode=audit_mode, status_mode=status_mode)
    if completed is None:
        return
    assert completed.returncode != 0
    assert "pass safety-audit" not in completed.stdout


def test_runner_and_installer_reject_unsafe_paths_and_scheduler_replacement() -> None:
    runner, installer = _source_contract()
    assert "Assert-RelativePath" in runner
    assert "Assert-PathHasNoReparsePoint" in runner
    assert "Operational report must be a .json file below $ReportsDirectory" in runner
    assert "Get-Item -LiteralPath $current" in runner
    assert "already exists; refusing to replace or update it" in installer
    assert installer.index("if (-not $Install.IsPresent)") < installer.index(
        "Register-ScheduledTask"
    )


@pytest.mark.parametrize("reparse_ancestor", ["dataset", "report"])
def test_runner_rejects_reparse_ancestor_before_calling_uv(
    tmp_path: Path, reparse_ancestor: str
) -> None:
    _source_contract()
    with tempfile.TemporaryDirectory(prefix="vbj-") as short_temp:
        temp_root = Path(short_temp)
        completed = _run_runner(temp_root, reparse_ancestor=reparse_ancestor)
        if completed is None:
            return
        assert completed.returncode != 0
        assert not (temp_root / "uv.log").exists()


def test_powershell_scripts_parse_without_pester() -> None:
    _source_contract()
    shell = _powershell()
    if shell is None:
        return
    for path in (RUNNER, INSTALLER):
        command = "$null = [ScriptBlock]::Create([IO.File]::ReadAllText($env:SCRIPT_TO_PARSE))"
        environment = os.environ.copy()
        environment["SCRIPT_TO_PARSE"] = str(path)
        completed = subprocess.run(
            [shell, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", command],
            text=True,
            capture_output=True,
            check=False,
            env=environment,
        )
        assert completed.returncode == 0, completed.stderr
