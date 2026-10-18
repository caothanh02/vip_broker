[CmdletBinding(DefaultParameterSetName = "Inspect")]
param(
    [Parameter(ParameterSetName = "Install")]
    [switch]$Install,
    [Parameter(ParameterSetName = "Inspect")]
    [switch]$Inspect
)

$ErrorActionPreference = "Stop"
$TaskName = "VipBrokerNeutralOperationalCheck"
$TaskTime = "09:00"

function Get-RepositoryRoot {
    return [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
}

function Assert-RegularRunner {
    param([Parameter(Mandatory)][string]$RepositoryRoot)

    $runner = Join-Path $RepositoryRoot "scripts/run-operational-check.ps1"
    $rootItem = Get-Item -LiteralPath $RepositoryRoot -Force -ErrorAction Stop
    if (($rootItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Repository root must not be a reparse point"
    }
    foreach ($path in @((Join-Path $RepositoryRoot "scripts"), $runner)) {
        $item = Get-Item -LiteralPath $path -Force -ErrorAction Stop
        if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Operational runner path must not contain a symlink, junction, or reparse point"
        }
    }
    $runnerItem = Get-Item -LiteralPath $runner -Force -ErrorAction Stop
    if ($runnerItem.PSIsContainer) {
        throw "Operational runner must be a regular file"
    }
    return $runner
}

function Assert-CleanSyncedMaster {
    param([Parameter(Mandatory)][string]$RepositoryRoot)

    Push-Location -LiteralPath $RepositoryRoot
    try {
        $status = @(& git status --porcelain 2>&1)
        $branch = (& git branch --show-current 2>&1).Trim()
        $head = (& git rev-parse HEAD 2>&1).Trim()
        $originMaster = (& git rev-parse origin/master 2>&1).Trim()
    }
    finally {
        Pop-Location
    }
    if ($LASTEXITCODE -ne 0 -or $status.Count -ne 0 -or $branch -ne "master" -or $head -ne $originMaster) {
        throw "Scheduled task installation requires a clean master matching origin/master"
    }
}

function Get-ExistingTask {
    try {
        return Get-ScheduledTask -TaskName $TaskName -ErrorAction Stop
    }
    catch {
        if ($_.Exception.Message -match "cannot find|not found") {
            return $null
        }
        throw "Could not inspect scheduled task '$TaskName'"
    }
}

function Get-TaskInspection {
    param([object]$Task)

    $result = [ordered]@{
        schema_version = "1.0"
        task_name = $TaskName
        schedule = "daily 09:00 local"
        exists = ($null -ne $Task)
        install_required = ($null -eq $Task)
        action_contract = "PowerShell -NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -File scripts/run-operational-check.ps1"
        principal_contract = "limited interactive user token; no password, service account, or highest privileges"
        mutates_scheduler = $false
    }
    if ($null -ne $Task) {
        $result["actions"] = @($Task.Actions | ForEach-Object { [ordered]@{ execute = $_.Execute; arguments = $_.Arguments } })
        $result["triggers"] = @($Task.Triggers | ForEach-Object { [ordered]@{ start_boundary = $_.StartBoundary; type = $_.CimClass.CimClassName } })
        $result["principal"] = [ordered]@{
            user_id = $Task.Principal.UserId
            logon_type = $Task.Principal.LogonType.ToString()
            run_level = $Task.Principal.RunLevel.ToString()
        }
    }
    return $result
}

function Assert-TaskInstallationPrerequisites {
    param([Parameter(Mandatory)][string]$RepositoryRoot)

    $null = Assert-RegularRunner -RepositoryRoot $RepositoryRoot
    Assert-CleanSyncedMaster -RepositoryRoot $RepositoryRoot
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    if ($null -eq $identity -or [string]::IsNullOrWhiteSpace($identity.Name)) {
        throw "Cannot determine an interactive user token; refusing scheduled task installation"
    }
    return $identity.Name
}

$repositoryRoot = Get-RepositoryRoot
$existingTask = Get-ExistingTask

if (-not $Install.IsPresent) {
    Get-TaskInspection -Task $existingTask | ConvertTo-Json -Depth 6 -Compress
    exit 0
}

if ($null -ne $existingTask) {
    throw "Scheduled task '$TaskName' already exists; refusing to replace or update it"
}

$userId = Assert-TaskInstallationPrerequisites -RepositoryRoot $repositoryRoot
$runner = Assert-RegularRunner -RepositoryRoot $repositoryRoot
$powerShell = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
if (-not (Test-Path -LiteralPath $powerShell -PathType Leaf)) {
    throw "Windows PowerShell executable is unavailable; refusing scheduled task installation"
}

$actionArguments = "-NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -File `"$runner`""
if ($actionArguments -match "ExecutionPolicy|EncodedCommand|Bypass") {
    throw "Unsafe PowerShell scheduler action was constructed"
}
$action = New-ScheduledTaskAction -Execute $powerShell -Argument $actionArguments
$trigger = New-ScheduledTaskTrigger -Daily -At 09:00AM
$principal = New-ScheduledTaskPrincipal -UserId $userId -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -Hidden -StartWhenAvailable -MultipleInstances IgnoreNew

try {
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $action `
        -Trigger $trigger `
        -Principal $principal `
        -Settings $settings `
        -Description "Local read-only neutral operational checks for frozen development data" `
        -ErrorAction Stop | Out-Null
}
catch {
    throw "Scheduled task registration failed without elevated or service-account fallback"
}

Get-TaskInspection -Task (Get-ExistingTask) | ConvertTo-Json -Depth 6 -Compress
