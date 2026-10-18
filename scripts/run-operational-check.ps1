[CmdletBinding()]
param(
    [switch]$BootstrapSmoke
)

$ErrorActionPreference = "Stop"

$DatasetPath = "data/raw/btcusdt_1h_development_2022_2024.csv"
$MetadataPath = "$DatasetPath.metadata.json"
$AnomalyPath = "data/raw/btcusdt_1h_development_2022_2024.anomalies.json"
$ReportsDirectory = "reports/operations"
$FalseSafetyLocks = @(
    "live_trading_enabled",
    "broker_used",
    "orders_submitted",
    "ml_used",
    "ml_inference_used",
    "recommendation_engine_used",
    "risk_engine_used",
    "dry_run_broker_used",
    "authenticated_binance_api_used",
    "network_used"
)
$SecretLikePattern = "(?i)(sk-[a-z0-9]{16,}|api[_-]?(key|secret)\s*[:=]|binance[^\r\n]{0,40}(key|secret)|password\s*[:=]|token\s*[:=])"

function Get-RequiredProperty {
    param(
        [Parameter(Mandatory)]
        [object]$Object,
        [Parameter(Mandatory)]
        [string]$Name,
        [Parameter(Mandatory)]
        [string]$Context
    )

    if ($null -eq $Object) {
        throw "$Context is missing"
    }
    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property) {
        throw "$Context is missing required field '$Name'"
    }
    return $property.Value
}

function Assert-ExactBoolean {
    param(
        [Parameter(Mandatory)]
        [object]$Value,
        [Parameter(Mandatory)]
        [bool]$Expected,
        [Parameter(Mandatory)]
        [string]$Context
    )

    if ($Value -isnot [bool] -or $Value -ne $Expected) {
        throw "$Context must be boolean $Expected"
    }
}

function Assert-RelativePath {
    param(
        [Parameter(Mandatory)]
        [string]$RelativePath,
        [Parameter(Mandatory)]
        [string]$Context
    )

    if ([IO.Path]::IsPathRooted($RelativePath)) {
        throw "$Context must be relative"
    }
    $parts = $RelativePath -split "[\\/]"
    if ($parts.Count -eq 0 -or $parts -contains ".." -or $parts -contains "") {
        throw "$Context must not contain traversal"
    }
}

function Assert-PathHasNoReparsePoint {
    param(
        [Parameter(Mandatory)]
        [string]$WorkspaceRoot,
        [Parameter(Mandatory)]
        [string]$RelativePath,
        [Parameter(Mandatory)]
        [string]$Context
    )

    Assert-RelativePath -RelativePath $RelativePath -Context $Context
    $root = [IO.Path]::GetFullPath($WorkspaceRoot)
    $rootItem = Get-Item -LiteralPath $root -Force -ErrorAction Stop
    if (($rootItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Repository root must not be a reparse point"
    }

    $current = $root
    foreach ($part in ($RelativePath -split "[\\/]")) {
        $current = Join-Path $current $part
        $item = Get-Item -LiteralPath $current -Force -ErrorAction Stop
        if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "$Context must not contain a symlink, junction, or reparse point: $RelativePath"
        }
    }

    $fullPath = [IO.Path]::GetFullPath($current)
    $rootWithSeparator = "$root$([IO.Path]::DirectorySeparatorChar)"
    if (-not $fullPath.StartsWith($rootWithSeparator, [StringComparison]::OrdinalIgnoreCase)) {
        throw "$Context escapes the repository: $RelativePath"
    }
    return $fullPath
}

function Assert-WorkspaceRegularFile {
    param(
        [Parameter(Mandatory)]
        [string]$WorkspaceRoot,
        [Parameter(Mandatory)]
        [string]$RelativePath,
        [Parameter(Mandatory)]
        [string]$Context
    )

    $fullPath = Assert-PathHasNoReparsePoint `
        -WorkspaceRoot $WorkspaceRoot -RelativePath $RelativePath -Context $Context
    $item = Get-Item -LiteralPath $fullPath -Force -ErrorAction Stop
    if ($item.PSIsContainer -or (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0)) {
        throw "$Context must be a regular non-symlink file: $RelativePath"
    }
    return $fullPath
}

function Assert-WorkspaceDirectory {
    param(
        [Parameter(Mandatory)]
        [string]$WorkspaceRoot,
        [Parameter(Mandatory)]
        [string]$RelativePath,
        [Parameter(Mandatory)]
        [string]$Context
    )

    $fullPath = Assert-PathHasNoReparsePoint `
        -WorkspaceRoot $WorkspaceRoot -RelativePath $RelativePath -Context $Context
    $item = Get-Item -LiteralPath $fullPath -Force -ErrorAction Stop
    if (-not $item.PSIsContainer -or (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0)) {
        throw "$Context must be a normal non-symlink directory: $RelativePath"
    }
    return $fullPath
}

function Assert-ReportPath {
    param(
        [Parameter(Mandatory)]
        [string]$WorkspaceRoot,
        [Parameter(Mandatory)]
        [string]$RelativePath
    )

    Assert-RelativePath -RelativePath $RelativePath -Context "Operational report path"
    if (-not $RelativePath.StartsWith("$ReportsDirectory/", [StringComparison]::OrdinalIgnoreCase) `
        -or -not $RelativePath.EndsWith(".json", [StringComparison]::OrdinalIgnoreCase)) {
        throw "Operational report must be a .json file below $ReportsDirectory"
    }
    $reportsRoot = Assert-WorkspaceDirectory `
        -WorkspaceRoot $WorkspaceRoot -RelativePath "reports" -Context "Reports directory"
    $operationsPath = [IO.Path]::GetFullPath((Join-Path $WorkspaceRoot $ReportsDirectory))
    if (Test-Path -LiteralPath $operationsPath) {
        $null = Assert-WorkspaceDirectory `
            -WorkspaceRoot $WorkspaceRoot -RelativePath $ReportsDirectory -Context "Operational report directory"
    }
    $fullPath = [IO.Path]::GetFullPath((Join-Path $WorkspaceRoot $RelativePath))
    $reportsWithSeparator = "$reportsRoot$([IO.Path]::DirectorySeparatorChar)"
    if (-not $fullPath.StartsWith($reportsWithSeparator, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Operational report escapes the reports directory"
    }
    if (Test-Path -LiteralPath $fullPath) {
        throw "Refusing to overwrite existing report: $RelativePath"
    }
    return $fullPath
}

function ConvertFrom-RequiredJson {
    param(
        [Parameter(Mandatory)]
        [string]$Json,
        [Parameter(Mandatory)]
        [string]$Context
    )

    try {
        $parsed = $Json | ConvertFrom-Json -ErrorAction Stop
    }
    catch {
        throw "$Context is not valid JSON"
    }
    if ($null -eq $parsed -or $parsed -isnot [pscustomobject]) {
        throw "$Context must be a JSON object"
    }
    return $parsed
}

function Assert-NoSecretLikeValue {
    param(
        [Parameter(Mandatory)]
        [object]$Value,
        [Parameter(Mandatory)]
        [string]$Context
    )

    if ($Value -is [pscustomobject]) {
        foreach ($property in $Value.PSObject.Properties) {
            Assert-NoSecretLikeValue -Value $property.Value -Context $Context
        }
        return
    }
    if ($Value -is [System.Collections.IEnumerable] -and $Value -isnot [string]) {
        foreach ($item in $Value) {
            Assert-NoSecretLikeValue -Value $item -Context $Context
        }
        return
    }
    if ($Value -is [string] -and $Value -match $SecretLikePattern) {
        throw "$Context contains a prohibited secret-like value"
    }
}

function Assert-SafetyLocks {
    param(
        [Parameter(Mandatory)]
        [object]$Locks,
        [Parameter(Mandatory)]
        [string]$Context
    )

    if ($Locks -isnot [pscustomobject]) {
        throw "$Context safety_locks must be an object"
    }
    if ((Get-RequiredProperty -Object $Locks -Name "default_recommendation" -Context $Context) -cne "NEUTRAL") {
        throw "$Context default recommendation is not NEUTRAL"
    }
    $protocols = Get-RequiredProperty -Object $Locks -Name "development_protocols" -Context $Context
    if ($protocols -isnot [pscustomobject] `
        -or (Get-RequiredProperty -Object $protocols -Name "v1" -Context $Context) -cne "no_policy_selected" `
        -or (Get-RequiredProperty -Object $protocols -Name "v2" -Context $Context) -cne "no_policy_selected") {
        throw "$Context development policy lock is invalid"
    }
    Assert-ExactBoolean -Value (Get-RequiredProperty -Object $Locks -Name "strict_oos_sealed" -Context $Context) `
        -Expected $true -Context "$Context strict OOS sealed lock"
    Assert-ExactBoolean -Value (Get-RequiredProperty -Object $Locks -Name "strict_oos_evaluated" -Context $Context) `
        -Expected $false -Context "$Context strict OOS evaluated lock"
    $strictOos = Get-RequiredProperty -Object $Locks -Name "strict_oos_2025" -Context $Context
    if ($strictOos -isnot [pscustomobject]) {
        throw "$Context strict_oos_2025 must be an object"
    }
    Assert-ExactBoolean -Value (Get-RequiredProperty -Object $strictOos -Name "sealed" -Context $Context) `
        -Expected $true -Context "$Context strict OOS 2025 sealed lock"
    Assert-ExactBoolean -Value (Get-RequiredProperty -Object $strictOos -Name "executed" -Context $Context) `
        -Expected $false -Context "$Context strict OOS 2025 executed lock"
    foreach ($name in $FalseSafetyLocks) {
        Assert-ExactBoolean -Value (Get-RequiredProperty -Object $Locks -Name $name -Context $Context) `
            -Expected $false -Context "$Context $name lock"
    }
}

function Assert-AuditPayload {
    param([Parameter(Mandatory)][object]$Payload)

    Assert-ExactBoolean -Value (Get-RequiredProperty -Object $Payload -Name "passed" -Context "Safety audit") `
        -Expected $true -Context "Safety audit passed"
    Assert-ExactBoolean -Value (Get-RequiredProperty -Object $Payload -Name "not_a_certification" -Context "Safety audit") `
        -Expected $true -Context "Safety audit not_a_certification"
    Assert-SafetyLocks -Locks (Get-RequiredProperty -Object $Payload -Name "safety_locks" -Context "Safety audit") `
        -Context "Safety audit"
    Assert-NoSecretLikeValue -Value $Payload -Context "Safety audit"
}

function Assert-StatusPayload {
    param([Parameter(Mandatory)][object]$Payload)

    Assert-ExactBoolean -Value (Get-RequiredProperty -Object $Payload -Name "not_research_or_oos_evidence" -Context "Operational status") `
        -Expected $true -Context "Operational status research provenance"
    if ((Get-RequiredProperty -Object $Payload -Name "status_kind" -Context "Operational status") -cne "read_only_operational_observability") {
        throw "Operational status kind is invalid"
    }
    $dataset = Get-RequiredProperty -Object $Payload -Name "dataset" -Context "Operational status"
    if ($dataset -isnot [pscustomobject]) {
        throw "Operational status dataset must be an object"
    }
    foreach ($name in @("path", "symbol", "timeframe", "generation_id", "checksums", "requested_range", "stored_range", "candle_count", "freshness")) {
        $null = Get-RequiredProperty -Object $dataset -Name $name -Context "Operational status dataset"
    }
    if ((Get-RequiredProperty -Object $dataset -Name "path" -Context "Operational status dataset") -cne $DatasetPath `
        -or (Get-RequiredProperty -Object $dataset -Name "symbol" -Context "Operational status dataset") -cne "BTC/USDT" `
        -or (Get-RequiredProperty -Object $dataset -Name "timeframe" -Context "Operational status dataset") -cne "1h") {
        throw "Operational status dataset provenance is invalid"
    }
    $checksums = Get-RequiredProperty -Object $dataset -Name "checksums" -Context "Operational status dataset"
    if ($checksums -isnot [pscustomobject]) {
        throw "Operational status checksums must be an object"
    }
    foreach ($name in @("csv_sha256", "metadata_sha256", "anomaly_sidecar_sha256", "verification_mode")) {
        $value = Get-RequiredProperty -Object $checksums -Name $name -Context "Operational status checksums"
        if ($value -isnot [string] -or [string]::IsNullOrWhiteSpace($value)) {
            throw "Operational status checksum provenance is invalid"
        }
    }
    Assert-ExactBoolean -Value (Get-RequiredProperty -Object $checksums -Name "metadata_csv_checksum_verified" -Context "Operational status checksums") `
        -Expected $true -Context "Operational status metadata checksum"
    Assert-ExactBoolean -Value (Get-RequiredProperty -Object $checksums -Name "anomaly_sidecar_checksum_verified" -Context "Operational status checksums") `
        -Expected $true -Context "Operational status anomaly checksum"
    $tradability = Get-RequiredProperty -Object $Payload -Name "tradability" -Context "Operational status"
    if ($tradability -isnot [pscustomobject]) {
        throw "Operational status tradability must be an object"
    }
    $null = Get-RequiredProperty -Object $tradability -Name "audited_non_tradable_interruptions" -Context "Operational status tradability"
    Assert-SafetyLocks -Locks (Get-RequiredProperty -Object $Payload -Name "safety_locks" -Context "Operational status") `
        -Context "Operational status"
    Assert-NoSecretLikeValue -Value $Payload -Context "Operational status"
}

function Invoke-ReadOnlyCommand {
    param(
        [Parameter(Mandatory)]
        [ValidateSet("audit-safety", "operational-status")]
        [string]$Subcommand,
        [Parameter(Mandatory)]
        [string[]]$Arguments
    )

    $uvArguments = @("run", "--offline", "trading-bot", $Subcommand) + $Arguments
    $output = @(& uv @uvArguments 2>&1)
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "$Subcommand failed with exit code $exitCode"
    }
    return (($output | ForEach-Object { $_.ToString() }) -join [Environment]::NewLine)
}

function Get-VerifiedReport {
    param(
        [Parameter(Mandatory)][string]$WorkspaceRoot,
        [Parameter(Mandatory)][string]$RelativePath,
        [Parameter(Mandatory)][string]$Context
    )

    $fullPath = Assert-WorkspaceRegularFile -WorkspaceRoot $WorkspaceRoot -RelativePath $RelativePath -Context $Context
    try {
        $contents = [IO.File]::ReadAllText($fullPath, [Text.Encoding]::UTF8)
    }
    catch {
        throw "$Context could not be read"
    }
    $payload = ConvertFrom-RequiredJson -Json $contents -Context $Context
    Assert-NoSecretLikeValue -Value $payload -Context $Context
    return $payload
}

function Assert-CleanRepository {
    param([Parameter(Mandatory)][bool]$AllowBootstrapSmoke)

    $status = @(& git status --porcelain 2>&1)
    if ($LASTEXITCODE -ne 0) {
        throw "Could not inspect repository status; refusing operational check"
    }
    if ($status.Count -eq 0) {
        return
    }
    if (-not $AllowBootstrapSmoke) {
        throw "Repository worktree is not clean; refusing operational check"
    }
    $changed = @(
        & git diff --name-only 2>&1
        & git diff --cached --name-only 2>&1
        & git ls-files --others --exclude-standard 2>&1
    ) | Where-Object { $_ }
    if ($LASTEXITCODE -ne 0) {
        throw "Could not inspect bootstrap changes; refusing operational check"
    }
    $allowed = @("docs/operations.md", "scripts/run-operational-check.ps1")
    if (@($changed | Sort-Object -Unique | Where-Object { $_ -notin $allowed }).Count -ne 0) {
        throw "Repository has changes outside the bootstrap runbook; refusing operational check"
    }
}

$repositoryRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
Set-Location -LiteralPath $repositoryRoot

Assert-CleanRepository -AllowBootstrapSmoke $BootstrapSmoke.IsPresent

$branch = (& git branch --show-current 2>&1).Trim()
$head = (& git rev-parse HEAD 2>&1).Trim()
$originMaster = (& git rev-parse origin/master 2>&1).Trim()
if ($LASTEXITCODE -ne 0 -or $branch -ne "master" -or $head -ne $originMaster) {
    throw "Repository is not a clean master matching origin/master; refusing operational check"
}

foreach ($artifact in @($DatasetPath, $MetadataPath, $AnomalyPath)) {
    $null = Assert-WorkspaceRegularFile -WorkspaceRoot $repositoryRoot -RelativePath $artifact -Context "Development artifact"
}

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$auditReport = "$ReportsDirectory/safety-audit-$timestamp.json"
$statusReport = "$ReportsDirectory/development-status-$timestamp.json"
foreach ($report in @($auditReport, $statusReport)) {
    $null = Assert-ReportPath -WorkspaceRoot $repositoryRoot -RelativePath $report
}

$auditJson = Invoke-ReadOnlyCommand -Subcommand "audit-safety" -Arguments @("--output", $auditReport)
$auditPayload = ConvertFrom-RequiredJson -Json $auditJson -Context "Safety audit command output"
Assert-AuditPayload -Payload $auditPayload
$auditReportPayload = Get-VerifiedReport -WorkspaceRoot $repositoryRoot -RelativePath $auditReport -Context "Safety audit report"
Assert-AuditPayload -Payload $auditReportPayload

$statusJson = Invoke-ReadOnlyCommand -Subcommand "operational-status" -Arguments @("--input", $DatasetPath, "--output", $statusReport)
$statusPayload = ConvertFrom-RequiredJson -Json $statusJson -Context "Operational status command output"
Assert-StatusPayload -Payload $statusPayload
$statusReportPayload = Get-VerifiedReport -WorkspaceRoot $repositoryRoot -RelativePath $statusReport -Context "Operational status report"
Assert-StatusPayload -Payload $statusReportPayload

foreach ($report in @($auditReport, $statusReport)) {
    $null = & git check-ignore -q -- $report 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Generated report is not Git-ignored: $report"
    }
}

Write-Output "$timestamp pass safety-audit $auditReport"
Write-Output "$timestamp pass operational-status $statusReport"
