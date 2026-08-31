[CmdletBinding()]
param(
    [string]$Manifest = "",
    [string]$OutputRoot = "",
    [switch]$Resume,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$scriptStarted = Get-Date
$repoRoot = (Resolve-Path "$PSScriptRoot\..\..").Path
$cmnistRoot = Join-Path $repoRoot "CMNIST"
$pythonExe = Join-Path $repoRoot "dgil_env\Scripts\python.exe"
if ([string]::IsNullOrWhiteSpace($Manifest)) {
    $Manifest = Join-Path $PSScriptRoot "groupdro_controlled_seed012.txt"
}
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $repoRoot "results\cmnist_groupdro_control_v1"
}
$manifestPath = (Resolve-Path $Manifest).Path
$outputPath = if ([System.IO.Path]::IsPathRooted($OutputRoot)) {
    [System.IO.Path]::GetFullPath($OutputRoot)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $repoRoot $OutputRoot))
}
$runnerLogRoot = Join-Path $outputPath "runner_logs"
$stdoutRoot = Join-Path $runnerLogRoot "stdout"
$stderrRoot = Join-Path $runnerLogRoot "stderr"
$progressPath = Join-Path $runnerLogRoot "progress.csv"
$summaryPath = Join-Path $runnerLogRoot "summary.txt"

foreach ($directory in @($outputPath, $runnerLogRoot, $stdoutRoot, $stderrRoot)) {
    New-Item -ItemType Directory -Force -Path $directory | Out-Null
}

if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Python executable not found: $pythonExe"
}

$commands = @(Get-Content -LiteralPath $manifestPath | Where-Object {
    $_.Trim().Length -gt 0 -and -not $_.Trim().StartsWith('#')
})
if ($commands.Count -eq 0) {
    throw "No executable commands found in $manifestPath"
}

$completed = @{}
if ($Resume -and (Test-Path -LiteralPath $progressPath)) {
    foreach ($row in @(Import-Csv -LiteralPath $progressPath)) {
        if ($row.status -eq "success") {
            $completed[[int]$row.command_index] = $true
        }
    }
}

if (-not (Test-Path -LiteralPath $progressPath) -or -not $Resume) {
    "command_index,status,exit_code,start_time,end_time,duration_seconds,stdout_log,stderr_log" | Set-Content -LiteralPath $progressPath -Encoding UTF8
}

$failed = 0
$skipped = 0
Write-Host "Manifest: $manifestPath"
Write-Host "Commands: $($commands.Count)"
Write-Host "Output root: $outputPath"
Write-Host "Progress CSV: $progressPath"

for ($index = 0; $index -lt $commands.Count; $index++) {
    $commandIndex = $index + 1
    $command = $commands[$index].Trim()
    $stdoutLog = Join-Path $stdoutRoot ("command_{0:D3}.out.txt" -f $commandIndex)
    $stderrLog = Join-Path $stderrRoot ("command_{0:D3}.err.txt" -f $commandIndex)

    if ($completed.ContainsKey($commandIndex)) {
        $skipped++
        Write-Host ("[{0}/{1}] SKIP success already recorded" -f $commandIndex, $commands.Count)
        continue
    }

    $start = Get-Date
    Write-Host ("[{0}/{1}] START {2}" -f $commandIndex, $commands.Count, $start.ToString("s"))
    Write-Host "  $command"
    "[$($start.ToString('s'))] START command_index=$commandIndex" | Add-Content -LiteralPath $summaryPath
    $command | Add-Content -LiteralPath $summaryPath

    $status = "dry_run"
    $exitCode = 0
    if (-not $DryRun) {
        $fullCommand = $command -replace '^python\s+', ('"' + $pythonExe + '" ')
        Push-Location $cmnistRoot
        try {
            & cmd.exe /d /c $fullCommand 1> $stdoutLog 2> $stderrLog
            $exitCode = $LASTEXITCODE
            $status = if ($exitCode -eq 0) { "success" } else { "failed" }
        }
        finally {
            Pop-Location
        }
    }

    $end = Get-Date
    $duration = ($end - $start).TotalSeconds
    if ($status -eq "failed") {
        $failed++
    }
    $row = [pscustomobject]@{
        command_index = $commandIndex
        status = $status
        exit_code = $exitCode
        start_time = $start.ToString("o")
        end_time = $end.ToString("o")
        duration_seconds = [math]::Round($duration, 2)
        stdout_log = $stdoutLog
        stderr_log = $stderrLog
    }
    $row | ConvertTo-Csv -NoTypeInformation | Select-Object -Skip 1 | Add-Content -LiteralPath $progressPath
    "[$($end.ToString('s'))] END command_index=$commandIndex status=$status exit_code=$exitCode duration_seconds=$([math]::Round($duration, 2))" | Add-Content -LiteralPath $summaryPath
    Write-Host ("[{0}/{1}] {2} exit_code={3} duration={4:N1}s" -f $commandIndex, $commands.Count, $status.ToUpper(), $exitCode, $duration)
}

$finished = Get-Date
"Started:  $($scriptStarted.ToString('o'))" | Add-Content -LiteralPath $summaryPath
"Finished: $($finished.ToString('o'))" | Add-Content -LiteralPath $summaryPath
"Commands: $($commands.Count); skipped_success: $skipped; failed: $failed" | Add-Content -LiteralPath $summaryPath
Write-Host "Finished. skipped_success=$skipped failed=$failed"
if ($failed -gt 0) {
    exit 1
}
