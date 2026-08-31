[CmdletBinding()]
param(
    [string]$ExperimentRoot = "results\imbalance_clean_v1",
    [int]$ExpectedJobs = 75
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path "$PSScriptRoot\..\..\..").Path
$pythonExe = Join-Path $repoRoot "dgil_env\Scripts\python.exe"
$root = if ([System.IO.Path]::IsPathRooted($ExperimentRoot)) { $ExperimentRoot } else { Join-Path $repoRoot $ExperimentRoot }
$progress = Join-Path $root "runner_logs\progress.csv"
$resultsDir = Join-Path $root "results"
$analysisDir = Join-Path $root "analysis"
$exportDir = Join-Path $root "export"

if (-not (Test-Path -LiteralPath $progress)) {
    throw "Missing runner progress file: $progress"
}
$rows = @(Import-Csv -LiteralPath $progress)
$success = @($rows | Where-Object { $_.status -eq "success" })
$failed = @($rows | Where-Object { $_.status -ne "success" })
if ($rows.Count -ne $ExpectedJobs -or $success.Count -ne $ExpectedJobs -or $failed.Count -gt 0) {
    throw "Refusing to analyze incomplete E3 run: completed=$($rows.Count)/$ExpectedJobs, success=$($success.Count), non_success=$($failed.Count)"
}

New-Item -ItemType Directory -Force -Path $analysisDir, $exportDir | Out-Null
Push-Location $repoRoot
try {
    & $pythonExe CMNIST\export_results_csv.py $resultsDir --output_dir $exportDir --prefix imbalance_clean
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $pythonExe CMNIST\plot_domain_stress.py $resultsDir --output_dir $analysisDir
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $summary = [ordered]@{
        generated_at = (Get-Date).ToString("o")
        expected_jobs = $ExpectedJobs
        successful_jobs = $success.Count
        failed_jobs = $failed.Count
        results_dir = $resultsDir
        export_dir = $exportDir
        analysis_dir = $analysisDir
    }
    $summary | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $analysisDir "run_summary.json") -Encoding UTF8
}
finally {
    Pop-Location
}
Write-Host "E3 corrected analysis complete: $analysisDir"
