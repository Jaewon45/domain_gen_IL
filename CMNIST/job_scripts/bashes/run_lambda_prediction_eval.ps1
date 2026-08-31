[CmdletBinding()]
param(
    [string]$OutputRoot = "results\cmnist_lambda_prediction_eval_v2",
    [int]$MaxCheckpoints = 2
)

$ErrorActionPreference = "Continue"
$repoRoot = (Resolve-Path "$PSScriptRoot\..\..\..").Path
$pythonExe = Join-Path $repoRoot "dgil_env\Scripts\python.exe"
$script = Join-Path $repoRoot "CMNIST\evaluate_lambda_predictions.py"
$checkpointDir = Join-Path $repoRoot "results\cmnist_exp\ckpts"
$outputPath = if ([System.IO.Path]::IsPathRooted($OutputRoot)) { $OutputRoot } else { Join-Path $repoRoot $OutputRoot }
$logDir = Join-Path $outputPath "runner_logs"
$logPath = Join-Path $logDir "lambda_prediction_eval.log"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
if (Test-Path -LiteralPath $outputPath) {
    $existing = @(Get-ChildItem -LiteralPath $outputPath -Force)
    if ($existing.Count -gt 1 -or ($existing.Count -eq 1 -and $existing[0].Name -ne "runner_logs")) {
        throw "Refusing to overwrite non-empty output: $outputPath"
    }
}
$start = Get-Date
"[$($start.ToString('o'))] START lambda prediction evaluation" | Set-Content -LiteralPath $logPath -Encoding UTF8
$arguments = @(
    $script,
    $checkpointDir,
    "--output_dir", $outputPath,
    "--algorithms", "iro,inftask",
    "--max_checkpoints", "$MaxCheckpoints",
    "--distinct_algorithms",
    "--device", "cpu",
    "--eval_envs", "0.0,0.1,0.5,0.9,1.0",
    "--lambda_grid", "0.0:1.0:0.1"
)
Push-Location (Join-Path $repoRoot "CMNIST")
try {
    & $pythonExe @arguments 2>&1 | Tee-Object -FilePath $logPath -Append
    $exitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
$end = Get-Date
"[$($end.ToString('o'))] END exit_code=$exitCode duration_seconds=$([math]::Round(($end-$start).TotalSeconds,2))" | Add-Content -LiteralPath $logPath
exit $exitCode
