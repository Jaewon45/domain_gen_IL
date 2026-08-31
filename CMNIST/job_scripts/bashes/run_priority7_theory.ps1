[CmdletBinding()]
param(
    [string]$OutputRoot = "results\cmnist_priority7_theory_v1",
    [switch]$Smoke
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path "$PSScriptRoot\..\..\..").Path
$pythonExe = Join-Path $repoRoot "dgil_env\Scripts\python.exe"
$script = Join-Path $repoRoot "CMNIST\priority7_theory.py"
$outputPath = if ([System.IO.Path]::IsPathRooted($OutputRoot)) { $OutputRoot } else { Join-Path $repoRoot $OutputRoot }
$logDir = Join-Path $outputPath "runner_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logPath = Join-Path $logDir "priority7_runner.log"

$arguments = @("$script", "--output_dir", "$outputPath")
if ($Smoke) { $arguments += "--smoke" }
$start = Get-Date
"[$($start.ToString('o'))] START python=$pythonExe args=$($arguments -join ' ')" | Set-Content -LiteralPath $logPath -Encoding UTF8
& $pythonExe @arguments 2>&1 | Tee-Object -FilePath $logPath -Append
$exitCode = $LASTEXITCODE
$end = Get-Date
"[$($end.ToString('o'))] END exit_code=$exitCode duration_seconds=$([math]::Round(($end-$start).TotalSeconds,2))" | Add-Content -LiteralPath $logPath
if ($exitCode -ne 0) { exit $exitCode }
