param(
    [string]$ResultsRoot = "results\imagenet100c_seed0",
    [string]$SubmissionRoot = "results_submit_img100",
    [switch]$Offline
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot
if ($Offline) {
    $env:HF_DATASETS_OFFLINE = "1"
    $env:HF_HUB_OFFLINE = "1"
}
$python = Join-Path $repoRoot "dgil_env\Scripts\python.exe"
if (-not [IO.Path]::IsPathRooted($ResultsRoot)) { $ResultsRoot = Join-Path $repoRoot $ResultsRoot }
if (-not [IO.Path]::IsPathRooted($SubmissionRoot)) { $SubmissionRoot = Join-Path $repoRoot $SubmissionRoot }
$ResultsRoot = [IO.Path]::GetFullPath($ResultsRoot)
$SubmissionRoot = [IO.Path]::GetFullPath($SubmissionRoot)
$anchors = "gaussian_noise,defocus_blur,snow,contrast"
$runs = Get-ChildItem -LiteralPath $ResultsRoot -Directory | Where-Object {
    $_.Name -like "E3_*" -or $_.Name -like "E3b_*"
} | Sort-Object Name
if (@($runs).Count -ne 28) { throw "Expected 28 E3/E3b runs, found $(@($runs).Count)" }

foreach ($run in $runs) {
    $checkpoint = Join-Path $run.FullName "checkpoints\final.pt"
    $output = Join-Path $run.FullName "evaluation_pilot_anchor100"
    $result = Join-Path $output "evaluation.jsonl"
    if (Test-Path -LiteralPath $result) {
        Write-Output "SKIP  $(Get-Date -Format o): $($run.Name)"
        continue
    }
    if (Test-Path -LiteralPath $output) {
        throw "Incomplete pilot output exists: $output"
    }
    Write-Output "START $(Get-Date -Format o): $($run.Name)"
    & $python -m IMAGENET100C.evaluate $checkpoint `
        --output_dir $output `
        --max_eval_images 100 `
        --corruption_types $anchors `
        --batch_size 64 `
        --workers 0
    if ($LASTEXITCODE -ne 0) { throw "Pilot evaluation failed: $($run.Name)" }
    Write-Output "DONE  $(Get-Date -Format o): $($run.Name)"
}

& $python -m IMAGENET100C.plot_pilot_submission `
    --runs_root $ResultsRoot `
    --output_dir $SubmissionRoot
if ($LASTEXITCODE -ne 0) { throw "Pilot plotting failed" }
Write-Output "ALL PILOT EVALUATIONS AND FIGURES COMPLETE: $(Get-Date -Format o)"
