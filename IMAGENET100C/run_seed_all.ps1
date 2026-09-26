param(
    [int]$Seed = 0,
    [ValidateSet("frozen_feature_pilot", "finetune_last_stage")]
    [string]$BackboneMode = "finetune_last_stage",
    [string]$ResultsRoot = "results\imagenet100c_seed0",
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
if (-not (Test-Path -LiteralPath $python)) {
    throw "Project Python executable not found: $python"
}

if (-not [System.IO.Path]::IsPathRooted($ResultsRoot)) {
    $ResultsRoot = Join-Path $repoRoot $ResultsRoot
}
$ResultsRoot = [System.IO.Path]::GetFullPath($ResultsRoot)
$algorithms = @("erm", "groupdro", "inftask", "iro")

function Invoke-TrainingRun {
    param(
        [Parameter(Mandatory = $true)][string]$Tag,
        [Parameter(Mandatory = $true)][string]$Algorithm,
        [Parameter(Mandatory = $true)][string[]]$ExperimentArgs
    )

    $outputDirectory = Join-Path $ResultsRoot "${Tag}_${Algorithm}"
    $finalCheckpoint = Join-Path $outputDirectory "checkpoints\final.pt"

    if (Test-Path -LiteralPath $finalCheckpoint) {
        Write-Output "SKIP  $(Get-Date -Format o): completed checkpoint exists: $finalCheckpoint"
        return
    }
    if (Test-Path -LiteralPath $outputDirectory) {
        throw "Incomplete output directory already exists: $outputDirectory. Move or rename it before resuming."
    }

    $arguments = @(
        "-m", "IMAGENET100C.train",
        "--seed", "$Seed",
        "--algorithm", $Algorithm,
        "--backbone_mode", $BackboneMode,
        "--checkpoint_selection", "final",
        "--output_dir", $outputDirectory
    ) + $ExperimentArgs

    Write-Output "START $(Get-Date -Format o): $outputDirectory"
    & $python @arguments
    $trainingExitCode = $LASTEXITCODE
    if ($trainingExitCode -ne 0) {
        throw "Training failed with exit code ${trainingExitCode}: $outputDirectory"
    }
    if (-not (Test-Path -LiteralPath $finalCheckpoint)) {
        throw "Training returned success but final checkpoint is missing: $finalCheckpoint"
    }
    Write-Output "DONE  $(Get-Date -Format o): $outputDirectory"
}

foreach ($algorithm in $algorithms) {
    Invoke-TrainingRun -Tag "E0" -Algorithm $algorithm -ExperimentArgs @(
        "--experiment", "E0"
    )
}

foreach ($sourceCount in @(2, 4, 8, 12)) {
    foreach ($algorithm in $algorithms) {
        Invoke-TrainingRun -Tag "E1_${sourceCount}types" -Algorithm $algorithm -ExperimentArgs @(
            "--experiment", "E1",
            "--source_count", "$sourceCount"
        )
    }
}

foreach ($samplesPerType in @(1000, 5000, 10000)) {
    foreach ($algorithm in $algorithms) {
        Invoke-TrainingRun -Tag "E2_${samplesPerType}pertype" -Algorithm $algorithm -ExperimentArgs @(
            "--experiment", "E2",
            "--samples_per_type", "$samplesPerType"
        )
    }
}

foreach ($condition in @("balanced", "mild_imbalance", "strong_imbalance")) {
    foreach ($algorithm in $algorithms) {
        Invoke-TrainingRun -Tag "E3_${condition}" -Algorithm $algorithm -ExperimentArgs @(
            "--experiment", "E3",
            "--condition", $condition
        )
    }
}

foreach ($condition in @("balanced", "long_tail", "near_missing", "missing")) {
    foreach ($algorithm in $algorithms) {
        Invoke-TrainingRun -Tag "E3b_${condition}" -Algorithm $algorithm -ExperimentArgs @(
            "--experiment", "E3b",
            "--condition", $condition
        )
    }
}

foreach ($algorithm in $algorithms) {
    Invoke-TrainingRun -Tag "severity_support" -Algorithm $algorithm -ExperimentArgs @(
        "--experiment", "severity_support"
    )
}

Write-Output "ALL TRAINING RUNS COMPLETE: $(Get-Date -Format o)"
