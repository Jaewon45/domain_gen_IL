# PowerShell script to launch all seed 3/4 jobs for E1, E3, E3b experiments
# This script runs 120 jobs total (60 per seed) across three experiments
# Output is logged to a timestamped file

# PowerShell script to run all seed 3/4 jobs for E1, E3, E3b experiments sequentially with full logging
param(
    [string]$WorkingDir = "C:\Users\320257223\PycharmProjects\domain_gen_IL\CMNIST",
    [string]$LogDir = "C:\Users\320257223\PycharmProjects\domain_gen_IL\CMNIST\job_scripts",
    [string]$PythonExe = "C:\Users\320257223\PycharmProjects\domain_gen_IL\dgil_env\Scripts\python.exe",
    [switch]$Resume
)

# Set up logging
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = Join-Path $LogDir "seed34_run_${timestamp}.log"
$progressPath = Join-Path $LogDir "seed34_resume_progress.csv"

# Change to CMNIST directory
Set-Location $WorkingDir

# Track job count and timing
$startTime = Get-Date
$jobCount = 0
$skipped = 0
$failed = 0
$completed = @{}

# List of manifest files to process in order
$manifests = @(
    "domain_count_clean_seed3.txt",
    "domain_count_clean_seed4.txt",
    "imbalance_clean_seed3.txt",
    "imbalance_clean_seed4.txt",
    "e3b_tail_support_seed3.txt",
    "e3b_tail_support_seed4.txt"
)

if ($Resume) {
    if (Test-Path $progressPath) {
        foreach ($row in @(Import-Csv -LiteralPath $progressPath)) {
            if ($row.status -eq "success") {
                $completed[[int]$row.command_index] = $true
            }
        }
        Add-Content -Path $logFile -Value "Resuming from progress file: $progressPath"
        Write-Host "Resuming from existing progress file: $progressPath" -ForegroundColor Yellow
    } else {
        Write-Host "No existing progress file found; starting fresh from the beginning." -ForegroundColor Yellow
        $Resume = $false
    }
}

if (-not $Resume) {
    if (Test-Path $progressPath) {
        Remove-Item -LiteralPath $progressPath -Force
    }
    "command_index,manifest,status,exit_code,start_time,end_time,duration_seconds,stdout_log,stderr_log,command" | Set-Content -LiteralPath $progressPath -Encoding UTF8
}

# Write initial log header
Add-Content -Path $logFile -Value "Seed 3/4 Job Batch Launch Log"
Add-Content -Path $logFile -Value "Started at: $(Get-Date)"
Add-Content -Path $logFile -Value "Working directory: $WorkingDir"
Add-Content -Path $logFile -Value "Python executable: $PythonExe"
Add-Content -Path $logFile -Value "Manifests to process: $($manifests -join ', ')"
Add-Content -Path $logFile -Value "Total expected jobs: 120"
if ($Resume) {
    Add-Content -Path $logFile -Value "Resume mode enabled: skipping completed jobs"
}
Add-Content -Path $logFile -Value "====================================`n"

Write-Host "Starting seed 3/4 batch execution..." -ForegroundColor Green
Write-Host "Log file: $logFile" -ForegroundColor Yellow

# Process each manifest file
foreach ($manifest in $manifests) {
    $manifestPath = Join-Path $LogDir $manifest
    
    if (Test-Path $manifestPath) {
        Write-Host "Processing $manifest ($(Get-Date -Format 'HH:mm:ss'))..." -ForegroundColor Cyan
        Add-Content -Path $logFile -Value "=== Processing $manifest at $(Get-Date) ==="
        
        # Read and execute each command from the manifest
        $commands = Get-Content $manifestPath
        foreach ($cmd in $commands) {
            if ($cmd.Trim().Length -gt 0) {
                $jobCount++

                if ($Resume -and $completed.ContainsKey($jobCount)) {
                    $skipped++
                    Add-Content -Path $logFile -Value "[$jobCount] SKIPPED (already completed in prior run): $cmd"
                    Write-Host "  [$jobCount] SKIP already successful" -ForegroundColor Gray
                    continue
                }

                $startJobTime = Get-Date
                $stdoutLog = Join-Path $LogDir ("seed34_job_{0:D3}.out.txt" -f $jobCount)
                $stderrLog = Join-Path $LogDir ("seed34_job_{0:D3}.err.txt" -f $jobCount)
                $status = "failed"
                $exitCode = 1
                $endJobTime = $startJobTime

                # Normalize the command to ensure a valid metric-environment selection is always present.
                $normalizedCmd = $cmd.Trim()
                if ($normalizedCmd -match ' --test_envs ') {
                    $testEnvMatch = [regex]::Match($normalizedCmd, '--test_envs\s+([^\s]+)')
                    if ($testEnvMatch.Success) {
                        $testEnvValues = $testEnvMatch.Groups[1].Value.Split(',') | ForEach-Object { $_.Trim() }
                        if (-not ($normalizedCmd -match ' --test_env_ms ')) {
                            $metricEnv = if ($testEnvValues.Count -gt 0) { $testEnvValues[-1] } else { '0.9' }
                            $normalizedCmd = "$normalizedCmd --test_env_ms $metricEnv"
                        }
                    }
                }
                if ($normalizedCmd -notmatch ' --test_env_ms ') {
                    $normalizedCmd = "$normalizedCmd --test_env_ms 0.9"
                }

                # Parse the manifest command into an explicit Python executable and argument list.
                $pythonArgs = @(
                    ($normalizedCmd -replace '^python\s+', '') -split '\s+' |
                    Where-Object { $_.Trim().Length -gt 0 }
                )

                # Log the command
                Add-Content -Path $logFile -Value "[$jobCount] $normalizedCmd"
                
                # Execute the command directly without introducing an extra cmd.exe shell layer.
                try {
                    & $PythonExe @pythonArgs 1> $stdoutLog 2> $stderrLog
                    $exitCode = $LASTEXITCODE
                    $status = if ($exitCode -eq 0) { "success" } else { "failed" }
                    $endJobTime = Get-Date
                    Add-Content -Path $logFile -Value "Job $jobCount completed with exit code $exitCode at $($endJobTime.ToString('o'))`n"
                }
                catch {
                    $err = $_.Exception.Message
                    $endJobTime = Get-Date
                    Add-Content -Path $logFile -Value ("ERROR in Job " + $jobCount + ": " + $err + "`n")
                }

                $durationSeconds = [math]::Round(($endJobTime - $startJobTime).TotalSeconds, 2)
                $row = [pscustomobject]@{
                    command_index = $jobCount
                    manifest = $manifest
                    status = $status
                    exit_code = $exitCode
                    start_time = $startJobTime.ToString("o")
                    end_time = $endJobTime.ToString("o")
                    duration_seconds = $durationSeconds
                    stdout_log = $stdoutLog
                    stderr_log = $stderrLog
                    command = $cmd
                }
                $row | ConvertTo-Csv -NoTypeInformation | Select-Object -Skip 1 | Add-Content -LiteralPath $progressPath

                if ($status -eq "failed") {
                    $failed++
                }

                if ($jobCount % 5 -eq 0) {
                    Write-Host "  Completed $jobCount / 120 jobs..." -ForegroundColor Gray
                }
            }
        }
    } else {
        Add-Content -Path $logFile -Value "ERROR: Manifest file not found: $manifestPath"
        Write-Host "ERROR: Manifest $manifest not found!" -ForegroundColor Red
    }
}

# Summary
$endTime = Get-Date
$elapsedTime = $endTime - $startTime

Add-Content -Path $logFile -Value "`n=== JOB SUMMARY ==="
Add-Content -Path $logFile -Value "Total jobs executed: $jobCount"
Add-Content -Path $logFile -Value "Jobs skipped due to previous success: $skipped"
Add-Content -Path $logFile -Value "Jobs failed: $failed"
Add-Content -Path $logFile -Value "Start time: $startTime"
Add-Content -Path $logFile -Value "End time: $endTime"
Add-Content -Path $logFile -Value "Elapsed time: $($elapsedTime.TotalSeconds) seconds"

Write-Host "Batch run finished. Total processed: $jobCount | skipped: $skipped | failed: $failed" -ForegroundColor Green
Write-Host "Elapsed time: $($elapsedTime.ToString('hh\:mm\:ss'))" -ForegroundColor Green
Write-Host "Log file: $logFile" -ForegroundColor Yellow
Write-Host "Progress file: $progressPath" -ForegroundColor Yellow

