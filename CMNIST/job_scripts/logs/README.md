# Job Submission Logs

This directory stores stdout/stderr logs from experiment job submissions.

## Contents

- `domain_stress_main_seed*.log` — Historical job logs from domain-stress experiments
- `seed*.log` — Per-seed submission logs (generated during experiment runs)
- `out.txt`, `err.txt` — HPC job launcher output/error logs

## Usage

When running experiments via job manifests (e.g., `domain_count_clean.txt`), logs are captured in this directory:

```bash
cd ..  # CMNIST/
source job_scripts/domain_count_clean.txt > job_scripts/logs/domain_count_submission.log 2>&1

# Or with submitit (HPC):
python job_scripts.submit_jobs -c job_scripts/domain_count_clean.txt --time 30
# Logs saved to job_scripts/logs/
```

## Debugging

If jobs fail, check the corresponding log file:

```bash
tail -f job_scripts/logs/domain_count_submission.log
grep ERROR job_scripts/logs/*.log
```

## Cleanup

Old logs can be safely deleted after results have been collected and verified.

```bash
rm job_scripts/logs/*.log
```
