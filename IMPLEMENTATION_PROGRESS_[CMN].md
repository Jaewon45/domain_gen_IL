# [CMN] Implementation Progress

This document records the implementation work completed for Experiment 1, the CMNIST stress test, and the validation results collected during setup.

## Scope Completed

The following parts of Experiment 1 are now implemented in the repository:

- Phase 1: varying the number of training domains.
- Phase 2: controlling the number of samples per training domain.
- Phase 3: generating imbalance schedules through per-domain sample sizes.

The following analysis and workflow support tasks are also now implemented:

- A reduced first-pass command file for small CMNIST stress-test runs.
- Phase-aware result aggregation for stress-test outputs.
- A dedicated λ-grid evaluation entry point for saved checkpoints.
- A plotting entry point for the first CMNIST stress-test figures.

## Code Changes

### 1. CMNIST Dataset Subsampling Support

File changed:

- `CMNIST/datasets.py`

What was added:

- A helper to subsample a `TensorDataset` to a requested size.
- Support for `train_env_sizes`.
- Support for `train_env_size_mode` with `random` and `first` modes.

What this enables:

- Balanced low-data training setups.
- Explicit domain imbalance schedules.
- Reproducible per-domain size control.

### 2. CMNIST Training CLI Support

File changed:

- `CMNIST/train_sandbox.py`

What was added:

- `--train_env_sizes`
- `--train_env_size_mode`

What was fixed:

- The script previously used `parse_args(args=[], ...)`, which ignored all CLI arguments.
- This was corrected so command-line options now actually control the run.

What this enables:

- Running stress-test configurations directly from the terminal.
- Smoke-testing custom domain-count and imbalance settings.

### 3. Stress-Test Job Generation

File changed:

- `CMNIST/job_scripts/gen_exps.py`

What was changed:

- The script now generates commands using `train_sandbox.py`.
- A dedicated `domain_stress` experiment mode was added.

Current `domain_stress` coverage:

- Phase 1 domain-count sweep:
  - 2 domains
  - 4 domains
  - 8 domains
- Phase 2 balanced sample-size sweep:
  - 2000 per domain
  - 4000 per domain
  - 8000 per domain
- Phase 3 imbalance sweep on 4 domains:
  - balanced: `2000,2000,2000,2000`
  - mild imbalance: `2000,2000,2000,4000`
  - strong imbalance: `2000,2000,2000,10000`

Algorithms currently included in the generated sweep:

- `erm`
- `irm`
- `groupdro`
- `iro`
- `inftask`

### 4. Reduced Stress-Test Command File

File added:

- `CMNIST/job_scripts/domain_stress_small.txt`

What was added:

- A 13-command reduced subset covering E0, E1, E2, and E3.
- A first-pass algorithm set focused on `erm`, `groupdro`, `iro`, and `inftask` where appropriate.
- Separate experiment names for the reduced reproduction, domain-count, sample-size, and imbalance checks.

What this enables:

- Running the planned small subset before attempting the full 450-command sweep.
- Faster iteration on the analysis and reporting pipeline.

### 5. Stress-Aware Result Aggregation

File updated:

- `CMNIST/collect_results.py`

What was added:

- Recursive result-file loading.
- Derived metadata fields for:
  - `phase`
  - `n_train_domains`
  - `sample_size_per_domain`
  - `imbalance_type`
- Derived summary metrics including worst-domain and average-domain accuracy.
- A grouped-summary CLI path using `--group_by` and `--metric`.

What this enables:

- Summarizing stress-test outputs by condition rather than only by the original CMNIST table format.
- Reusing the same result records for both execution logging and downstream analysis.

### 6. Lambda-Grid Evaluation Entry Point

File added:

- `CMNIST/evaluate_lambda_grid.py`

What was added:

- A checkpoint-based λ-grid evaluator.
- Support for evaluating `iro`, `inftask`, and fixed-prediction baselines across a configurable λ grid.
- Saved JSONL outputs containing:
  - `lambda_eval`
  - per-environment accuracy and loss
  - aggregated risk
  - summary accuracy statistics

What this enables:

- Running the planned E4 λ-sensitivity analysis on saved CMNIST checkpoints.

### 7. Stress-Test Plotting Entry Point

Files added or updated:

- `CMNIST/plot_domain_stress.py`
- `CMNIST/requirements.txt`

What was added:

- A plotting script for the first stress-test figures.
- `matplotlib>=3.7.0` added to CMNIST requirements.

What this enables:

- Plotting E0 accuracy-by-environment curves.
- Plotting E1, E2, and E3 worst-domain summary figures from aggregated result records.

### 8. Multi-Seed Bash Runner

File added:

- `CMNIST/job_scripts/run_domain_stress_small_seeds.sh`

What was added:

- A bash script that loops over seeds 0–2 and runs domain_stress_small.txt with separate output directories.
- Automatically appends `--seed`, `--deterministic`, `--n_workers 0`, and `--output_dir ../cmnist_exp_small_seed<N>` to each command.
- Preserves reproducibility controls across multi-seed runs.

What this enables:

- Running the reduced sweep across multiple seeds without manual command editing.
- Collecting per-seed result artifacts in isolated directories for downstream analysis.

### 9. CSV Export Infrastructure

File added:

- `CMNIST/export_results_csv.py`

What was added:

- A script that converts JSONL result files to reportable CSV formats.
- Three output formats:
  - `*_run_level.csv`: Flattened records with all args columns + metrics (one row per JSONL record).
  - `*_env_metric_long.csv`: Long-format per-test-environment metrics (test_env, metric, model_selection, value columns).
  - `*_summary.csv`: Grouped aggregates by phase, algorithm, n_train_domains, and imbalance type.

What this enables:

- Exporting JSONL results to human-readable tabular formats.
- Analysis-ready long-format tables for plotting and statistical summary.
- Spreadsheet-compatible aggregated summaries for reporting.

## Dependency Update

File changed:

- `CMNIST/requirements.txt`

What was added:

- `scipy>=1.10.1`

Reason:

- `CMNIST/lib/iro_utils.py` imports `scipy.stats.beta`, so the original requirements were incomplete for CMNIST execution.

## Validation Results

### 1. Virtual Environment

Completed:

- Removed the old `.venv` environment.
- Created the README-style environment `dgil_env` at the repository root.
- Installed the CMNIST requirements into `dgil_env`.
- Switched the workspace interpreter to `dgil_env`.

Interpreter path:

- `C:\Users\<USER_ID>\PycharmProjects\domain_gen_IL\dgil_env\Scripts\python.exe`

### 2. GPU Feasibility Check

Command result inside `dgil_env`:

```python
{'torch_version': '2.4.1+cu124', 'cuda_available': True, 'cuda_device_count': 1, 'cuda_version': '12.4', 'device_name': 'NVIDIA RTX A1000 6GB Laptop GPU'}
```

Interpretation:

- The environment is functional.
- The installed PyTorch build is CUDA-enabled.
- PyTorch can see one CUDA device in the current environment.

Current status:

- Both CPU and GPU execution are now feasible in `dgil_env`.
- The active CUDA-visible device is `NVIDIA RTX A1000 6GB Laptop GPU`.

### 3. CMNIST Smoke Test

The following minimal stress-test run completed successfully:

```bash
cd CMNIST
..\dgil_env\Scripts\python.exe train_sandbox.py \
  --steps 1 \
  --eval_freq 1 \
  --batch_size 256 \
  --algorithm erm \
  --train_envs 0.1,0.2,0.5,0.9 \
  --train_env_sizes 128,128,128,128 \
  --test_envs 0.1,0.9 \
  --output_dir ../cmnist_exp_smoke \
  --exp_name smoke_domain_stress \
  --n_workers 0
```

Observed evidence that the new implementation worked:

- CLI arguments were parsed correctly.
- `train_env_ps` matched the requested 4-domain setup.
- `train_env_sizes_parsed` matched `128,128,128,128`.
- Reported training environment sample counts were:

```text
[128, 128, 128, 128]
```

This confirms that:

- Phase 1 domain-count control works.
- Phase 2 explicit sample-size control works.
- Phase 3 imbalance support is available through the same mechanism.

### 4. Stress Command Generation

Validation command:

```bash
cd CMNIST
..\dgil_env\Scripts\python.exe job_scripts\gen_exps.py \
  --data_dir c:/Users/<USER_ID>/PycharmProjects/domain_gen_IL/data \
  --output_dir c:/Users/<USER_ID>/PycharmProjects/domain_gen_IL/cmnist_exp \
  --exp_name domain_stress
```

Result:

- Generated `CMNIST/job_scripts/domain_stress.txt`
- Total commands generated: `450`

This confirms that the batch entry point for the stress-test experiment is live.

### 5. Smoke and Subset Run Status

The current execution status is:

- `smoke_domain_stress`: completed successfully.
- `smoke_phase1_domains`: completed successfully.
- `smoke_phase2_balanced_sizes`: completed successfully.
- `smoke_phase3_imbalance`: completed successfully.

What these runs covered:

- `smoke_domain_stress` validated the original end-to-end smoke path for explicit domain-size control.
- `smoke_phase1_domains` validated Phase 1 with a reduced number of training domains.
- `smoke_phase2_balanced_sizes` validated Phase 2 with balanced explicit per-domain sample caps.
- `smoke_phase3_imbalance` validated Phase 3 with an imbalanced sample schedule.

Observed status for each subset run:

- Phase 1 respected `train_envs=0.1,0.2` and reported training sample counts `[25000, 25000]`.
- Phase 2 respected `train_env_sizes=128,128,128,128` and reported `[128, 128, 128, 128]`.
- Phase 3 respected `train_env_sizes=128,128,128,512` and reported `[128, 128, 128, 512]`.

Output locations:

- Results root: `cmnist_exp_smoke/results/`
- Logs root: `cmnist_exp_smoke/logs/`

Generated result folders:

- `cmnist_exp_smoke/results/smoke_domain_stress/`
- `cmnist_exp_smoke/results/smoke_phase1_domains/`
- `cmnist_exp_smoke/results/smoke_phase2_balanced_sizes/`
- `cmnist_exp_smoke/results/smoke_phase3_imbalance/`

Generated log folders:

- `cmnist_exp_smoke/logs/smoke_domain_stress/`
- `cmnist_exp_smoke/logs/smoke_phase1_domains/`
- `cmnist_exp_smoke/logs/smoke_phase2_balanced_sizes/`
- `cmnist_exp_smoke/logs/smoke_phase3_imbalance/`

Result-file checks confirm that each subset run wrote a JSONL result record whose saved arguments match the intended phase configuration.

### 6. Aggregation Validation

Validation command:

```bash
cd CMNIST
..\dgil_env\Scripts\python.exe collect_results.py ..\cmnist_exp_smoke\results --group_by phase --metric worst_domain_acc_best
```

Observed grouped output included:

- `domain_count`
- `sample_size`
- `imbalance`
- `validation_smoke`

Interpretation:

- The updated aggregation path can now derive and group the current smoke results by stress-test phase.

### 7. Plotting Validation

Validation command:

```bash
cd CMNIST
..\dgil_env\Scripts\python.exe plot_domain_stress.py ..\cmnist_exp_smoke\results --output_dir ..\cmnist_exp_smoke\plots
```

Generated figures:

- `cmnist_exp_smoke/plots/e0_accuracy_by_test_env.png`
- `cmnist_exp_smoke/plots/e1_domain_count_worst_domain_accuracy.png`
- `cmnist_exp_smoke/plots/e2_sample_size_worst_domain_accuracy.png`
- `cmnist_exp_smoke/plots/e3_imbalance_worst_domain_accuracy.png`

Interpretation:

- The plotting path is now working end to end on the existing smoke results.

### 8. Lambda Evaluation Script Validation

Validation command:

```bash
cd CMNIST
..\dgil_env\Scripts\python.exe evaluate_lambda_grid.py --help
```

Interpretation:

- The λ-grid evaluator entry point parses correctly and is ready to run once saved checkpoints are available.
- A full λ-evaluation run was not executed yet in this validation pass because the current smoke outputs do not include saved checkpoints.

## Files Added or Updated

Updated:

- `CMNIST/datasets.py`
- `CMNIST/train_sandbox.py`
- `CMNIST/job_scripts/gen_exps.py`
- `CMNIST/collect_results.py`
- `CMNIST/requirements.txt`

Added:

- `CMNIST/job_scripts/domain_stress_small.txt`
- `CMNIST/job_scripts/run_domain_stress_small_seeds.sh`
- `CMNIST/evaluate_lambda_grid.py`
- `CMNIST/plot_domain_stress.py`
- `CMNIST/export_results_csv.py`
- `EXPERIMENT_PLANS.md`
- `IMPLEMENTATION_PROGRESS.md`

## [CMN] Current Execution Status

### ✅ Completed Work

- **Reduced sweep (domain_stress_small)**: All 13 commands executed successfully with seed 0.
  - Covers E0 (reproduction), E1 (domain count), E2 (sample size), and E3 (imbalance).
  - Results saved under `cmnist_exp_small/results/` and `cmnist_exp_small/logs/`.
  - Checkpoints saved for `groupdro`, `iro`, and `inftask` runs.

- **CSV export**: Successfully tested on reduced-sweep results.
  - Generated 3 CSV types under `cmnist_exp_small/`:
    - `cmnist_exp_small_run_level.csv` (flattened JSONL records)
    - `cmnist_exp_small_env_metric_long.csv` (per-test-env analysis-ready format)
    - `cmnist_exp_small_summary.csv` (aggregated by phase/algorithm)

- **Multi-seed bash runner**: Created and documented for future use.
  - Ready to run seeds 0–2 with separate output directories.
  - Command: `bash CMNIST/job_scripts/run_domain_stress_small_seeds.sh`

### ⏳ Pending Work

- **Reduced-main staged sweep (domain_stress_main_seed0..4)**: partially executed.
  - Manifest files exist for seeds 0–4, each with 45 commands.
  - Current result artifacts contain records for seeds 0–2, including recovery/duplicate records; they should not be treated as a clean 135-job completion without deduplication and log review.
  - Seeds 3–4 have not been verified as completed.

- **Full sweep (domain_stress.txt)**: 450 commands available as the current full-grid generator output (10 seeds × 45 jobs/seed).
  - Intended for final publication-grade results (all 5 algorithms, 10 seeds).
  - Throughput-based runtime estimate is now much lower than earlier drafts; use recent per-seed log timing as the planning baseline.

- **Lambda-grid evaluation (E4)**: The script and checkpoint-loading path exist; report-grade CMNIST lambda results and plots remain pending.

## Experiment Budget & Scope

### Jobs Per Seed Breakdown

The full experiment sweep (`domain_stress.txt`) is organized as follows (time estimates use ~15-25 minutes per job from recent logs):

| Experiment | Condition Variations | Algorithms | Jobs/Seed | Total Time/Seed | Notes |
|---|---|---|---|---|---|
| **E0** (Baseline/Reproduction) | 1 setting: default 4 domains, balanced | erm, irm, groupdro, iro, inftask | 5 | ~1.25-2.1 h | Baseline configuration; all algorithms tested |
| **E1** (Domain Count) | 3 settings: n_train_domains = {2, 4, 8} | erm, irm, groupdro, iro, inftask | 15 | ~3.75-6.25 h | 3 conditions × 5 algorithms |
| **E2** (Sample Size) | 3 settings: train_env_sizes = {2000,2000,2000,2000}, {4000,4000,4000,4000}, {8000,8000,8000,8000} | erm, irm, groupdro, iro, inftask | 15 | ~3.75-6.25 h | 4 domains, balanced sizes |
| **E3** (Imbalance) | 3 settings: balanced {2000,2000,2000,2000}, mild {2000,2000,2000,4000}, strong {2000,2000,2000,10000} | erm, irm, groupdro, iro, inftask | 15 | ~3.75-6.25 h | 4 domains, severity-based imbalance |
| **E4** (λ-Sensitivity)* | λ grid: 0.0, 0.1, ..., 1.0 (11 points), on selected checkpoints | iro, inftask | ~22-44* | ~5.5-18.3 h* | Separate λ-grid evaluation on E0-E3 checkpoints |
| **TOTAL (E0-E3)** | 10 condition settings across E0-E3 | — | **45 per seed** | **~11.25-18.75 h** | Full sweep across all seeds |

*E4 is evaluated separately post-hoc using saved checkpoints from E0–E3 runs; not included in the 450-command main sweep count.

### Sweep Configurations

#### Small Sweep (`domain_stress_small.txt`)
- **Scope**: E0–E3 baseline validation
- **Seeds**: 1 (seed 0 only)
- **Algorithms**: 4 (erm, groupdro, iro, inftask; excludes irm)
- **Total jobs**: 13
- **Purpose**: Fast iteration, smoke testing, result pipeline validation
- **Status**: ✅ Complete

#### Reduced-Main Staged Sweep (`domain_stress_main_seed0..4.txt`)
- **Scope**: E0–E3 with seed staging
- **Seeds**: 5 (seeds 0–4, separate command files)
- **Algorithms**: 5 (erm, irm, groupdro, iro, inftask)
- **Jobs per seed**: 45
- **Total jobs**: 225 (5 seeds × 45 jobs/seed)
- **Purpose**: Intermediate robustness validation before full sweep
- **Status**: 🟡 Partially complete (seeds 0–2 done, seeds 3–4 pending)

#### [CMN] Full Sweep (`domain_stress.txt`)
- **Scope**: E0–E3 complete coverage
- **Seeds**: 10 (seeds 0–9)
- **Algorithms**: 5 (erm, irm, groupdro, iro, inftask)
- **Jobs per seed**: 45
- **Total jobs**: 450 (10 seeds × 45 jobs/seed)
- **Purpose**: Publication-grade final results
- **Status**: ⏳ Pending

### Budget Summary

| Milestone | Jobs | Seeds | Status |
|---|---|---|---|
| Smoke tests | ~26 | 1 (s0) | ✅ Complete |
| Small sweep | 13 | 1 (s0) | ✅ Complete |
| Reduced-main (staged) | 225 | 5 (s0–s4) | 🟡 135/225 (s0–s2) |
| Full sweep | 450 | 10 (s0–s9) | ⏳ Pending |
| **Grand total** | **~714** | **~13–26 total** | — |

### Runtime Estimates

Based on recent execution logs from seeds 1–2:

- **Per-job average**: ~15–25 minutes (GPU-enabled, 1 seed)
- **Per-seed (45 jobs)**: ~11–19 hours
- **Full sweep (450 jobs, 10 seeds)**: ~110–190 hours (~5–8 days at continuous throughput)

**Recommended strategy**:
1. Complete reduced-main sweep (seeds 3–4) before staging full sweep.
2. Use per-seed log timing to estimate total wallclock time accounting for system load.
3. Consider distributed execution if multiple GPUs available.

## Current Limitations

### Sweep Scope

The `domain_stress` generator currently emits **450 commands** (10 seeds × 45 condition+algorithm combos):

- **Small sweep** (`domain_stress_small.txt`): 13 commands, seed 0 only. ✅ Complete.
  - Covers E0–E3 with representative subsets (erm, groupdro, iro, inftask; no irm).
  - Test environments: explicit `0.1,0.5,0.9`.

- **Full sweep** (`domain_stress.txt`): 450 commands, seeds 0–9.
  - Covers E0–E3 with all algorithms (erm, irm, groupdro, iro, inftask).
  - Test environments: determined by train-env defaults.
  - E1 domain-count conditions: 2, 4, 8.
  - E2 sample-size conditions: 2000, 4000, 8000 per domain.
  - E3 imbalance conditions: balanced, mild_imbalance, strong_imbalance.

### Stress-Grid Interpretation

The current generated `domain_stress` sweep is functional but has caveats:

- The generated Phase 1 train-environment sets use the repo's predefined sets (e.g., `[0.01, 0.12, 0.5, 0.99]` for 4 domains), not the cleaner comparison sets proposed in `EXPERIMENT_PLANS.md`.
- The generated Phase 3 sweep is currently severity-based (balanced/mild/strong) and does not include mirrored first-heavy/last-heavy directional variants.
- If majority-vs-minority directional claims are needed, add mirrored schedules explicitly before final reporting.

### [CMN] Lambda Evaluation

The dedicated λ-grid evaluation script is ready but not yet validated on actual checkpoint outputs.

## Planned Next Tasks

### Priority 1: Execute Full Lambda-Grid Evaluation

Run `CMNIST/evaluate_lambda_grid.py` on saved `iro` and `inftask` checkpoints from the completed reduced sweep.

Command pattern:
```bash
cd CMNIST
..\dgil_env\Scripts\python.exe evaluate_lambda_grid.py \
  ../cmnist_exp_small/ckpts \
  --output_dir ../cmnist_exp_small/lambda_results \
  --lambda_grid 0.0:1.0:0.1
```

Reason:

- This will convert the λ-evaluation script from an entry point into validated outputs.
- Required for E4 (lambda-sensitivity) plotting.

### Priority 2: Complete E4 Plotting

Run `CMNIST/plot_domain_stress.py` with λ-evaluation outputs to generate the E4 aggregated-risk figure.

Command:
```bash
cd CMNIST
..\dgil_env\Scripts\python.exe plot_domain_stress.py \
  ../cmnist_exp_small/results --output_dir ../cmnist_exp_small/plots
```

Reason:

- E0-E3 plotting is validated; E4 needs actual λ-evaluation outputs.

### Priority 3: Optional Multi-Seed Run

Execute the bash runner to generate seeds 1–2 results for robustness validation.

Command:
```bash
cd CMNIST/job_scripts
bash run_domain_stress_small_seeds.sh
```

Reason:

- Validates the multi-seed automation workflow.
- Provides seed-level variability for final reporting (optional for publication).

### Priority 4: Staging Full Sweep (Future)

When ready, execute the full `domain_stress.txt` sweep across all 10 seeds and all 5 algorithms.

Considerations:

- Current generated full sweep size: 450 jobs.
- Runtime should be estimated from recent per-seed logs (seed1/seed2), rather than the older 6-12 day placeholder.
- Can be run in batch mode or distributed across multiple machines.
- Use the same CSV export + plotting pipeline for final aggregation.

### Priority 5: Optional Generator Cleanup (Nice-to-Have)

Consider updating `CMNIST/job_scripts/gen_exps.py` to support:

- Semantic phase labels in generated commands or saved metadata.
- An explicit `--small_sweep` mode that emits only `domain_stress_small.txt`.

Reason:

- Improves final interpretability but is lower priority than the λ and multi-seed work.

## Recommended Immediate Next Steps

1. Run `CMNIST/evaluate_lambda_grid.py` on the completed reduced-sweep checkpoints → generates E4 outputs.
2. Run `CMNIST/plot_domain_stress.py` with the λ-evaluation outputs → generates full E0–E4 figure set.
3. Optional: Execute `run_domain_stress_small_seeds.sh` to validate multi-seed automation (seeds 1–2).
4. Export all results using `CMNIST/export_results_csv.py` for final reporting.
5. Stage `domain_stress.txt` for batch execution when compute resources are available.

# AISTATS / CMNIST Closure TODOs

## Priority 0 — Freeze the exact scientific object

* [x] Trace the CMNIST training code from per-example losses to pooled source minibatch losses and CVaR.
* [x] Determine that IRO uses pooled per-example losses from active source minibatches and does not apply \(n_a/N\) weights in training.
* [x] Write the implemented IRO objective and CVaR behavior in `README_[CMN]_EXPERIMENT_RESULTS.md`.
* [x] Write the implemented GroupDRO objective in `README_[CMN]_EXPERIMENT_RESULTS.md`.
* [x] State explicitly that logged empirical priors \(n_a/N\) are used for E3b analysis, not the CMNIST training objectives.
* [x] Confirm that zero-count domains are removed from training but retained with positive deployment weight in test analysis.
* [x] Align the README notation with the implemented weighting rule before making prior-mismatch claims.

## Priority 1 — Clean the current result records

* [x] Build a canonical run key using algorithm, seed, experiment phase, train environments, train sizes, steps, hyperparameters, and model-selection rule (`CMNIST/audit_results.py`).
* [x] Deduplicate complete candidates into a new non-destructive export (`results/cmnist_exp_clean_priority1_v3/clean_training_results.jsonl`). Recovery/duplicate candidates remain flagged in the audit.
* [x] Separate training records from post-training lambda-evaluation records (`clean_training_results.jsonl` and `lambda_results.jsonl`).
* [x] Inspect the available runner logs and classify 140 sequential `Args:` blocks; all reached final and best accuracy markers with no detected error markers. The 15 duplicate/recovered result candidates remain flagged in the result audit.
* [x] Produce a canonical result audit table with source file and source line for each retained run (`results/cmnist_exp/audit_priority1_v5.csv`).
* [x] Produce a manifest-to-result audit (`results/cmnist_exp/manifest_audit.csv`); 125 of 215 staged commands match retained clean records.
* [x] Do not interpret 553 JSONL rows as 553 independent jobs.
* [x] Recompute means and standard deviations only from the 120 selected complete records (`results/cmnist_exp_clean_priority1_v4/clean_metric_summary.csv`).
* [x] Export a new versioned clean result root rather than overwriting the current result root (`results/cmnist_exp_clean_priority1_v3/`).
* [x] Record the exact available seed set in the audit export; the retained clean training records cover seeds 0, 1, and 2.

Priority 1 audit command:

```text
dgil_env\Scripts\python.exe CMNIST\audit_results.py results\cmnist_exp\results --output results\cmnist_exp\audit_priority1_v5.csv --clean_output results\cmnist_exp_clean_priority1_v4 --lambda_results_dir results\cmnist_exp\lambda_results --log_dir results\cmnist_exp\logs
```

The current audit reads 135 source rows and identifies 120 canonical keys: 105 unique complete candidates and 15 keys with duplicate or recovered candidates. It exports 120 selected complete records and recomputed metric summaries without modifying the raw result files.

## Priority 2 — Close the main tail-support experiment

* [x] Decide that E3b will contain four final conditions: `balanced_visible`, `long_tail_visible`, `near_missing_tail`, and `missing_tail`.
* [x] Run the missing `near_missing_tail` condition for all intended algorithms and seeds using `CMNIST/job_scripts/e3b_tail_support_near_missing_seed012.txt` (15 jobs).
* [x] Reject the three-condition alternative; `near_missing_tail` remains part of the final experimental design.
* [x] Confirm that all E3b conditions use the intended fixed test environments: `0.0,0.1,...,1.0`.
* [x] Confirm the exact head-to-tail ordering: source environments `0.1,0.2,0.5,0.9`, with head `0.1` and tail `0.9`.
* [x] Verify that `missing_tail` assigns zero training samples but positive evaluation weight to the missing domain. The saved record has empirical tail weight `0.0`, deployment weight `1/11`, and active source environments `0.1,0.2,0.5`.
* [x] Regenerate `raw_results.csv`, `summary_by_condition.csv`, and `slide_table.csv` from the completed 60-record E3b result set.
* [x] Regenerate tail accuracy, worst-domain accuracy, head-tail gap, and CVaR-gap plots from the same E3b CSV source.
* [x] Report per-seed values in `raw_results.csv` in addition to mean and standard deviation in `summary_by_condition.csv`.
* [x] Treat E3b as the principal CMNIST experiment in the AISTATS paper, subject to the remaining empirical GroupDRO and lambda audits.

## Priority 3 — Audit GroupDRO

* [x] Verify that GroupDRO receives one minibatch per active source environment, preserving positional domain identifiers.
* [x] Verify that GroupDRO updates its group weights during training rather than behaving as ERM. Focused checks are in `CMNIST/tests/test_groupdro.py`.
* [x] Confirm the current sampling policy: `zip(*train_loaders)` supplies one batch from every active loader; all current E3b domains fit in one batch with `batch_size=25000`. Larger domains would require documenting the shortest-loader cycling caveat.
* [x] Report worst observed-source-group loss for the controlled comparison. The v2 results report GroupDRO source-worst loss per seed; historical stress records remain without source-loss fields. Manifest: `CMNIST/job_scripts/groupdro_controlled_seed012_v2.txt`.
* [x] Report worst held-out-target-domain accuracy separately; existing records retain per-test-environment accuracy and derived `worst_domain_acc_best`.
* [x] Check source-to-target transfer in the controlled comparison. GroupDRO improves observed-source worst loss substantially, but its held-out target worst/average accuracy is lower than ERM in this three-seed run; this is a limitation, not evidence of transfer.
* [x] Compare GroupDRO and ERM using the same architecture, optimizer, number of steps, batch policy, and model-selection rule. The v2 controlled three-seed manifest uses the same settings for both methods and completed six successful runs.
* [x] Record GroupDRO-specific settings: `groupdro_eta=0.1`, Adam after pretraining, learning-rate cosine schedule, and `save_ckpts` in the generated stress manifest.
* [x] Do not describe GroupDRO as “not working” unless it also fails on the observed-group objective it is designed to optimize.

### Priority 3 Findings

The GroupDRO implementation computes one scalar mean loss per active source minibatch, updates `q` by `q_a <- q_a exp(groupdro_eta * loss_a)`, normalizes `q`, and minimizes the weighted loss `sum_a q_a loss_a`. The initial `q` is uniform over active domains. Empirical sample-count priors are not used in this update.

The current clean historical export contains 24 GroupDRO records, all with `steps=1000` and `groupdro_eta=0.1`, but none has source-loss fields because those fields were not logged at the time. The matched-budget v2 comparison now provides source/target evidence in `results/cmnist_groupdro_control_v2/analysis/source_target_by_seed.csv`: GroupDRO source-worst loss mean `0.118 +/- 0.032` versus ERM `0.597 +/- 0.033`, while held-out target worst accuracy mean is `0.600 +/- 0.014` for GroupDRO versus `0.607 +/- 0.016` for ERM.

## Priority 4 — Correct E1 domain-count evaluation

* [x] Do not treat the original 2-, 4-, and 8-domain result as a clean domain-count ablation.
* [x] Replace the exploratory environment sets with nested or approximately symmetric sets in `CMNIST/job_scripts/domain_count_clean.txt`.
* [x] Remove duplicated environment values from the corrected 8-domain condition in `CMNIST/job_scripts/domain_count_clean.txt`; the original exploratory manifest remains preserved.
* [x] Keep the total source budget fixed in the corrected E1 generator and document that choice.
* [x] Keep test environments identical across all corrected domain-count conditions.
* [x] Run three seeds for the corrected E1 comparison; all 60 commands succeeded.
* [x] Decide not to add alternative domain subsets for the current AISTATS scope; corrected E1 is explicitly a controlled single-configuration secondary/appendix study.
* [x] State that no distribution across alternative domain subsets is available and avoid universal domain-count claims. Additional subsets remain optional strengthening.
* [x] Label the original non-nested E1 result exploratory/appendix-only; the corrected E1 result is available under `results/cmnist_domain_count_clean_v1/`.

## Priority 5 — Correct E3 imbalance interpretation

* [x] Rename the current E3 conditions as balanced, mild last-domain-heavy, and strong last-domain-heavy in the CMNIST documentation.
* [x] State that the last listed source environment is being overweighted.
* [x] Do not call the current result a general minority-underrepresentation experiment.
* [x] Add mirrored first-domain-heavy conditions in `CMNIST/job_scripts/imbalance_clean.txt`.
* [x] Add fixed-total-budget imbalance schedules in `CMNIST/job_scripts/imbalance_clean.txt` so prior direction is separated from total source count.
* [x] Report per-environment accuracy curves for balanced and strong first/last-heavy fixed-budget imbalance in `results/imbalance_clean_v1/analysis/e3_imbalance_accuracy_by_test_env.png`.
* [x] Keep the original E3 as supporting evidence for the distinction between visible imbalance and missing support; use only the corrected fixed-budget E3 for directional claims.

Priority 5 interpretation note: the corrected E3 is sufficient for the current paper. Report exact schedules and within-method balanced-to-imbalanced changes first; avoid unsupported majority/minority labels and strong cross-method claims because optimization schedules differ.

Current corrected E3 status: all `75/75` jobs succeeded with zero failures. The artifact index at `results/cmnist_artifact_index.csv` tracks this status alongside the completed E1, E3b, E4, and Priority 7 outputs.

## Priority 6 — Validate E4 lambda sensitivity

* [x] Match every lambda-evaluation record to one unique training checkpoint (`results/cmnist_lambda_audit_v1/lambda_coverage.csv`).
* [x] Check for duplicate or recovered checkpoint evaluations; the current 418 lambda rows contain no duplicate checkpoint/lambda keys.
* [x] Document the current evaluator behavior: `lambda_eval` is passed to the alpha-conditioned predictor and also to the CVaR aggregator; separate `lambda_model`/`alpha_eval` controls are not currently exposed.
* [x] For IRO and INF-TASK, report per-environment accuracy as `lambda_eval` changes in `results/cmnist_lambda_prediction_eval_v2/prediction_lambda_metrics.csv`. The current evaluator uses the same value for model conditioning and risk aggregation.
* [x] State clearly that ERM and GroupDRO predictions are fixed under lambda evaluation and only the evaluation risk functional would change; current lambda artifacts contain no ERM/GroupDRO rows.
* [x] Compute prediction disagreement between low- and high-lambda predictions. In the selected v2 checkpoints, maximum disagreement from lambda 0 is about 3.9% for IRO and 15.9% for INF-TASK.
* [x] Compute the best-to-worst accuracy range across lambda. The per-checkpoint mean-environment ranges are about 0.25 percentage points for IRO and 3.19 percentage points for INF-TASK.
* [x] Compute the maximum neighboring-lambda change. The per-checkpoint mean-environment maxima are about 0.98 percentage points for IRO and 8.56 percentage points for INF-TASK.
* [x] Keep non-invasive deployment-wide pseudo-regret logging from existing lambda metrics (`CMNIST/analyze_regret.py` and `results/cmnist_lambda_pseudoregret_v2/`). One oracle lambda is selected per checkpoint using mean accuracy across deployment environments; this is not empirical operator regret.
* [ ] Compute true operator regret relative to fixed-lambda reference models if those references are later trained. This is optional for the current AISTATS study; the preferred reduced version is 9 IRO runs (`3 lambdas x 3 seeds`), while the broader report version is 135 runs (`3 lambdas x 3 algorithms x 3 seeds x 5 conditions`, where the 5 conditions are E0 + 4 E3b conditions).
* [x] Regenerate the lambda figure from the clean fixed-input prediction evaluation: `results/cmnist_lambda_prediction_eval_v2/plots/lambda_prediction_accuracy_curve.png`.
* [x] Do not claim successful preference conditioning from a flat aggregated-risk curve alone.
* [x] Treat the current E4 interpretation as provisional: model-level disagreement and sensitivity checks are complete for the selected checkpoints, but operator-regret references and a clean final figure remain pending.

Priority 6 audit outputs:

- `results/cmnist_lambda_audit_v1/lambda_records_deduplicated.jsonl`
- `results/cmnist_lambda_audit_v1/lambda_sensitivity_summary.csv`
- `results/cmnist_lambda_audit_v1/lambda_coverage.csv`

## Priority 7 — Add the theory-aligned synthetic experiment

* [x] Implement the balanced deployment prior and long-tailed empirical prior in `CMNIST/priority7_theory.py`.
* [x] Implement two controlled risk profiles with a head-versus-tail trade-off.
* [x] Compute deployment CVaR and empirical CVaR for both hypotheses.
* [x] Estimate ranking-reversal probability over repeated source samples.
* [x] Sweep CVaR level, sample size, missing-tail fraction, and long-tail exponent.
* [x] Produce heatmap and sample-size plots of ranking-reversal probability.
* [x] Verify the sample-size effect on disagreement. It is bimodal, not monotone in one direction: with `exponent=0` (no persistent mismatch) reversal probability shrinks toward 0 as sample size grows; with `exponent>0` (persistent long-tailed mismatch) reversal probability instead rises toward 1, because more samples make the empirical prior converge more confidently to a fixed target that differs systematically from deployment. See `README_CMN_EXPERIMENT_RESULTS.md` Priority 7 section for the full breakdown.
* [x] Produce the theorem-aligned identification-width diagnostic: `results/cmnist_priority7_theory_v1/identification_width_by_alpha.csv` and `identification_width_by_alpha.png`. It uses the explicit bounded-risk assumption that missing-domain risks lie in `[0,1]` and reports the conservative bound `min(1, epsilon/(1-alpha))`.
* [x] Use this experiment as a direct synthetic validation of the ranking-reversal mechanism; it is not a complete proof of the main theorem.
* [x] Keep neural-network optimization effects separate from the synthetic identification result.

Priority 7 outputs:

- `results/cmnist_priority7_theory_v1/ranking_reversal_summary.csv`
- `results/cmnist_priority7_theory_v1/trial_diagnostics.csv`
- `results/cmnist_priority7_theory_v1/ranking_reversal_heatmap.png`
- `results/cmnist_priority7_theory_v1/ranking_reversal_by_sample_size.png`

## Priority 8 — Complete report-grade CMNIST evidence

* [x] Select a final core seed count of three clean seeds for the current report scope; seeds 0, 1, and 2 are available in the corrected E1 and E3b runs.
* [x] Complete seeds 3-4 for five-seed reporting. E1 (100/100), E3 (125/125), and E3b (100/100) all now cover seeds 0-4 with zero failures; E1 seed-3 raw duplicates from pre-fix launcher retries were resolved by `CMNIST/dedup_and_refresh_e1_e3.py`, keeping the most recently written record per (seed, algorithm, train_envs, train_env_sizes).
* [x] Prefer a smaller clean theory-aligned experiment set over completing all 450 jobs; the current report scope uses corrected E1, complete E3b, controlled GroupDRO, lambda sensitivity, and Priority 7 simulation outputs.
* [x] Reproduce the corrected E1 summary directly from the versioned CSV `results/cmnist_domain_count_clean_v1/analysis_seed0-4/domain_count_by_algorithm.csv`.
* [x] Verify corrected E1 and E3 summary mean/std values against clean records; all 60 E1 checks pass in `results/cmnist_domain_count_clean_v1/analysis_seed0-4/E1_report_table_verification.csv`, and all 75 E3 checks pass in `results/imbalance_clean_v1/analysis_seed0-4/E3_report_table_verification.csv`.
* [x] Report the exact result roots, command manifests, seed sets, algorithm sets, and model-selection rules in the CMNIST results README and generated audit files.
* [x] Record architecture, optimizer, learning rate, batch size, steps, evaluation frequency, and checkpoint-selection details in saved `args` and the CMNIST documentation.
* [x] Clearly distinguish smoke, reduced, staged-main, corrected E1, E3b, and final report-grade results.
* [x] Add explicit limitations for uncontrolled domain choice, incomplete subset coverage, unavailable true regret references, and computational budget.

## Priority 9 — ImageNet-C optional pilot

* [ ] Do not begin a broad ImageNet-C sweep until the CMNIST result root and E4 evaluation are clean.
* [ ] Keep the native 1000-class task.
* [ ] Use corruption type as the primary domain variable.
* [ ] Keep severity as a separate ordered analysis axis rather than flattening all 75 conditions into unrelated domains.
* [ ] Validate the ImageNet and ImageNet-C directory layouts before training.
* [ ] Verify deterministic evaluation repeatability on one fixed checkpoint.
* [ ] Run one held-out-corruption fold, one seed, and ERM/GroupDRO/IRO first.
* [ ] Confirm that the IRO head is actually lambda-conditioned during training.
* [ ] Expand to three folds and three seeds only if the pilot is interpretable.
* [ ] Treat ImageNet-C as optional external validation for AISTATS and as a fuller experiment for the journal extension.

## Priority 10 — AISTATS readiness gate

* [ ] The implemented objective and the theoretical probability law match.
* [x] The main result root is deduplicated and auditable through the Priority 1 clean export and audit CSVs.
* [x] The E3b support experiment is complete with four conditions and 100 records (seeds 0-4).
* [x] GroupDRO is validated on its observed-source objective through the matched-budget v2 comparison.
* [x] E4 measures actual predictor adaptation to lambda for selected IRO and INF-TASK checkpoints.
* [x] The synthetic ranking-reversal experiment is complete as a qualified mechanism-level validation.
* [ ] The main theoretical result goes beyond the perturbation inequality and quantifies missing-support identification error or operator regret.
* [ ] All main tables and figures can be regenerated from versioned scripts and clean CSV files.
* [x] Any unfinished ImageNet-C work is explicitly marked as optional or future work.
