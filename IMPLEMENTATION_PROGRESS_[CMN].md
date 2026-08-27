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

#  AISTATS / CMNIST Closure TODOs

## Priority 0 — Freeze the exact scientific object

* [ ] Trace the CMNIST training code from per-example losses to per-domain risks and then to CVaR.
* [ ] Determine whether IRO assigns equal mass to each active domain or weights domains proportionally to sample counts.
* [ ] Write the exact implemented IRO objective in one equation in the README.
* [ ] Write the exact implemented GroupDRO objective in one equation in the README.
* [ ] State explicitly whether logged empirical priors \(n_a/N\) are used in training, used only for analysis, or both.
* [ ] Confirm that zero-count domains are removed from training but retained with positive weight in deployment/test evaluation.
* [ ] Align the theoretical notation with the implemented weighting rule before finalizing any prior-mismatch claim.

## Priority 1 — Clean the current result records

* [ ] Build a canonical run key using algorithm, seed, experiment phase, train environments, train sizes, steps, hyperparameters, and model-selection rule.
* [ ] Deduplicate recovery, partial, and repeated JSONL records in `results/cmnist_exp/`.
* [ ] Separate training records from post-training evaluation records.
* [ ] Inspect the logs for every retained run and mark each run as complete, failed, recovered, or partial.
* [ ] Produce a manifest-to-result audit table showing one intended command and one retained final record per run.
* [ ] Do not interpret 553 JSONL rows as 553 independent jobs.
* [ ] Recompute all means and standard deviations only from the clean retained records.
* [ ] Export a new versioned clean result root rather than overwriting the current result root.
* [ ] Record the exact seed set used in every reported table.

## Priority 2 — Close the main tail-support experiment

* [ ] Decide whether E3b will contain three or four final conditions.
* [ ] If using four conditions, run the missing `near_missing_tail` condition for all intended algorithms and seeds.
* [ ] If using three conditions, remove `near_missing_tail` from the final experimental design and state that the final study uses balanced-visible, long-tail-visible, and missing-tail.
* [ ] Confirm that all E3b conditions use the intended fixed test environments.
* [ ] Confirm the exact head-to-tail ordering of the four source environments.
* [ ] Verify that `missing_tail` assigns zero training samples but positive evaluation weight to the missing domain.
* [ ] Regenerate `raw_results.csv`, `summary_by_condition.csv`, and `slide_table.csv` from the clean records.
* [ ] Regenerate tail accuracy, worst-domain accuracy, head-tail gap, and CVaR-gap plots from the same clean CSV source.
* [ ] Report per-seed values in addition to mean and standard deviation.
* [ ] Treat E3b as the principal CMNIST experiment in the AISTATS paper.

## Priority 3 — Audit GroupDRO

* [ ] Verify that GroupDRO receives correct group/domain identifiers during training.
* [ ] Verify that GroupDRO updates its group weights during training rather than behaving as ERM.
* [ ] Confirm that all source domains are represented in each training cycle or document the sampling policy.
* [ ] Report worst observed-source-group loss for GroupDRO.
* [ ] Report worst held-out-target-domain accuracy separately.
* [ ] Check whether GroupDRO improves source-worst performance but fails to transfer that improvement to unseen targets.
* [ ] Compare GroupDRO and ERM using the same architecture, optimizer, number of steps, batch policy, and model-selection rule.
* [ ] Record any GroupDRO-specific regularization or hyperparameter choices.
* [ ] Do not describe GroupDRO as “not working” unless it also fails on the observed-group objective it is designed to optimize.

## Priority 4 — Correct E1 domain-count evaluation

* [ ] Do not treat the current 2-, 4-, and 8-domain result as a clean domain-count ablation.
* [ ] Replace the current environment sets with nested or approximately symmetric sets.
* [ ] Remove duplicated environment values from the 8-domain condition.
* [ ] Keep either the total sample budget or the samples per domain fixed and state which quantity is controlled.
* [ ] Keep test environments identical across all domain-count conditions.
* [ ] Run at least three seeds for the corrected E1 comparison.
* [ ] Consider sampling several different domain subsets for each domain count.
* [ ] Report the distribution across domain subsets rather than relying on one favorable or unfavorable subset.
* [ ] Move the current E1 result to the appendix or label it exploratory if the corrected run is not completed.

## Priority 5 — Correct E3 imbalance interpretation

* [ ] Rename the current E3 conditions as balanced, mild last-domain-heavy, and strong last-domain-heavy.
* [ ] State which environment is being overweighted.
* [ ] Do not call the current result a general minority-underrepresentation experiment.
* [ ] Add mirrored first-domain-heavy conditions if directional imbalance claims are required.
* [ ] Consider fixed-total-budget imbalance schedules so that imbalance is not confounded with total sample size.
* [ ] Report per-environment accuracy curves for balanced and strong imbalance.
* [ ] Keep E3 as supporting evidence for the distinction between visible imbalance and missing support.

## Priority 6 — Validate E4 lambda sensitivity

* [ ] Match every lambda-evaluation record to one unique training checkpoint.
* [ ] Remove duplicate or recovered checkpoints before computing lambda statistics.
* [ ] Distinguish `lambda_model`, which is supplied to \(h(x,\lambda)\), from `alpha_eval`, which changes the CVaR aggregation of fixed losses.
* [ ] For IRO and INF-TASK, report per-environment accuracy as `lambda_model` changes.
* [ ] For ERM and GroupDRO, state clearly that predictions are fixed and only the evaluation risk functional changes.
* [ ] Compute prediction disagreement between low- and high-lambda IRO predictions.
* [ ] Compute the best-to-worst accuracy range across lambda.
* [ ] Compute the maximum neighboring-lambda change.
* [ ] Compute operator regret relative to fixed-lambda reference models if those references are available.
* [ ] Regenerate the lambda figure from the clean checkpoint set.
* [ ] Do not claim successful preference conditioning from a flat aggregated-risk curve alone.
* [ ] Treat the current E4 interpretation as provisional until these checks are complete.

## Priority 7 — Add the theory-aligned synthetic experiment

* [ ] Implement the balanced deployment prior and long-tailed empirical prior.
* [ ] Implement two controlled risk profiles with a head-versus-tail trade-off.
* [ ] Compute deployment CVaR and empirical CVaR for both hypotheses.
* [ ] Estimate ranking-reversal probability over repeated source samples.
* [ ] Sweep CVaR level, sample size, missing-tail mass, and long-tail exponent.
* [ ] Produce one heatmap or phase diagram of ranking-reversal probability.
* [ ] Verify that disagreement increases as source evidence becomes less informative.
* [ ] Use this experiment as the direct validation of the main theorem.
* [ ] Keep neural-network optimization effects separate from the synthetic identification result.

## Priority 8 — Complete report-grade CMNIST evidence

* [ ] Select a final core seed count of at least three clean seeds.
* [ ] Complete seeds 3–4 only if five-seed reporting is needed after the three-seed result is stable.
* [ ] Prefer a smaller clean theory-aligned experiment set over completing all 450 jobs.
* [ ] Reproduce every paper table directly from a versioned CSV.
* [ ] Verify every table number against the CSV before submission.
* [ ] Report the exact result root, command manifest, seed set, algorithm set, and model-selection rule in each caption or methods subsection.
* [ ] Include architecture, optimizer, learning rate, batch size, steps, evaluation frequency, and checkpoint-selection details.
* [ ] Clearly distinguish smoke, reduced, staged-main, and final report-grade results.
* [ ] Add explicit limitations for uncontrolled domain choice, incomplete support, and computational budget where applicable.

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
* [ ] The main result root is deduplicated and auditable.
* [ ] The E3b support experiment is complete or formally reduced to three conditions.
* [ ] GroupDRO is validated on its observed-source objective.
* [ ] E4 measures actual predictor adaptation to lambda.
* [ ] The synthetic ranking-reversal experiment is complete.
* [ ] The main theoretical result goes beyond the perturbation inequality and quantifies missing-support identification error or operator regret.
* [ ] All main tables and figures can be regenerated from versioned scripts and clean CSV files.
* [ ] Any unfinished ImageNet-C work is explicitly marked as optional or future work.
