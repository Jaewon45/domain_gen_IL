# Implementation Progress

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
  - 6 domains
  - 8 domains
- Phase 2 balanced sample-size sweep:
  - 2000 per domain
  - 4000 per domain
  - 8000 per domain
- Phase 3 imbalance sweep on 4 domains:
  - balanced: `2000,2000,2000,2000`
  - last-domain-heavy mild: `2000,2000,2000,8000`
  - last-domain-heavy strong: `2000,2000,2000,12000`
  - first-domain-heavy mild: `8000,2000,2000,2000`
  - first-domain-heavy strong: `12000,2000,2000,2000`

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

- Running the planned small subset before attempting the full 500-command sweep.
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
- Total commands generated: `600`

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

## Current Execution Status

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

- **Full sweep (domain_stress.txt)**: 600 commands generated but not yet executed.
  - Intended for final publication-grade results (all 5 algorithms, 10 seeds).
  - Estimated runtime: 6–12 days on single GPU (100–200 hours wall-clock).

- **Lambda-grid evaluation (E4)**: Script exists but not yet run on saved checkpoints.
  - Will use `CMNIST/evaluate_lambda_grid.py` on reduced-sweep `iro` and `inftask` checkpoints.
  - Output needed for E4 plotting.

## Current Limitations

### Sweep Scope

The `domain_stress` generator currently emits **600 commands** (10 seeds × 60 condition+algorithm combos):

- **Small sweep** (`domain_stress_small.txt`): 13 commands, seed 0 only. ✅ Complete.
  - Covers E0–E3 with representative subsets (erm, groupdro, iro, inftask; no irm).
  - Test environments: explicit `0.1,0.5,0.9`.

- **Full sweep** (`domain_stress.txt`): 600 commands, seeds 0–9.
  - Covers E0–E3 with all algorithms (erm, irm, groupdro, iro, inftask).
  - Test environments: determined by train-env defaults.

### Stress-Grid Interpretation

The current generated `domain_stress` sweep is functional but has caveats:

- The generated Phase 1 train-environment sets use the repo's predefined sets (e.g., `[0.01, 0.12, 0.5, 0.99]` for 4 domains), not the cleaner comparison sets proposed in `EXPERIMENT_PLANS.md`.
- The generated Phase 3 imbalance sweep now supports balanced, last-domain-heavy, and first-domain-heavy schedules.
- Final write-ups should explicitly document which training environment each heavier schedule is intended to represent.

### Lambda Evaluation

The dedicated λ-grid evaluation script is ready but not yet validated on actual checkpoint outputs.

## Planned Next Tasks

### Priority 1: Execute Full Lambda-Grid Evaluation

Run `CMNIST/evaluate_lambda_grid.py` on saved `iro` and `inftask` checkpoints from the completed reduced sweep.

Command pattern:
```bash
cd CMNIST
..\dgil_env\Scripts\python.exe evaluate_lambda_grid.py \
  --ckpt_dirs ../cmnist_exp_small/ckpts \
  --output_dir ../cmnist_exp_small/lambda_results \
  --lambda_grid 0.0 0.1 0.2 ... 0.9
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

- Estimated runtime: 6–12 days on single GPU.
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