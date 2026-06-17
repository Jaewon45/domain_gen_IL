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

- `C:\Users\320257223\PycharmProjects\domain_gen_IL\dgil_env\Scripts\python.exe`

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
  --data_dir c:/Users/320257223/PycharmProjects/domain_gen_IL/data \
  --output_dir c:/Users/320257223/PycharmProjects/domain_gen_IL/cmnist_exp \
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
- `CMNIST/evaluate_lambda_grid.py`
- `CMNIST/plot_domain_stress.py`
- `EXPERIMENT_PLANS.md`
- `IMPLEMENTATION_PROGRESS.md`

## Current Limitations

### GPU

The current environment now has CUDA-enabled PyTorch available. Earlier smoke validations were CPU-only, but future runs can use the detected NVIDIA GPU.

### Sweep Scale

The `domain_stress` generator currently emits 600 commands. This is suitable for batch execution, but it is likely too large for a single local interactive run without narrowing seeds or algorithms.

### Result Aggregation

The stress-test aggregation path now exists, but it should still be exercised on larger multi-seed outputs beyond the current smoke validations.

### Stress-Grid Interpretation

The current generated `domain_stress` sweep is useful, but it does not yet fully match the cleaner final experimental design described in `EXPERIMENT_PLANS.md`.

Current caveats:

- The generated Phase 1 train-environment sets are the repo's current predefined sets, not the later cleaner comparison sets proposed for final reporting.
- The generated Phase 3 imbalance sweep now supports balanced, last-domain-heavy, and first-domain-heavy schedules.
- The current labels are still positional rather than semantic, so any final write-up should explicitly document which training environment each heavier schedule is intended to represent.

### Lambda Evaluation

The dedicated λ-grid evaluation script now exists, but the repository has not yet recorded a full validated λ-evaluation run on saved CMNIST checkpoints.

## Planned Next Tasks

The main remaining work is now follow-through validation on trained checkpoints plus optional generator cleanup.

### Priority 1: Finish the Reduced Sweep

Run the new reduced command file and retain saved checkpoints for the λ-analysis path.

Reason:

- The implementation surfaces now exist, but the reduced sweep still needs to complete end to end under the updated workflow.

### Priority 2: Run a Full Lambda Evaluation Pass

Execute `CMNIST/evaluate_lambda_grid.py` on saved `iro` and `inftask` checkpoints from the reduced sweep.

Reason:

- This will convert the new λ-evaluation script from an implemented entry point into a validated experiment artifact.

### Priority 3: Add the E4 Plot

Run `CMNIST/plot_domain_stress.py` with λ-evaluation outputs so the E4 aggregated-risk figure is produced alongside the existing E0-E3 plots.

Reason:

- E0-E3 plotting is now validated; E4 still depends on actual λ-evaluation outputs.

### Priority 4: Optional Generator Cleanup

After the reduced runs and aggregation pipeline work, consider updating `CMNIST/job_scripts/gen_exps.py` to support:

- clearer phase labels in generated commands or saved metadata,
- an explicit small-sweep mode.

Reason:

- This is useful for final interpretability, but it is lower priority than getting the reduced analysis pipeline working end to end.

## Recommended Next Steps

1. Finish the reduced `domain_stress_small` run and retain the saved checkpoints.
2. Run `CMNIST/evaluate_lambda_grid.py` on the reduced-sweep `iro` and `inftask` checkpoints.
3. Re-run `CMNIST/plot_domain_stress.py` with λ-evaluation outputs to add the E4 figure.
4. Optionally extend `CMNIST/job_scripts/gen_exps.py` with clearer phase labels or an explicit small-sweep mode.
5. Re-run the reduced sweep on GPU if faster turnaround is needed for the remaining checkpoints and λ-analysis artifacts.