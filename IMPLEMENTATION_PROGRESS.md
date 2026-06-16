# Implementation Progress

This document records the implementation work completed for Experiment 1, the CMNIST stress test, and the validation results collected during setup.

## Scope Completed

The following parts of Experiment 1 are now implemented in the repository:

- Phase 1: varying the number of training domains.
- Phase 2: controlling the number of samples per training domain.
- Phase 3: generating imbalance schedules through per-domain sample sizes.

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
  - mild: `2000,2000,2000,8000`
  - strong: `2000,2000,2000,12000`

Algorithms currently included in the generated sweep:

- `erm`
- `irm`
- `groupdro`
- `iro`
- `inftask`

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
{'torch_version': '2.4.1+cpu', 'cuda_available': False, 'cuda_device_count': 0}
```

Interpretation:

- The environment is functional.
- The installed PyTorch build is CPU-only.
- No CUDA device is available to PyTorch in the current environment.

Current status:

- Basic training is feasible on CPU.
- GPU execution is not currently feasible without reinstalling a CUDA-enabled PyTorch build that matches the local GPU and driver stack.

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
- Total commands generated: `500`

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

## Files Added or Updated

Updated:

- `CMNIST/datasets.py`
- `CMNIST/train_sandbox.py`
- `CMNIST/job_scripts/gen_exps.py`
- `CMNIST/requirements.txt`

Added:

- `EXPERIMENT_PLANS.md`
- `IMPLEMENTATION_PROGRESS.md`

## Current Limitations

### GPU

The current environment does not have CUDA-enabled PyTorch, so all verified runs are CPU-only.

### Sweep Scale

The `domain_stress` generator currently emits 500 commands. This is suitable for batch execution, but it is likely too large for a single local interactive run without narrowing seeds or algorithms.

### Result Aggregation

The implementation supports running the experiments, but analysis and summary reporting for the new stress-test grid may still need a small follow-up update in `CMNIST/collect_results.py` depending on how the final tables should be grouped.

## Recommended Next Steps

1. Run a small subset of `CMNIST/job_scripts/domain_stress.txt` first, for example one seed and two algorithms.
2. If GPU execution is required, reinstall PyTorch with a CUDA-enabled build in `dgil_env`.
3. Add result-grouping fields or plotting scripts for the new stress dimensions if the analysis output needs to be publication-ready.