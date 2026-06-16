# Experiment Extension Plans

This document outlines implementation plans for three follow-up experiments in this repository:

1. A CMNIST stress test that varies the number of training domains and the amount of data per domain.
2. A $\lambda$-sensitivity analysis for imprecise learning.
3. An extension to one or more additional real-world datasets with unseen test domains.

The plans below are grounded in the current codebase structure and focus on the smallest set of changes needed to produce a credible empirical extension.

## Repository Entry Points

The most relevant existing files are:

- `CMNIST/train_sandbox.py`: main training and evaluation script for ColoredMNIST.
- `CMNIST/datasets.py`: creates CMNIST environments.
- `CMNIST/algorithms.py`: contains `IRO`, `Inftask`, and the baseline algorithms.
- `CMNIST/lib/misc.py`: evaluation helpers, including CVaR sweeps over multiple $\alpha$ values.
- `UCI-Bike-Rental/data.py`: builds environments for the Bike Sharing dataset.
- `UCI-Bike-Rental/iro.py`: real-data imprecise learning utilities.
- `UCI-Bike-Rental/final.ipynb`: current real-data experiment notebook.
- `sim/compare_without_assumption/without_assumption_lambda2.ipynb`: related simulation analysis for $\lambda$ behavior.

## Experiment 1: CMNIST Stress Test

### Goal

Test when the imprecise learning method remains competitive, and when it begins to fail, under more difficult training-domain conditions.

### Main Variants

Evaluate one or more of the following:

- Number of training domains.
- Domain imbalance.
- Number of samples per domain.

### Why This Is Feasible

This is the most direct extension in the repo.

- `CMNIST/train_sandbox.py` already accepts `--train_envs` as a variable-length list.
- `CMNIST/datasets.py` already uses `len(train_envs)` to partition the dataset into training environments.
- The current CMNIST pipeline already supports baseline comparisons and held-out test environments.

Only domain imbalance and explicit per-domain sample control require small code changes.

### Implementation Plan

#### Phase 1: Training-Domain Count Sweep

Add a sweep over the number of training domains without changing algorithm code.

- Reuse `--train_envs` with lists of different lengths, for example:
  - 2 domains
  - 4 domains
  - 6 domains
  - 8 domains
- Keep test environments fixed for comparability.
- Run the same baselines currently used in the CMNIST setup, plus `iro` and `inftask`.

Suggested implementation:

- Update `CMNIST/job_scripts/gen_exps.py` to generate experiment grids for several `train_envs` configurations.
- Save these under a new experiment name, for example `domain_stress`.

#### Phase 2: Samples-Per-Domain Control

Add optional controls for how many examples each training domain receives.

Suggested new arguments in `CMNIST/train_sandbox.py`:

- `--train_env_sizes`
- `--train_env_size_mode`

Suggested behavior:

- If `train_env_sizes` is provided, subsample each training domain after dataset creation.
- If not provided, preserve current behavior.

Suggested code change:

- Add a helper in `CMNIST/datasets.py` or locally in `CMNIST/train_sandbox.py` that truncates or randomly subsamples each environment tensor pair to the requested size.

#### Phase 3: Domain Imbalance Sweep

Construct imbalanced training sets where some environments are much larger than others.

Example schedules:

- Balanced: `[1, 1, 1, 1]`
- Mild imbalance: `[1, 1, 1, 4]`
- Strong imbalance: `[1, 1, 1, 10]`

Suggested implementation:

- Express imbalance via `train_env_sizes`, derived from a common base size and a multiplier pattern.
- Keep the total sample budget either fixed or explicitly report when it changes.

### Metrics and Analysis

Report:

- Per-test-environment accuracy and loss.
- CVaR across test environments using the existing `misc.cvar(...)` helper.
- Regret relative to the best fixed $\alpha$ expert when appropriate.

Recommended plots:

- Test performance versus number of training domains.
- Test CVaR versus imbalance ratio.
- Regret versus minimum samples per domain.

### Minimal File Changes

- `CMNIST/train_sandbox.py`
- `CMNIST/datasets.py`
- `CMNIST/job_scripts/gen_exps.py`
- Optional: `CMNIST/collect_results.py` if grouped summaries are needed.

### Expected Contribution

This extension can show whether imprecise learning is robust to limited domain coverage and skewed training support. That is a concrete and credible follow-up to the original benchmark.

## Experiment 2: $\lambda$-Sensitivity Analysis

### Goal

Measure how sensitive the method is when the operator's risk preference is uncertain or misspecified.

### Important Codebase Constraint

This repo does not train `IRO` with one fixed $\lambda$.

- `Inftask` samples $\alpha \sim \mathrm{Beta}(1, 1)` during training.
- `IRO` samples $\alpha \sim \mathrm{Beta}(a, b)` with learned Beta parameters.

That means the cleanest extension is not “retrain IRO for each fixed $\lambda$,” but rather:

- evaluate learned models across a dense range of $\lambda$ values,
- compare against fixed-$\lambda$ baselines,
- quantify robustness to operator misspecification.

### Implementation Plan

#### Phase 1: Post-Training Evaluation Sweep

Use the existing final evaluation pattern and extend it to store detailed results across a denser grid of $\alpha$ values.

Suggested grid:

- $\alpha \in \{0.0, 0.05, 0.10, \dots, 0.95\}$

Suggested implementation:

- Add a dedicated evaluation function in `CMNIST/train_sandbox.py` or `CMNIST/lib/misc.py` that:
  - computes per-environment risk at each $\alpha$,
  - computes CVaR at that $\alpha$,
  - records the selected decision rule for hypernetwork-based methods if needed.

Output format:

- JSON or CSV with one row per seed, algorithm, and $\alpha$.

#### Phase 2: Robustness-to-Misspecification Analysis

Define a mismatch between the deployment preference $\lambda_\text{true}$ and the training or decision preference $\lambda_\text{used}$.

Two useful comparisons:

- Fixed-$\alpha$ baselines evaluated at a different $\alpha$ than they were optimized for.
- `IRO` and `Inftask` evaluated across all $\alpha$ values to see whether they maintain low regret.

Suggested summary measures:

- Worst-case regret across $\alpha$.
- Average regret across $\alpha$.
- Maximum absolute performance drop between neighboring $\alpha$ values.

#### Phase 3: Baseline Comparison

Compare `IRO` against:

- ERM
- worst-case or high-$\alpha$ fixed baselines
- average-case or low-$\alpha$ fixed baselines
- `Inftask`

This turns the analysis from a descriptive sweep into a robustness claim.

### Optional Extension

If a fixed-$\lambda$ retraining experiment is desired, it is better implemented first for the simpler real-data or simulation code than by modifying CMNIST `IRO` directly. For the CMNIST codepath, fixed-$\lambda$ retraining would be a method variant rather than a pure evaluation extension.

### Minimal File Changes

- `CMNIST/train_sandbox.py`
- `CMNIST/lib/misc.py`
- Optional analysis notebook or script under `CMNIST/` or `sim/`

### Expected Contribution

This extension can support the claim that imprecise learning is useful when operator risk preference is not cleanly specified in advance, provided the analysis is framed as robustness rather than only hyperparameter tuning.

## Experiment 3: Additional Real-World Dataset Evaluation

### Goal

Test whether the method remains useful on one or more real datasets with meaningful unseen-domain shift, such as time, region, or population.

### Why This Is Feasible but Heavier

The repo already contains one real-data example based on UCI Bike Sharing.

- `UCI-Bike-Rental/data.py` creates environments using season and year.
- `UCI-Bike-Rental/final.ipynb` runs the current experiment workflow.

So the codebase already demonstrates the intended pattern, but there is no generalized reusable dataset API yet. Each additional dataset will need custom preprocessing and environment construction.

### Dataset Selection Criteria

Choose a dataset only if it provides:

- a clear domain axis,
- enough samples in each domain,
- a realistic train-test shift,
- a task where distributional robustness is meaningful.

Examples of acceptable domain definitions:

- train on earlier years, test on later years,
- train on some regions, test on held-out regions,
- train on one population mix, test on another.

### Implementation Plan

#### Phase 1: Pick One Strong Dataset

Prefer one well-motivated dataset over several weak ones.

For each candidate, check:

- target variable type: regression or classification,
- enough domain diversity,
- low leakage risk,
- accessible preprocessing pipeline.

#### Phase 2: Build an Environment Constructor

Mirror the structure of `UCI-Bike-Rental/data.py`.

Implementation steps:

- Load raw data.
- Define the domain variable or variables.
- Split into train and held-out test environments.
- Normalize features using train-domain statistics only.
- Convert to tensors and build `env_dict_train` and `env_dict_test`.

Suggested file layout:

- `NEW_DATASET/data.py`
- `NEW_DATASET/iro.py` if dataset-specific training utilities are needed
- `NEW_DATASET/final.ipynb` or a Python script for the experiments

#### Phase 3: Reproduce the Bike-Sharing Evaluation Pattern

For comparability, keep the same high-level analysis structure:

- train fixed-$\alpha$ reference models,
- train imprecise models,
- evaluate regret and CVaR over unseen domains,
- compare average-case, worst-case, and imprecise learners.

#### Phase 4: Add One Stress Variant

To make the result more than “one extra dataset,” add one controlled perturbation such as:

- fewer training domains,
- reduced training samples in minority domains,
- more severe temporal holdout.

This gives the real-data extension a sharper contribution.

### Minimal File Changes

- New dataset folder modeled after `UCI-Bike-Rental/`
- Optional shared utility extraction if duplicated code becomes large

### Expected Contribution

This can be a meaningful extension if the new dataset exposes a clear and realistic unseen-domain problem. By itself, one extra dataset is usually weaker than the CMNIST stress test unless the domain shift is especially compelling.

## Recommended Execution Order

If the goal is to maximize contribution while minimizing engineering overhead, implement the experiments in this order:

1. CMNIST stress test.
2. $\lambda$-sensitivity analysis.
3. Additional real-world dataset.

This order follows the current codebase maturity:

- CMNIST already has the most reusable infrastructure.
- $\lambda$ analysis mainly needs evaluation and reporting extensions.
- New real-world datasets require the most custom code.

## Deliverables Checklist

For each experiment, the recommended final deliverables are:

- a reproducible command or notebook entry point,
- a saved result table over multiple seeds,
- one concise figure showing the main trend,
- one short written interpretation of when `iro` helps and when it degrades.

## Summary

The strongest low-friction extension in this repository is a CMNIST stress test on domain count and imbalance. The $\lambda$ study is also well-supported if framed as robustness to risk-preference uncertainty. The real-world dataset extension is feasible, but it should ideally include a strong domain-shift story and at least one additional stress condition to make the contribution substantial.