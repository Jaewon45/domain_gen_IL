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



## [CMN] UPDATE — Current Implementation Status

## PART A — COMPLETED WORK

### [CMN] Experiment 1: CMNIST Stress Test — Implemented; reduced execution validated

The repository contains the CMNIST stress-test implementation. The reduced pipeline and smoke checks have been validated; the full generated grid has not been completed.

**Infrastructure:**

- ✅ `CMNIST/train_sandbox.py` — Accepts `--train_envs`, `--train_env_sizes`, `--train_env_size_mode`, `--seed`, `--deterministic`.
- ✅ `CMNIST/datasets.py` — Subsamples training environments with configurable per-domain sizes and random/first modes.
- ✅ `CMNIST/job_scripts/gen_exps.py` — Generates two command sweeps:
  - `domain_stress_small.txt`: **13 commands** (seed 0, E0–E3, 4 algorithms)
   - `domain_stress.txt`: **450 commands** (seeds 0–9, E0–E3, all 5 algorithms)
- ✅ `CMNIST/collect_results.py` — Aggregates results with derived fields: `phase`, `n_train_domains`, `sample_size_per_domain`, `imbalance_type`.
- ✅ `CMNIST/plot_domain_stress.py` — Plots E0–E3 figures from result files.
- ✅ `CMNIST/evaluate_lambda_grid.py` — Evaluates saved checkpoints across λ grid for E4 (pending checkpoint execution).
- ✅ `CMNIST/export_results_csv.py` — Exports JSONL results to three CSV formats: run-level, long-format per-env, and aggregated summary.

**Execution Files:**

- ✅ `CMNIST/job_scripts/domain_stress_small.txt` — Ready-to-run reduced sweep (13 commands).
- ✅ `CMNIST/job_scripts/domain_stress.txt` — Generated full sweep (450 commands; not fully executed).
- ✅ `CMNIST/job_scripts/run_domain_stress_small_seeds.sh` — Bash runner for multi-seed (0–2) execution with separate output dirs.

**Validation Status:**

- ✅ Reduced sweep (`domain_stress_small.txt`): **All 13 commands completed** with seed 0.
   - Results: `results/cmnist_exp_small/results/` and `results/cmnist_exp_small/logs/`
  - Checkpoints saved for `groupdro`, `iro`, `inftask` runs.
  - CSV exports generated and validated.
- ✅ Smoke tests: Domain-count, sample-size, and imbalance phases all validated.
- ✅ CSV export: Tested on reduced-sweep results → 3 CSV types generated.
- ⏳ Full sweep: 450 commands generated; not fully executed.
- ⏳ Lambda-grid evaluation: Script ready, needs checkpoint execution.

### [CMN] Experiment 2: Lambda-Sensitivity Analysis — Partially implemented

- ✅ Evaluation entry point (`CMNIST/evaluate_lambda_grid.py`) created and validated.
- ⏳ Full λ-grid results: Pending execution on saved checkpoints from reduced sweep.
- ⏳ E4 plotting: Will be added once λ-evaluation outputs are available.

### [ImgC] Experiment 3: ImageNet-C Extension — Smoke pipeline implemented

ImageNet-C has separate loaders, models, training/evaluation scripts, fold specification, and smoke artifacts. No report-grade real-data ImageNet-C run has been completed. UCI-Bike-Rental remains an existing dataset, not part of the ImageNet-C implementation.

---

### Exact Differences: `domain_stress_small.txt` vs `domain_stress.txt`

| Aspect | Small (13 cmds) | Full (600 cmds) |
|--------|-----------------|-------------------------|
| **Seeds** | 0 only | 0–9 (10 seeds) |
| **Algorithms** | erm, groupdro, iro, inftask (4) | erm, irm, groupdro, iro, inftask (5) |
| **Phase coverage** | E0–E3 (4 phases) | E0–E3 (4 phases) |
| **Test envs** | Explicit `0.1,0.5,0.9` | Train-derived defaults |
| **E0 (reproduction)** | 1 config (4 domains) | 1 config (2 domains) |
| **E1 (domain count)** | 1 variant (2 domains) | 3 variants (2, 4, 8 domains) |
| **E2 (sample size)** | 1 size (2k/domain) | 3 sizes (2k, 4k, 8k per domain) |
| **E3 (imbalance)** | 1 schedule (last-heavy mild) | 3 schedules (balanced + 2 last-domain-heavy variants) |
| **Output dirs** | Relative `../results/cmnist_exp_small/` | Absolute `/c:/Users/.../results/cmnist_exp/` |
| **Estimated runtime** | Reduced validation | Not a completed-run measurement |

**Purpose:**
- Small: **Fast validation** (1 day) of the pipeline; confirms E0–E3 mechanics work.
- Full: **Publication-grade** results (all seeds, all algorithms) for final reporting.

**Status:**
- Small: ✅ **Complete** (13 commands run, results exported to CSV).
- Full: ⏳ Generated but not fully executed.

---

### Current Implementation Caveats

Some aspects of the current implementation differ slightly from the idealized experimental design:

- **Phase 1 train-environment sets:** The generator uses predefined sets (e.g., `[0.01, 0.12, 0.5, 0.99]` for 4 domains) rather than the cleaner comparison sets proposed in the plan above. **Action:** Document the exact train-envs in final write-ups.
- **Phase 3 imbalance schedules:** The current grid is balanced, mild last-domain-heavy, and strong last-domain-heavy. It does not establish a general minority-underrepresentation result.
- **E3b tail support:** The current `results/E3b_tail_support/` artifact contains 45 records and is missing the `near_missing_tail` condition from the 60-job, 3-seed matrix.
- **Lambda-grid evaluation:** The evaluator and checkpoints exist, but report-grade E4 results and plots are still pending.

---

## PART B — RECOMMENDED NEXT STEPS (Post-Reduced-Sweep)

The reduced sweep has been executed successfully. The following steps complete the analysis pipeline:

### Step 1: Execute Lambda-Grid Evaluation on Reduced-Sweep Checkpoints

**Purpose:** Generate E4 (λ-sensitivity) outputs for evaluation and plotting.

**Command:**
```bash
cd CMNIST
..\dgil_env\Scripts\python.exe evaluate_lambda_grid.py \
   ../results/cmnist_exp_small/ckpts \
   --output_dir ../results/cmnist_exp_small/lambda_results \
   --lambda_grid 0.0:1.0:0.1
```

**Outputs:**
- JSONL files under `results/cmnist_exp_small/lambda_results/` with per-λ evaluation metrics.
- Will include per-environment accuracy, aggregated risk, and summary statistics.

**Status:** ✅ Script ready. Awaiting checkpoint execution.

### Step 2: Generate E4 Plot

**Purpose:** Add λ-sensitivity figure to the E0–E3 figure set.

**Command:**
```bash
cd CMNIST
..\dgil_env\Scripts\python.exe plot_domain_stress.py \
   ../results/cmnist_exp_small/results \
   --output_dir ../results/cmnist_exp_small/plots \
   --lambda_results ../results/cmnist_exp_small/lambda_results
```

**Outputs:**
- Updated plot set including `e4_lambda_sensitivity.png`.

**Status:** ✅ Script updated to support λ-inputs. Awaiting Step 1 outputs.

### Step 3: Optional Multi-Seed Validation Run

**Purpose:** Confirm reproducibility and seed variability across seeds 1–2.

**Command:**
```bash
cd CMNIST/job_scripts
bash run_domain_stress_small_seeds.sh
```

**Outputs:**
- Results directories: `../results/cmnist_exp_small_seed1/` and `../results/cmnist_exp_small_seed2/`.

**Status:** ✅ Script ready. Optional for robustness validation before full sweep.

### Step 4: Export All Results to CSV

**Purpose:** Generate reportable tables for final write-up.

**Command:**
```bash
cd CMNIST
..\dgil_env\Scripts\python.exe export_results_csv.py \
   ../results/cmnist_exp_small/results \
   --output_dir ../results/export \
  --prefix cmnist_exp_small
```

**Status:** ✅ Already executed for seed 0. Re-run to include multi-seed outputs if Step 3 is executed.

### Step 5: Future — Full Sweep Execution

When ready, execute the full `domain_stress.txt` sweep (450 commands, all 10 seeds and 5 algorithms):

```bash
cd CMNIST/job_scripts
# Execute domain_stress.txt in batch mode (see submission scripts)
```

**Estimated runtime:** 6–12 days on single GPU (100–200 hours wall-clock).

---

## Deliverables Checklist — Reduced Sweep Status

| Deliverable | Status | Location |
|-------------|--------|----------|
| Reduced command file | ✅ Complete | `domain_stress_small.txt` (13 commands) |
| Execution (seed 0) | ✅ Complete | `results/cmnist_exp_small/results/`, `results/cmnist_exp_small/logs/` |
| E0–E3 plots | ✅ Complete | `results/cmnist_exp_small/plots/e{0,1,2,3}_*.png` |
| E4 λ-evaluation script | ✅ Ready | `evaluate_lambda_grid.py` |
| E4 plots | ⏳ Pending Step 1 | Will be in `plots/` |
| CSV exports | ✅ Complete | `results/export/cmnist_exp_small_*.csv` |
| Multi-seed bash runner | ✅ Ready | `run_domain_stress_small_seeds.sh` |
| Full sweep command file | ✅ Generated | `domain_stress.txt` (450 commands) |

---

## Summary of Changes

### What is Different from the Original Plan?

The original `EXPERIMENT_PLANS.md` proposed a detailed phase-by-phase execution plan (E0, E1, E2, E3). The current implementation has consolidated this into:

1. **One integrated reduced sweep** (`domain_stress_small.txt`): 13 commands covering all four phases in a single execution, with results saved and analyzed together.
2. **One integrated full sweep** (`domain_stress.txt`): 450 commands with the same phase structure, generated but not fully executed.
3. **Unified analysis pipeline**: A single set of CSV export and plotting scripts that work on both small and full sweeps.

This approach is more efficient and produces the same experimental coverage while simplifying the workflow and reducing manual command management.

### What Remains?

- Completion and cleanup of the full `domain_stress.txt` sweep.
- Lambda-grid evaluation on saved checkpoints, followed by E4 plots and robustness summaries.
- Optional multi-seed runs for robustness validation (script ready, can be deferred).

All infrastructure is in place; execution is now the limiting factor.

---

**Note:** Detailed phase-by-phase execution guides (E0, E1, E2, E3) from the original plan have been consolidated into the integrated sweep files. These are archived in git history if needed for reference.

What to run:
- Dataset: CMNIST
- Train environments: start with the repo’s default or [0.1, 0.2, 0.5, 0.9]
- Test environments: [0.0, 0.1, ..., 1.0]
- Algorithms: ERM, GroupDRO, IRO first; then INF-TASK and IRM if feasible
- Seeds: start with 1 seed, then increase to 3 if runtime allows
- Steps: use the paper/repo default if feasible; otherwise use a smaller setting and clearly label it as a reduced reproduction

What to implement:
1. A small config or command list for reproduction runs.
2. Result aggregation for:
   - algorithm,
   - seed,
   - train environments,
   - test environment,
   - accuracy,
   - risk/loss if available,
   - λ if applicable.
3. A reproduction plot:
   - x-axis: test environment e
   - y-axis: accuracy
   - one line per algorithm
4. A reproduction table:
   - average accuracy,
   - worst-domain accuracy,
   - max/approx regret if available.

Expected output:
- One plot of accuracy across test environments.
- One small table comparing algorithms.
- A short paragraph: “The reproduction is approximate/reduced, but it confirms the expected pattern...” or honestly explain if it does not.

Decision rule:
Only proceed to larger stress tests after E0 produces understandable result files and plots.


------------------------------------------------------------
E1 — Training domain-count stress test
------------------------------------------------------------

Goal:
Test whether IRO still helps when the learner observes fewer source domains.

Main question:
Does IRO need many source domains to construct a useful imprecise risk profile?

Suggested domain-count grid:
- 2 training domains
- 4 training domains
- 6 training domains
- 8 training domains

Current generator-compatible train environment sets:
- 2 domains: [0.1, 0.2]
- 4 domains: [0.01, 0.12, 0.5, 0.99]
- 6 domains: [0.01, 0.12, 0.0, 0.5, 0.7, 0.99]
- 8 domains: [0.01, 0.12, 0.0, 0.0, 0.14, 0.5, 0.7, 0.99]

Optional later comparison set:
- 2 domains: [0.1, 0.9]
- 4 domains: [0.1, 0.2, 0.5, 0.9]
- 6 domains: [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
- 8 domains: [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]

If using the current generated sweep, document the exact train environments above rather than the cleaner comparison set.

What to run:
- Algorithms first: ERM, GroupDRO, IRO
- Add later: INF-TASK, IRM
- Seeds: 1 first, then 3 if feasible
- Sample size per domain: keep fixed, e.g. 2000 or 4000, to isolate the effect of domain count

What to implement:
1. Ensure `train_envs` can be passed explicitly.
2. Ensure the number of training domains is saved in each JSONL result record.
3. Add an analysis grouping field:
   - `phase = domain_count`
   - `n_train_domains = 2/4/6/8`
4. Add a plot:
   - x-axis: number of training domains
   - y-axis: worst-domain accuracy or max regret
   - separate lines/bars per algorithm
5. Add a second plot or appendix plot:
   - accuracy across test environments for each domain-count condition.

Expected interpretation:
- If IRO improves with more domains, argue that the imprecise learner needs enough domain evidence to construct a useful risk profile.
- If IRO is robust even with few domains, that supports the method.
- If ERM or GroupDRO wins in low-domain settings, discuss the limits of imprecision under insufficient domain diversity.


------------------------------------------------------------
E2 — Per-domain sample-size stress test
------------------------------------------------------------

Goal:
Test whether IRO needs many samples within each source domain.

Main question:
Does IRO become unstable when each source domain is represented by few samples?

Suggested sample-size grid:
- 2000 samples/domain
- 4000 samples/domain
- 8000 samples/domain

Optional smoke grid:
- 128 or 512 samples/domain for quick validation only, not final results unless clearly labeled as toy-scale.

Fixed train environments:
- [0.1, 0.2, 0.5, 0.9]

What to run:
- Algorithms first: ERM, GroupDRO, IRO
- Add later: INF-TASK and IRM
- Seeds: 1 first, then 3 if feasible

What to implement:
1. Ensure `train_env_sizes` is saved in the result record.
2. Add analysis grouping fields:
   - `phase = sample_size`
   - `sample_size_per_domain = 2000/4000/8000`
3. Add a plot:
   - x-axis: samples per domain
   - y-axis: worst-domain accuracy or max regret
   - separate lines/bars per algorithm
4. Add optional runtime table:
   - samples/domain,
   - algorithm,
   - runtime,
   - final metric.

Expected interpretation:
- If IRO needs more samples than ERM/GroupDRO, discuss the cost of learning across λ.
- If IRO remains stable at low sample size, that is strong evidence of practical robustness.
- If all methods fail at low sample size, emphasize data uncertainty rather than generalisation uncertainty.


------------------------------------------------------------
E3 — Domain/sample imbalance stress test
------------------------------------------------------------

Goal:
Test what happens when risky or minority domains are underrepresented.

Main question:
Does IRO still help when the training data is dominated by one domain regime?

Fixed train environments:
- [0.1, 0.2, 0.5, 0.9]

Important implementation detail:
Clarify which environment receives extra samples. This affects the interpretation.

Recommended imbalance schedules:

A. Current generator-supported schedules:
- [2000, 2000, 2000, 2000]
- [2000, 2000, 2000, 8000]
- [2000, 2000, 2000, 12000]

B. Planned mirrored extension for majority-heavy imbalance:
- [8000, 2000, 2000, 2000]
- [12000, 2000, 2000, 2000]

C. Interpretation note:
- the currently generated imbalance schedules overweight the last listed training environment,
- if that environment is treated as the minority or opposite regime, the current sweep is not yet a true minority-underrepresentation test,
- add the mirrored majority-heavy schedules in B before making that stronger claim.

This phase should therefore be reported as a last-domain-heavy imbalance sweep unless the mirrored schedules are added.

What to run:
- Algorithms: ERM, GroupDRO, IRO, INF-TASK
- Optional: IRM
- Seeds: 1 first, then 3 if feasible

What to implement:
1. Add imbalance type labels:
   - `balanced`
   - `majority_heavy_mild`
   - `majority_heavy_strong`
   - `minority_heavy_mild`
   - `minority_heavy_strong`
2. Save the exact `train_envs` and `train_env_sizes` in every result file.
3. Add analysis grouping fields:
   - `phase = imbalance`
   - `imbalance_type`
   - `train_env_sizes`
4. Add plot:
   - x-axis: imbalance condition
   - y-axis: worst-domain accuracy or max regret
   - separate bars/lines per algorithm
5. Add plot:
   - test-environment accuracy curves for balanced vs strong imbalance.

Expected interpretation:
- If IRO fails when minority domains are underrepresented, this is an important limitation: imprecision cannot recover information that is absent or severely underweighted.
- If IRO remains robust, this is a strong positive result.
- If GroupDRO beats IRO under severe imbalance, discuss whether explicit worst-case training is better when the operator is strongly risk-averse and the minority domain is visible.


------------------------------------------------------------
E4 — λ-sensitivity analysis
------------------------------------------------------------

Goal:
Analyze how stable IRO is when the operator’s risk preference λ is unclear or difficult to specify.

Main question:
If the operator does not know the correct λ, how sensitive are the results to λ?

Use the models trained in E0–E3 where possible.

λ grid:
- λ ∈ {0.0, 0.1, 0.2, ..., 1.0}

Current status note:
The repository does not yet save a λ-grid evaluation table or `lambda_eval` records automatically. This remains planned follow-up work.

What to compute:
1. For IRO:
   - evaluate h(x, λ) across the λ grid.
   - calculate accuracy per test environment.
   - calculate CVaR/aggregated risk across test environments.

2. For INF-TASK:
   - evaluate if the augmented hypothesis supports λ-conditioned predictions.
   - compare its risk curve to IRO.

3. For ERM and GroupDRO:
   - they may not depend on λ in prediction.
   - evaluate their fixed predictions under different λ-based risk aggregation over test-domain losses.
   - This gives λ-dependent evaluation curves even if the model is not λ-conditioned.

What to implement:
1. Add or verify an evaluation mode that loops over λ values.
2. Save λ-specific results:
   - `lambda_eval`
   - `algorithm`
   - `seed`
   - `test_env`
   - `accuracy`
   - `loss`
   - `aggregated_risk`
3. Add plots:
   - λ on x-axis, aggregated risk on y-axis.
   - λ on x-axis, worst-domain accuracy or selected-domain accuracy on y-axis.
   - optional: heatmap with λ on x-axis and test environment e on y-axis.
4. Add a “λ robustness” summary:
   - best λ,
   - worst λ,
   - range of performance across λ,
   - sensitivity score = max(metric over λ) - min(metric over λ).

Expected interpretation:
- If IRO is relatively flat across λ, it is robust to operator preference uncertainty.
- If IRO strongly changes across λ, the method may require careful preference elicitation.
- If high λ improves minority/opposite domains but hurts majority domains, this supports the paper’s risk-preference interpretation.
- If λ has little meaningful effect, question whether the augmented hypothesis is actually using λ in the tested setting.


## PART C — IMMEDIATE NEXT TASKS

1. Check result schema
   Ensure each JSONL result contains or derives the fields needed for the stress analysis:
   - algorithm,
   - seed,
   - train environments,
   - train environment sizes,
   - test environments,
   - per-test-environment accuracy,
   - loss/risk if available,
   - λ-specific evaluation fields if applicable,
   - steps,
   - batch size,
   - output directory,
   - experiment-phase metadata.

2. Update result aggregation
   Add or update `CMNIST/collect_results.py` so it can group by:
   - phase,
   - algorithm,
   - seed,
   - n_train_domains,
   - sample_size_per_domain,
   - imbalance_type,
   - lambda_eval.

3. Create a small command file
   Create something like:
   - `CMNIST/job_scripts/domain_stress_small.txt`

   It should include:
   - E0 reduced reproduction,
   - E1 one seed for ERM, GroupDRO, IRO,
   - E2 one seed for ERM, GroupDRO, IRO,
   - E3 one seed for ERM, GroupDRO, IRO, INF-TASK.

4. Add λ evaluation script
   Add a script such as:
   - `CMNIST/evaluate_lambda_grid.py`
   or extend the existing evaluation script.

   It should:
   - load trained models,
   - evaluate λ grid from 0.0 to 1.0,
   - save λ-specific metrics.

5. Add plotting script
   Add:
   - `analysis/plot_domain_stress.py`
   or
   - `CMNIST/plot_domain_stress.py`

   Required figures:
   - accuracy by test environment for E0,
   - worst-domain accuracy or regret by number of domains for E1,
   - worst-domain accuracy or regret by sample size for E2,
   - worst-domain accuracy or regret by imbalance condition for E3,
   - aggregated risk over λ for E4.

6. Add a reproducibility README section
   Include:
   - environment setup,
   - how to run small subset,
   - how to run full sweep,
   - how to collect results,
   - how to generate plots.

## PART D — RUN ORDER AND MINIMUM DELIVERABLES

Run order
---------
Step 1:
Run E0 with one seed and three algorithms.

Step 2:
Run E1 with one seed and three algorithms.

Step 3:
Run E2 with one seed and three algorithms.

Step 4:
Run E3 with one seed and four algorithms.

Step 5:
Run E4 on the trained IRO/INF-TASK models.

Step 6:
Only then expand to more seeds or more algorithms.

Minimum final result set
------------------------
If time is short, final seminar can still work with:

- E0: small reproduction
- E1: domain-count sweep
- E3: imbalance sweep
- E4: λ-sensitivity on E0/E3

E2 can be shortened or moved to appendix if runtime is tight.


## UPDATE — E3b/E2 Tail-Support / Missing-Support Stress Test (Main Presentation)

This update replaces the seminar-facing E2 sample-size emphasis with a direct support-coverage test aligned with the theoretical failure set

$$
U_N=\{a\in\mathcal A:\pi_a^\star>0,\ n_a=0\}.
$$

The old E2 sample-size code remains available as backup, but the main presentation should use this tail-support experiment.

### Experiment Name

- Preferred: `E3b_tail_support`
- Alternate numbering: `E2_tail_support`

### Core Question

Does IRO remain protective on tail-domain performance when tail groups are visible, near-missing, or completely missing in source-domain evidence?

### Conditions (Fixed Source Budget Where Possible)

Use 4 source environments (head → tail ordering), fixed test environments, and these train counts:

1. `balanced_visible`: `[2000, 2000, 2000, 2000]`
2. `long_tail_visible`: `[5000, 2000, 800, 200]`
3. `near_missing_tail`: `[5800, 1800, 350, 50]`
4. `missing_tail`: `[6000, 1500, 500, 0]`

### Implementation Notes

- For `missing_tail`, zero-size source domains are dropped from training loaders while still kept in evaluation/test domains.
- Log per-domain sampled counts and empirical prior
   $$\widehat\pi_a^{(N)}=n_a/N.$$
- Missing source domains must get empirical prior mass `0` and positive deployment prior mass under uniform evaluation prior.

### Algorithms and Seeds

- Algorithms: `ERM`, `GroupDRO`, `INF-TASK`, `IRM`, `IRO`
- Use existing seed set; default practical start is `0,1,2` via configurable seed list.

### Required Outputs

Save under `results/E3b_tail_support/`:

- `raw_results.csv`
- `summary_by_condition.csv`
- `slide_table.csv`
- `tail_accuracy_by_condition.png`
- `worst_accuracy_by_condition.png`
- `head_tail_gap_by_condition.png`
- `cvar_gap_by_condition.png`
- `iro_lambda_tail_accuracy_by_condition.png`
- `iro_lambda_cvar_by_condition.png`

### Practical Run Flow

1. Generate commands via tail-support generator (`gen_exps.py`, `exp_name` containing `tail_support`).
2. Train E3b runs.
3. Run lambda-grid checkpoint evaluation.
4. Run tail-support analyzer to export slide-ready CSVs/plots.


## UPDATE — Proposed ImageNet-C Extension and Stability Analysis

This update does not replace the historical candidate sections above. It records the currently preferred extension direction for an additional dataset while preserving the earlier real-world-dataset alternatives for reference.

### Output Preservation Policy

The ImageNet-C extension must not overwrite any artifacts from existing CMNIST, E3b, lambda-grid, or prior exploratory runs.

Use fresh output roots under `results/` for every new ImageNet-C phase. Recommended roots:

- `results/imagenet_c_eval_repeatability_v1/`
- `results/imagenet_c_fold_generalization_v1/`
- `results/imagenet_c_support_stress_v1/`

If any of those roots already exist from prior exploratory work, create a new suffixed root such as `_v2` or a date-stamped variant instead of reusing the folder.

### Three Distinct Meanings of Stability

#### 1. Evaluation Repeatability

For a fixed checkpoint evaluated on fixed ImageNet-C inputs without random subsampling or test-time augmentation, repeated evaluations should be deterministic. If repeated runs differ materially, the problem is in preprocessing, checkpoint loading, evaluation mode, or metric aggregation rather than in the learning algorithm.

#### 2. Training Stability Across Seeds

Training stability asks whether the same qualitative conclusion persists across random seeds.

Final report-grade summaries should include:

- mean across seeds,
- standard deviation across seeds,
- individual seed values,
- count of seeds on which each method beats the baseline.

#### 3. Stability Across Domains and Severity

ImageNet-C contains 15 corruption types and 5 severity levels, giving 75 corruption-severity conditions. Stability should therefore be assessed across corruption domains and across severity, not only through a single aggregate metric.

Required domain-level analysis should include:

- average performance across corruption domains,
- worst-corruption performance,
- CVaR or another tail-risk metric across corruption domains,
- severity-stratified results,
- corruption-by-severity heatmap,
- number of corruption domains improved or degraded by each method.

These three notions of stability should remain conceptually separate in analysis and reporting.

### Adopted ImageNet-C Design

#### Research Question

Does the imprecise learning approach provide stable average-case and tail-domain performance under realistic image corruptions, and does its advantage remain consistent across random seeds, unseen corruption types, corruption severity, and operator risk preferences?

#### Domain Definition

Use corruption type as the primary domain variable.

- Primary domains: 15 corruption types
- Within-domain shift: severity levels 1-5
- Total evaluation conditions: 75

Primary aggregation should operate over the 15 corruption domains. Severity should be retained as a separate analysis axis rather than flattening all 75 conditions into unrelated domains.

#### Label Space

Retain the native 1000-class ImageNet classification task. Do not apply CMNIST-style binary relabeling to ImageNet-C.

#### Training / Evaluation Separation

Official ImageNet-C validation data is evaluation-only.

If training on corrupted inputs is introduced, corruptions must be generated from ImageNet training images rather than from the official ImageNet-C validation set. This avoids evaluation leakage.

#### Model Architecture

Use the following initial architecture for feasibility and variance control:

- pretrained ResNet-50 backbone,
- frozen backbone,
- extracted 2048-dimensional feature representation,
- trainable 1000-class classification head,
- optional lambda-conditioned head for IRO-style evaluation.

All compared algorithms should use the same frozen features and comparable head capacity.

#### Primary Algorithms

Main comparison set:

1. `ERM`
2. `GroupDRO`
3. `IRO`

`INF-TASK` may be added after the main pipeline is validated. `IRM` remains an optional secondary baseline rather than part of the initial core study.

#### Lambda Evaluation

Do not retrain IRO separately for every fixed lambda at the start. Reuse the repository's post-training lambda-grid evaluation pattern.

Use:

$$
\lambda \in \{0.0, 0.1, 0.2, \ldots, 1.0\}.
$$

For each lambda, save:

- per-corruption accuracy,
- per-corruption loss,
- average accuracy,
- worst-corruption accuracy,
- aggregated risk / CVaR,
- lambda sensitivity score (best-worst range over the grid).

#### Main Metrics

Primary metrics:

- clean ImageNet accuracy,
- mean corruption accuracy,
- worst-corruption accuracy,
- mean corruption loss,
- worst-corruption loss,
- CVaR across corruption domains,
- performance by severity,
- lambda sensitivity,
- mean and standard deviation across training seeds.

Mean Corruption Error may be included as a standard ImageNet-C secondary metric, but the core DGIL comparison should retain average-domain, worst-domain, and CVaR-style summaries.

### Experiment Set 1 — Deterministic Evaluation and Pipeline Validation

Purpose: verify that evaluation and aggregation are deterministic before introducing training variance.

Configuration:

- one fixed pretrained ResNet-50 checkpoint,
- no training,
- clean ImageNet validation set,
- all 15 ImageNet-C corruptions,
- all 5 severity levels,
- 3 repeated evaluation passes,
- no random subsampling,
- no test-time augmentation,
- lambda grid `0.0:1.0:0.1`.

Required outputs per repetition:

- accuracy and loss for each of the 75 conditions,
- corruption-level aggregation over severities,
- average corruption accuracy,
- worst-corruption accuracy,
- CVaR / aggregated risk,
- corruption-by-severity heatmap.

Stability criterion: all three repetitions should match up to negligible floating-point differences.

### Experiment Set 2 — Held-Out Corruption Generalization

Purpose: test whether IRO generalizes to corruption types not seen during training.

Use 3 fixed folds, each training on 10 corruption types and holding out 5 corruption types for primary testing.

Held-out Fold A:

- gaussian noise
- defocus blur
- glass blur
- snow
- contrast

Held-out Fold B:

- shot noise
- motion blur
- frost
- fog
- elastic transformation

Held-out Fold C:

- impulse noise
- zoom blur
- brightness
- pixelation
- JPEG compression

Training data policy per fold:

- fixed class-balanced subset of 100 ImageNet training images per class,
- 80 images per class for training,
- 20 images per class for validation,
- source corruptions generated only from the 10 visible corruption types,
- training severity sampled uniformly from levels 1-3,
- identical base-image subset and corruption assignment for all algorithms.

Evaluation:

- primary: 5 held-out corruption types at severities 1-5,
- secondary: all 15 corruption types at severities 1-5,
- clean ImageNet validation accuracy.

Training settings:

- frozen ResNet-50 backbone,
- trainable 1000-class head,
- batch size 256,
- max epochs 20,
- early stopping patience 3,
- optimizer `AdamW`,
- learning-rate grid `{3e-4, 1e-3}`,
- weight-decay grid `{0, 1e-4}`,
- hyperparameters selected using Fold A, seed 0, then locked.

Pilot execution:

$$
3\text{ folds} \times 3\text{ algorithms} \times 3\text{ seeds} = 27\text{ runs}.
$$

Final execution:

$$
3\text{ folds} \times 3\text{ algorithms} \times 5\text{ seeds} = 45\text{ runs}.
$$

Use seeds `0-2` for the pilot and `0-4` for the report-grade run.

Success criterion:

- same qualitative ranking in at least 2 of 3 folds,
- conclusion not driven by a single seed,
- IRO improves worst-domain or CVaR performance without excessive mean-accuracy loss,
- severity trends remain interpretable,
- lambda sensitivity is not dominated by isolated corruption domains.

### Experiment Set 3 — Corruption-Support Stress Test

Purpose: mirror the CMNIST tail-support experiment in a realistic corruption setting.

Use 4 representative source corruption domains:

1. gaussian noise
2. motion blur
3. fog
4. JPEG compression

Use severity levels 1-3 during training and a fixed total source budget of 40,000 corrupted training examples.

Conditions:

1. Balanced visible: `[10000, 10000, 10000, 10000]`
2. Long-tail visible: `[25000, 10000, 4000, 1000]`
3. Near-missing tail: `[29000, 9000, 1750, 250]`
4. Missing tail: `[30000, 7500, 2500, 0]`

For the missing-tail condition, the fourth source corruption gets zero training examples but remains present in evaluation with positive evaluation weight.

Evaluation:

- same 4 source corruptions at severities 4-5,
- remaining 11 unseen corruption types at severities 1-5,
- clean ImageNet validation set.

Report:

- tail-domain accuracy,
- worst-domain accuracy,
- head-tail performance gap,
- mean corruption accuracy,
- CVaR gap relative to balanced condition,
- IRO lambda curves for tail accuracy and aggregated risk.

Pilot execution:

$$
4\text{ conditions} \times 3\text{ algorithms} \times 3\text{ seeds} = 36\text{ runs}.
$$

Final execution:

$$
4\text{ conditions} \times 3\text{ algorithms} \times 5\text{ seeds} = 60\text{ runs}.
$$

### Recommended Execution Order For ImageNet-C

1. Run Experiment Set 1 and verify deterministic evaluation.
2. Run one fold, three algorithms, and one seed from Experiment Set 2 as a smoke test.
3. Run the full three-seed pilot for Experiment Set 2.
4. Inspect seed variance and corruption-level results.
5. Expand Experiment Set 2 to five seeds only if results remain interpretable.
6. Run Experiment Set 3 only after the held-out-corruption pipeline is validated.
7. Add `INF-TASK` only after the primary `ERM` / `GroupDRO` / `IRO` comparison is complete.

Minimum defensible extension: Experiment Sets 1 and 2. Experiment Set 3 is a stronger support-coverage follow-up, not the entry point.

### Immediate Next-Step Implementation Surface

To move from planning to implementation without disturbing existing experiments, reserve a separate dataset surface:

- `IMAGENET_C/datasets.py`
- `IMAGENET_C/features.py`
- `IMAGENET_C/models.py`
- `IMAGENET_C/train_head.py`
- `IMAGENET_C/eval_repeatability.py`
- `IMAGENET_C/evaluate_lambda_grid.py`
- `IMAGENET_C/analyze_imagenet_c.py`
- `IMAGENET_C/folds.json`
- `IMAGENET_C/job_scripts/`

This is intentionally separate from `CMNIST/` so the ImageNet-C extension does not inherit CMNIST-specific data generation assumptions.

### Concrete Output Layout For The First Two Experiment Sets

Experiment Set 1 root:

- `results/imagenet_c_eval_repeatability_v1/raw/repetition_0.jsonl`
- `results/imagenet_c_eval_repeatability_v1/raw/repetition_1.jsonl`
- `results/imagenet_c_eval_repeatability_v1/raw/repetition_2.jsonl`
- `results/imagenet_c_eval_repeatability_v1/summary/repeatability_summary.csv`
- `results/imagenet_c_eval_repeatability_v1/summary/corruption_severity_matrix.csv`
- `results/imagenet_c_eval_repeatability_v1/plots/corruption_severity_heatmap.png`
- `results/imagenet_c_eval_repeatability_v1/plots/corruption_accuracy_bar.png`
- `results/imagenet_c_eval_repeatability_v1/plots/lambda_sensitivity_curve.png`
- `results/imagenet_c_eval_repeatability_v1/manifests/eval_repeatability_repetitions.txt`

Experiment Set 2 root:

- `results/imagenet_c_fold_generalization_v1/raw/train_runs.jsonl`
- `results/imagenet_c_fold_generalization_v1/raw/lambda_eval.jsonl`
- `results/imagenet_c_fold_generalization_v1/summary/by_fold_seed.csv`
- `results/imagenet_c_fold_generalization_v1/summary/by_fold_algorithm.csv`
- `results/imagenet_c_fold_generalization_v1/summary/held_out_only.csv`
- `results/imagenet_c_fold_generalization_v1/summary/all_corruptions.csv`
- `results/imagenet_c_fold_generalization_v1/plots/held_out_worst_domain.png`
- `results/imagenet_c_fold_generalization_v1/plots/held_out_mean_accuracy.png`
- `results/imagenet_c_fold_generalization_v1/plots/severity_profile_by_algorithm.png`
- `results/imagenet_c_fold_generalization_v1/manifests/fold_a_seed0_smoke.txt`
- `results/imagenet_c_fold_generalization_v1/manifests/pilot_seed012.txt`
- `results/imagenet_c_fold_generalization_v1/manifests/final_seed01234.txt`

### Reproducible Fold Specification Requirement

Before any training script is written, the 3 held-out corruption folds should be encoded in a versioned spec file rather than being hardcoded inconsistently across scripts.

Required folds:

- `fold_a`: `gaussian_noise`, `defocus_blur`, `glass_blur`, `snow`, `contrast`
- `fold_b`: `shot_noise`, `motion_blur`, `frost`, `fog`, `elastic_transform`
- `fold_c`: `impulse_noise`, `zoom_blur`, `brightness`, `pixelate`, `jpeg_compression`

### First Command Manifests To Generate

Generate these manifests first once the scripts exist:

1. `eval_repeatability_repetitions.txt`
2. `fold_a_seed0_smoke.txt`
3. `pilot_seed012.txt`

Only after those are validated should the final `seed0-4` held-out-generalization manifest and any support-stress manifest be generated.

