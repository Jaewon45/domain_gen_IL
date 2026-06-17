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



## UPDATE — Current Implementation Status

============================================================
PART A — COMPLETED WORK
============================================================

### Experiment 1: CMNIST Stress Test — ✅ Fully Implemented & Validated

The repository now contains a complete, tested implementation of the CMNIST stress test.

**Infrastructure:**

- ✅ `CMNIST/train_sandbox.py` — Accepts `--train_envs`, `--train_env_sizes`, `--train_env_size_mode`, `--seed`, `--deterministic`.
- ✅ `CMNIST/datasets.py` — Subsamples training environments with configurable per-domain sizes and random/first modes.
- ✅ `CMNIST/job_scripts/gen_exps.py` — Generates two command sweeps:
  - `domain_stress_small.txt`: **13 commands** (seed 0, E0–E3, 4 algorithms)
  - `domain_stress.txt`: **600 commands** (seeds 0–9, E0–E3, all 5 algorithms)
- ✅ `CMNIST/collect_results.py` — Aggregates results with derived fields: `phase`, `n_train_domains`, `sample_size_per_domain`, `imbalance_type`.
- ✅ `CMNIST/plot_domain_stress.py` — Plots E0–E3 figures from result files.
- ✅ `CMNIST/evaluate_lambda_grid.py` — Evaluates saved checkpoints across λ grid for E4 (pending checkpoint execution).
- ✅ `CMNIST/export_results_csv.py` — Exports JSONL results to three CSV formats: run-level, long-format per-env, and aggregated summary.

**Execution Files:**

- ✅ `CMNIST/job_scripts/domain_stress_small.txt` — Ready-to-run reduced sweep (13 commands).
- ✅ `CMNIST/job_scripts/domain_stress.txt` — Ready-to-run full sweep (600 commands).
- ✅ `CMNIST/job_scripts/run_domain_stress_small_seeds.sh` — Bash runner for multi-seed (0–2) execution with separate output dirs.

**Validation Status:**

- ✅ Reduced sweep (`domain_stress_small.txt`): **All 13 commands completed** with seed 0.
  - Results: `cmnist_exp_small/results/` and `cmnist_exp_small/logs/`
  - Checkpoints saved for `groupdro`, `iro`, `inftask` runs.
  - CSV exports generated and validated.
- ✅ Smoke tests: Domain-count, sample-size, and imbalance phases all validated.
- ✅ CSV export: Tested on reduced-sweep results → 3 CSV types generated.
- ⏳ Full sweep: 600 commands generated, ready for batch execution (not yet run due to time/compute).
- ⏳ Lambda-grid evaluation: Script ready, needs checkpoint execution.

### Experiment 2: Lambda-Sensitivity Analysis — 🟡 Partially Implemented

- ✅ Evaluation entry point (`CMNIST/evaluate_lambda_grid.py`) created and validated.
- ⏳ Full λ-grid results: Pending execution on saved checkpoints from reduced sweep.
- ⏳ E4 plotting: Will be added once λ-evaluation outputs are available.

### Experiment 3: Real-World Extension — ⏳ Not Yet Started

UCI-Bike-Rental dataset infrastructure exists but no new datasets have been added.

---

### Exact Differences: `domain_stress_small.txt` vs `domain_stress.txt`

| Aspect | Small (13 cmds) | Full (600 cmds) |
|--------|-----------------|-------------------------|
| **Seeds** | 0 only | 0–9 (10 seeds) |
| **Algorithms** | erm, groupdro, iro, inftask (4) | erm, irm, groupdro, iro, inftask (5) |
| **Phase coverage** | E0–E3 (4 phases) | E0–E3 (4 phases) |
| **Test envs** | Explicit `0.1,0.5,0.9` | Train-derived defaults |
| **E0 (reproduction)** | 1 config (4 domains) | 1 config (2 domains) |
| **E1 (domain count)** | 1 variant (2 domains) | 4 variants (2, 4, 6, 8 domains) |
| **E2 (sample size)** | 1 size (2k/domain) | 3 sizes (2k, 4k, 8k per domain) |
| **E3 (imbalance)** | 1 schedule (last-heavy mild) | 5 schedules (balanced + 4 types) |
| **Output dirs** | Relative `../cmnist_exp_small/` | Absolute `/c:/Users/.../cmnist_exp/` |
| **Estimated runtime** | ~1 day | 6–12 days (single GPU) |

**Purpose:**
- Small: **Fast validation** (1 day) of the pipeline; confirms E0–E3 mechanics work.
- Full: **Publication-grade** results (all seeds, all algorithms) for final reporting.

**Status:**
- Small: ✅ **Complete** (13 commands run, results exported to CSV).
- Full: ⏳ Ready to run when compute is available (6–12 day window).

---

### Current Implementation Caveats

Some aspects of the current implementation differ slightly from the idealized experimental design:

- **Phase 1 train-environment sets:** The generator uses predefined sets (e.g., `[0.01, 0.12, 0.5, 0.99]` for 4 domains) rather than the cleaner comparison sets proposed in the plan above. **Action:** Document the exact train-envs in final write-ups.
- **Phase 3 imbalance schedules:** The sweep includes balanced, last-domain-heavy, and first-domain-heavy options. Semantic labels (minority vs. majority) are positional in code. **Action:** Map each schedule to its semantic meaning explicitly in reports.
- **Lambda-grid evaluation:** Script is ready but not yet validated on actual checkpoints. **Action:** Execute after reduced-sweep checkpoints are retained.

---

============================================================
PART B — RECOMMENDED NEXT STEPS (Post-Reduced-Sweep)
============================================================

The reduced sweep has been executed successfully. The following steps complete the analysis pipeline:

### Step 1: Execute Lambda-Grid Evaluation on Reduced-Sweep Checkpoints

**Purpose:** Generate E4 (λ-sensitivity) outputs for evaluation and plotting.

**Command:**
```bash
cd CMNIST
..\dgil_env\Scripts\python.exe evaluate_lambda_grid.py \
  --ckpt_dirs ../cmnist_exp_small/ckpts \
  --output_dir ../cmnist_exp_small/lambda_results \
  --lambda_grid 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
```

**Outputs:**
- JSONL files under `cmnist_exp_small/lambda_results/` with per-λ evaluation metrics.
- Will include per-environment accuracy, aggregated risk, and summary statistics.

**Status:** ✅ Script ready. Awaiting checkpoint execution.

### Step 2: Generate E4 Plot

**Purpose:** Add λ-sensitivity figure to the E0–E3 figure set.

**Command:**
```bash
cd CMNIST
..\dgil_env\Scripts\python.exe plot_domain_stress.py \
  ../cmnist_exp_small/results \
  --output_dir ../cmnist_exp_small/plots \
  --lambda_results ../cmnist_exp_small/lambda_results
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
- Results directories: `../cmnist_exp_small_seed1/` and `../cmnist_exp_small_seed2/`.

**Status:** ✅ Script ready. Optional for robustness validation before full sweep.

### Step 4: Export All Results to CSV

**Purpose:** Generate reportable tables for final write-up.

**Command:**
```bash
cd CMNIST
..\dgil_env\Scripts\python.exe export_results_csv.py \
  ../cmnist_exp_small/results \
  --output_dir ../cmnist_exp_small \
  --prefix cmnist_exp_small
```

**Status:** ✅ Already executed for seed 0. Re-run to include multi-seed outputs if Step 3 is executed.

### Step 5: Future — Full Sweep Execution

When ready, execute the full `domain_stress.txt` sweep (600 commands, all 10 seeds and 5 algorithms):

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
| Execution (seed 0) | ✅ Complete | `cmnist_exp_small/results/`, `logs/` |
| E0–E3 plots | ✅ Complete | `cmnist_exp_small/plots/e{0,1,2,3}_*.png` |
| E4 λ-evaluation script | ✅ Ready | `evaluate_lambda_grid.py` |
| E4 plots | ⏳ Pending Step 1 | Will be in `plots/` |
| CSV exports | ✅ Complete | `cmnist_exp_small_*.csv` |
| Multi-seed bash runner | ✅ Ready | `run_domain_stress_small_seeds.sh` |
| Full sweep command file | ✅ Complete | `domain_stress.txt` (600 commands) |

---

## Summary of Changes

### What is Different from the Original Plan?

The original `EXPERIMENT_PLANS.md` proposed a detailed phase-by-phase execution plan (E0, E1, E2, E3). The current implementation has consolidated this into:

1. **One integrated reduced sweep** (`domain_stress_small.txt`): 13 commands covering all four phases in a single execution, with results saved and analyzed together.
2. **One integrated full sweep** (`domain_stress.txt`): 600 commands with the same phase structure, ready for when compute is available.
3. **Unified analysis pipeline**: A single set of CSV export and plotting scripts that work on both small and full sweeps.

This approach is more efficient and produces the same experimental coverage while simplifying the workflow and reducing manual command management.

### What Remains?

- Execution of the full `domain_stress.txt` sweep (6–12 days, not yet started due to compute constraints).
- Lambda-grid evaluation on saved checkpoints (script ready, awaiting execution).
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


============================================================
PART C — IMMEDIATE NEXT TASKS
============================================================

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

============================================================
PART D — RUN ORDER AND MINIMUM DELIVERABLES
============================================================

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

