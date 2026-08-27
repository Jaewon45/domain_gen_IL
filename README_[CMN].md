# [CMN] CMNIST Experiment Results

This README describes only the Colored MNIST ([CMN]) experiments in this repository. ImageNet-C is documented separately in `IMAGENET_C/README_[ImgC].md`.

## Scope

CMNIST is a binary classification benchmark with environment-specific color-label correlation. The target is created by mapping digits `0-4` to `1` and digits `5-9` to `0`, then applying 25% label noise. The environment parameter `e` controls color flips. Training uses source environments and evaluates on held-out environment values.

The CMNIST implementation is separate from ImageNet-C:

- Dataset construction: `CMNIST/datasets.py`
- Training and evaluation: `CMNIST/train_sandbox.py`
- Algorithms: `CMNIST/algorithms.py`
- Result aggregation: `CMNIST/collect_results.py`
- Stress plots: `CMNIST/plot_domain_stress.py`
- Tail-support analysis: `CMNIST/analyze_tail_support.py`
- Lambda evaluation: `CMNIST/evaluate_lambda_grid.py`

## Result Status

| Result group | Current status | Main location |
| --- | --- | --- |
| Smoke checks | Completed | `cmnist_exp_smoke/` |
| Reduced CMNIST sweep | Completed for the 13-command seed-0 subset | `results/cmnist_exp_small/` when present |
| Staged main sweep | Partially executed; current artifacts include seeds 0-2 and recovery/duplicate records | `results/cmnist_exp/` |
| Full stress grid | Manifest generated, not fully executed | `CMNIST/job_scripts/domain_stress.txt` |
| E3b tail-support analysis | 45 result records; the `near_missing_tail` condition is missing from the current result root | `results/E3b_tail_support/` |
| Lambda sensitivity | Evaluation code and lambda artifacts exist; treat final scientific conclusions as pending inspection and cleanup | `results/cmnist_exp/lambda_results/` |

The files in `results/cmnist_exp/` should not be interpreted as a clean completed 450-job study. The root currently contains 553 JSONL result rows, including evaluation records from partially executed and recovered runs. Deduplicate and inspect logs before reporting seed-level statistics.

## Experiments

### E0: Reduced Reproduction

Purpose: verify that the CMNIST training, evaluation, result-writing, and plotting pipeline works on a small representative configuration.

The reduced command file is `CMNIST/job_scripts/domain_stress_small.txt`. It contains 13 commands for seed 0 and covers E0-E3 with the first-pass algorithm set:

- ERM
- GroupDRO
- IRO
- INF-TASK

The reduced run uses explicit test environments `0.1,0.5,0.9` and is pipeline-validation evidence, not a publication-grade multi-seed comparison.

Expected E0 analysis is accuracy by test environment. The current `results/cmnist_exp/plots/` directory contains stress plots, but an E0 accuracy plot is not currently present there; regenerate it from the intended clean result subset before using it in a report.

### E1: Training-Domain Count

Purpose: test whether performance changes when the learner observes fewer or more source domains.

The generated full-grid conditions are:

| Number of source domains | Training environments |
| ---: | --- |
| 2 | `0.1,0.2` |
| 4 | `0.01,0.12,0.5,0.99` |
| 8 | `0.01,0.12,0.0,0.0,0.14,0.5,0.7,0.99` |

The current implementation does not include the originally proposed 6-domain condition. The exact environment lists in the generated commands are the reproducibility source of truth.

The main derived metrics are average-domain accuracy, worst-domain accuracy, and best-domain accuracy across the test environments. Result records also retain the algorithm, seed, training environments, and evaluation metrics.

### E2: Balanced Sample-Size Stress

Purpose: test whether performance changes when every source environment has the same number of training examples.

The generated conditions are:

- `2000,2000,2000,2000`
- `4000,4000,4000,4000`
- `8000,8000,8000,8000`

The `train_env_size_mode` is `random` by default. Use `first` only when a deterministic first-slice comparison is intended. Requested sizes are validated against the available samples.

### E3: Imbalance Stress

Purpose: test sensitivity to unequal source-domain sample counts.

The current `domain_stress` conditions are:

- Balanced: `2000,2000,2000,2000`
- Mild last-domain-heavy: `2000,2000,2000,4000`
- Strong last-domain-heavy: `2000,2000,2000,10000`

These conditions do not by themselves prove a general minority-domain-underrepresentation result. They specifically overweight the last listed source environment. Any report must state which environment is head or tail and must not describe these results as both directional imbalance types.

### E3b: Tail-Support / Missing-Support Stress

Purpose: measure what happens when a tail source domain is visible, sparsely represented, or absent from training while remaining present during evaluation.

The generated conditions are:

- `balanced_visible`: `2000,2000,2000,2000`
- `long_tail_visible`: `5000,2000,800,200`
- `near_missing_tail`: `5800,1800,350,50`
- `missing_tail`: `6000,1500,500,0`

The full manifest is `CMNIST/job_scripts/e3b_tail_support.txt` with 60 commands for 3 seeds, 4 conditions, and 5 algorithms. The current `results/E3b_tail_support/` analysis contains 45 records covering:

- `balanced_visible`
- `long_tail_visible`
- `missing_tail`

It does not contain completed `near_missing_tail` results. The E3b analyzer exports:

- `raw_results.csv`
- `summary_by_condition.csv`
- `slide_table.csv`
- `tail_accuracy_by_condition.png`
- `worst_accuracy_by_condition.png`
- `head_tail_gap_by_condition.png`
- `cvar_gap_by_condition.png`

For `missing_tail`, CMNIST drops zero-sized source environments from the training loaders but keeps the corresponding environment in test evaluation. The logs retain the requested counts, active environments, and empirical training priors.

## Lambda Evaluation

CMNIST does not initially train IRO for one fixed lambda. IRO and INF-TASK are trained with lambda-related behavior and can then be evaluated over a post-training grid. Baselines can also be evaluated with fixed predictions while changing the risk aggregation.

The evaluator accepts a checkpoint path, glob, or directory:

```text
cd CMNIST
dgil_env\Scripts\python.exe evaluate_lambda_grid.py ..\results\cmnist_exp\ckpts --output_dir ..\results\cmnist_exp\lambda_results --lambda_grid 0.0:1.0:0.1
```

Each lambda-evaluation JSONL row contains the algorithm, seed, checkpoint metadata, per-environment accuracy and loss, aggregated risk, and summary accuracy fields. The current lambda plot is:

- `results/cmnist_exp/plots/e4_lambda_aggregated_risk.png`

Lambda outputs should be checked for duplicate checkpoints and matched to the corresponding training run before calculating final sensitivity or regret statistics.

## Result Schema

Training result rows written by `CMNIST/train_sandbox.py` include, where applicable:

- `algorithm`
- `seed`
- `args`
- `args_id`
- `train_envs`
- `train_env_sizes`
- `test_env` accuracy and loss fields
- best and final model-selection metrics
- `tail_support_condition`
- tail-support source counts and empirical priors

`CMNIST/collect_results.py` derives or normalizes:

- `phase`
- `n_train_domains`
- `sample_size_per_domain`
- `imbalance_type`
- average, worst, and best domain accuracy

Do not treat the number of JSONL rows as the number of independent training jobs. A run can produce multiple records, and recovery runs can leave duplicate or partial records.

## Reproduction Commands

Generate the full CMNIST stress manifest:

```text
cd CMNIST
dgil_env\Scripts\python.exe job_scripts\gen_exps.py --data_dir ..\data --output_dir ..\results\cmnist_exp --exp_name domain_stress
```

Generate the E3b tail-support manifest:

```text
cd CMNIST
dgil_env\Scripts\python.exe job_scripts\gen_exps.py --data_dir ..\data --output_dir ..\results\E3b_tail_support --exp_name e3b_tail_support --seed_list 0,1,2
```

Aggregate CMNIST JSONL results:

```text
cd CMNIST
dgil_env\Scripts\python.exe collect_results.py ..\results\cmnist_exp\results --group_by phase --metric worst_domain_acc_best
```

Create E0-E3 stress plots:

```text
cd CMNIST
dgil_env\Scripts\python.exe plot_domain_stress.py ..\results\cmnist_exp\results --output_dir ..\results\cmnist_exp\plots
```

Create E3b CSV outputs and plots:

```text
cd CMNIST
dgil_env\Scripts\python.exe analyze_tail_support.py ..\results\E3b_tail_support\results --lambda_results_dir ..\results\E3b_tail_support\lambda_results --output_dir ..\results\E3b_tail_support
```

## Interpretation Rules

- Reduced and smoke outputs demonstrate pipeline behavior, not final comparative evidence.
- The 450-command manifest is a generated experimental grid, not evidence that all 450 jobs completed.
- Seed-level means and standard deviations require one clean, deduplicated record per intended run.
- E3 and E3b must be described with their exact sample schedules and head/tail ordering.
- Lambda sensitivity is a post-training evaluation analysis unless a separate fixed-lambda training experiment is explicitly run.
- Any scientific conclusion should report the exact result root, manifest, seed set, algorithm set, and condition coverage.
