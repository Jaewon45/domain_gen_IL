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
| E3b tail-support analysis | Four-condition design complete: 100 records for 5 seeds, 4 conditions, and 5 algorithms | `results/E3b_tail_support/` |
| Lambda sensitivity | Evaluation code and lambda artifacts exist; treat final scientific conclusions as pending inspection and cleanup | `results/cmnist_exp/lambda_results/` |

The files in `results/cmnist_exp/` should not be interpreted as a clean completed 450-job study. The root currently contains 553 JSONL result rows, including evaluation records from partially executed and recovered runs. Deduplicate and inspect logs before reporting seed-level statistics.

Priority 1 cleanup artifacts are now available under `results/cmnist_exp_clean_priority1_v4/`:

- `clean_training_results.jsonl`: one selected complete training record per canonical run key (120 records).
- `clean_metric_summary.csv`: mean, standard deviation, and count for each retained condition/algorithm group, computed only from clean records.
- `lambda_results.jsonl`: separated post-training lambda records (418 records).
- `clean_summary.csv`: retained training-record counts by algorithm and seed.
- `results/cmnist_exp/audit_priority1_v5.csv`: canonical candidates and duplicate/recovery flags.
- `results/cmnist_exp/manifest_audit.csv`: command-to-retained-record matching for the staged seed manifests.
- `results/cmnist_exp/log_audit.csv`: aggregate evidence from the shared `out.txt` and `err.txt` logs.
- `results/cmnist_exp/log_run_audit.csv`: classification of each sequential `Args:` block in the available runner log.

The manifest audit matches 125 of 215 staged commands to retained records. The available runner log contains 140 complete invocation blocks with no detected error markers. Because jobs append to shared logs, the log evidence cannot distinguish every duplicate/recovery invocation from its corresponding result row. The clean metric summary is based only on the selected complete records.

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

The original exploratory `domain_stress` conditions were:

| Number of source domains | Training environments |
| ---: | --- |
| 2 | `0.1,0.2` |
| 4 | `0.01,0.12,0.5,0.99` |
| 8 | `0.01,0.12,0.0,0.0,0.14,0.5,0.7,0.99` |

Those original results use non-nested source sets and omit 6 domains; keep them exploratory or appendix-only. The corrected manifest is `CMNIST/job_scripts/domain_count_clean.txt`. It uses approximately nested and symmetric source sets for 2, 4, 6, and 8 domains, keeps the total source budget fixed by the dataset partitioning, and uses the identical test grid for every condition. Its result root is `results/cmnist_domain_count_clean_v1/`.

The main derived metrics are average-domain accuracy, worst-domain accuracy, and best-domain accuracy across the test environments. Result records also retain the algorithm, seed, training environments, and evaluation metrics.

The corrected E1 is sufficient as a secondary AISTATS experiment, not as the main empirical result. Only one source-domain set is tested for each domain count, so the result does not establish that performance universally improves or deteriorates solely because the number of domains changes. It should be reported as a controlled single-configuration study unless additional alternative subsets are run.

E1 is now complete for seeds 0-4 (100/100 successful runs). The raw result directory retains 108 jsonl files because 8 seed-3 configs were executed more than once during earlier pre-fix launcher attempts; analysis is built from a deduplicated 100-record set (`CMNIST/dedup_and_refresh_e1_e3.py`) that keeps only the most recently written record per (seed, algorithm, train_envs, train_env_sizes).

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

For Priority 5 interpretation, use only the corrected fixed-budget E3 for directional imbalance claims. The historical `domain_stress` E3 is exploratory because it changes both source prior and total source sample count. The corrected study uses **balanced**, **mild first-heavy**, **mild last-heavy**, **strong first-heavy**, and **strong last-heavy** schedules. Report the exact schedules and identify the overweighted source environment. Use `first-heavy` and `last-heavy`, not `majority` or `minority`, unless that interpretation is justified by the environment definition. The safest claim is the within-method change from balanced to imbalanced conditions; cross-method superiority claims require care because algorithm-specific optimization schedules differ.

A corrected Priority 5 study is complete in `results/imbalance_clean_v1/`, generated from `CMNIST/job_scripts/imbalance_clean.txt`. All 125 commands succeeded across five fixed-total-budget schedules: balanced, mild first-heavy, mild last-heavy, strong first-heavy, and strong last-heavy, covering seeds 0-4. Each schedule has total source count 8,000, uses source environments `0.1,0.2,0.5,0.9`, and covers all five algorithms. The guarded analyzer `CMNIST/job_scripts/bashes/analyze_imbalance_clean.ps1` generated the seed 0-2 CSVs; `CMNIST/dedup_and_refresh_e1_e3.py` regenerates `analysis_seed0-4/`, `analysis/run_summary.json`, and the per-environment curve `analysis/e3_imbalance_accuracy_by_test_env.png` for the full seed 0-4 set.

### E3b: Tail-Support / Missing-Support Stress

Purpose: measure what happens when a tail source domain is visible, sparsely represented, or absent from training while remaining present during evaluation.

The generated conditions are:

- `balanced_visible`: `2000,2000,2000,2000`
- `long_tail_visible`: `5000,2000,800,200`
- `near_missing_tail`: `5800,1800,350,50`
- `missing_tail`: `6000,1500,500,0`

The selected final design is the four-condition study. The full manifest is `CMNIST/job_scripts/e3b_tail_support.txt` with 100 commands for 5 seeds, 4 conditions, and 5 algorithms (originally 60 commands for seeds 0-2; seeds 3-4 were added later with no duplicate records). The dedicated missing-condition manifest is `CMNIST/job_scripts/e3b_tail_support_near_missing_seed012.txt` contains the same 15 near-missing-tail commands for targeted reruns. The current `results/E3b_tail_support/` analysis contains all 100 records covering:

- `balanced_visible`
- `long_tail_visible`
- `missing_tail`

The analyzer exports:

- `raw_results.csv`
- `summary_by_condition.csv`
- `slide_table.csv`
- `tail_accuracy_by_condition.png`
- `worst_accuracy_by_condition.png`
- `head_tail_gap_by_condition.png`
- `cvar_gap_by_condition.png`

For `missing_tail`, CMNIST drops zero-sized source environments from the training loaders but keeps the corresponding environment in test evaluation. The logs retain the requested counts, active environments, and empirical training priors.

### E3b 4-Anchor CVaR Analysis (post-hoc, no retraining)

The default E3b deployment CVaR is computed over all 11 fixed test environments (`0.0,0.1,...,1.0`), even though only 4 of them (`0.1,0.2,0.5,0.9`) are actual source anchors. This is a valid deployment-generalization measure, but it is not the tightest possible link to the missing-support theorem, which is stated over the source-anchor support.

`CMNIST/analyze_tail_support.py` already supports restricting the evaluation/deployment environment set via `--eval_envs`, so this required no retraining and no code changes:

```text
dgil_env\Scripts\python.exe CMNIST\analyze_tail_support.py results\E3b_tail_support\results --output_dir results\E3b_tail_support\analysis_4anchor_seed0-4 --eval_envs 0.1,0.2,0.5,0.9
```

This produces the same file set restricted to the 4 source anchors (400 rows in `raw_results.csv` = 100 run-records x 4 domains), with the deployment prior now uniform over exactly `{0.1,0.2,0.5,0.9}` instead of all 11 test environments. Submission copies use a `_4anchor` suffix: `results_submit/tables/E3b_tail_support/raw_results_4anchor.csv`, `summary_by_condition_4anchor.csv`, `slide_table_4anchor.csv`, and the four corresponding `*_4anchor.png` figures. This is the recommended analysis for directly connecting the missing-support theorem to the E3b experiment; the original 11-environment version remains available as the broader deployment-generalization measure.

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

## Implemented Objectives

Let $B_a$ be the active source minibatch for environment $a$, and let

$$
\ell_a(\theta,\alpha)
= \{\ell(h_\theta(x,\alpha),y):(x,y)\in B_a\}.
$$

The implementation uses the following objectives:

- **ERM:** concatenates all active source minibatches and minimizes the mean per-example loss. With equal minibatch sizes this is equivalent to the mean source-domain loss; with unequal minibatch sizes it gives larger minibatches more weight.
- **GroupDRO:** computes one mean loss $r_a(\theta)$ per active source minibatch, updates $q_a \leftarrow q_a\exp(\eta r_a)$, normalizes $q$, and minimizes $\sum_a q_a r_a$. The initial active-domain weights are uniform. The code does not use the logged empirical prior $n_a/N$ for this update.
- **INF-TASK:** samples five $\alpha$ values from $\operatorname{Beta}(1,1)$ and minimizes the mean of the corresponding CVaR values over pooled per-example losses from the active source minibatches.
- **IRO:** updates a Beta distribution from a gradient-norm signal, samples ten $\alpha$ values from the updated $\operatorname{Beta}(a,b)$, and minimizes the mean of the corresponding CVaR values over pooled per-example losses from the active source minibatches.

More explicitly, at update $t$ the implementation samples $\alpha_{t,j}\sim\operatorname{Beta}(a_t,b_t)$ for $j=1,\ldots,10$ and minimizes

$$
\mathcal{L}_{\mathrm{IRO}}(\theta)
= \frac{1}{10}\sum_{j=1}^{10}
\operatorname{CVaR}_{\alpha_{t,j}}
\left(
\bigcup_{a\in\mathcal{A}_{\mathrm{active}}}
\ell_a(\theta,\alpha_{t,j})
\right),
$$

where $(a_t,b_t)$ is updated before sampling by `Pareto_distribution.update` using the gradient norm of a copied network. This is a pooled per-example-loss objective; it is not a weighted sum of one scalar risk per domain.

For the CVaR implementation, the threshold is the linear-interpolation empirical quantile at $\alpha$, and the returned value is the unweighted mean of losses greater than or equal to that threshold. Therefore the current IRO/INF-TASK training objective is not a CVaR over one scalar risk per domain and does not directly apply a deployment prior. The logged empirical prior $\hat\pi_a=n_a/N$ is currently analysis metadata only.

For tail-support runs, a zero-count source domain is removed from the training loader. It remains in the test/evaluation set, where analysis assigns it zero empirical weight and positive deployment weight under the uniform evaluation prior. This is the implemented prior-mismatch experiment, not a claim that the missing domain contributes to the training objective.

## Priority 0 Conclusions

The exact scientific object currently supported by the code is:

1. Source training uses active source minibatches.
2. GroupDRO operates on active-domain mean losses with adaptive uniform-initialized group weights.
3. IRO and INF-TASK use alpha-conditioned predictors and CVaR over pooled source-example losses.
4. Empirical source priors are logged and used by the E3b analysis, but not by the CMNIST training objectives.
5. Deployment-prior CVaR is an analysis quantity in `CMNIST/analyze_tail_support.py`, separate from the training loss.

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

## GroupDRO Audit

GroupDRO receives one minibatch from each active source loader. It computes one mean loss per source domain, updates uniform-initialized adaptive weights with `groupdro_eta=0.1`, normalizes those weights, and minimizes their weighted sum. It does not use the logged empirical source prior in training.

The focused implementation checks are in `CMNIST/tests/test_groupdro.py`. New GroupDRO runs log `group_losses`, `group_weights`, and final `source_worst_loss_best` / `source_worst_loss_final` fields. Historical GroupDRO records in the clean export predate these fields, so source-objective transfer claims require rerunning or separately backfilling those runs.

Existing ERM and GroupDRO stress runs are not a strict controlled comparison: the generated manifest uses 600 steps and no pretraining for ERM, versus 1,000 steps and 400 ERM-pretraining steps for GroupDRO. Both use the same main architecture and batch size, but the optimization budgets differ.

**Paper placement:** the matched-budget GroupDRO comparison (`results/cmnist_groupdro_control_v2/`) is supplement-only and still three seeds (0-2), unchanged from earlier passes. It is fine as a control at that scope, but keep at most one main-text sentence: source-objective robustness does not necessarily transfer to target-domain robustness. Do not extend this claim beyond the matched-budget control or present it as five-seed evidence.

## Priority 7 Theory Simulation

The independent synthetic experiment in `CMNIST/priority7_theory.py` isolates prior mismatch from neural-network optimization. It uses a uniform deployment prior, a long-tailed source prior, two head/tail trade-off risk profiles, empirical and deployment CVaR, and repeated multinomial source samples. The full sweep varies alpha, source sample size, missing-tail fraction, and long-tail exponent.

Outputs are stored under `results/cmnist_priority7_theory_v1/`:

- `ranking_reversal_summary.csv`
- `trial_diagnostics.csv`
- `ranking_reversal_heatmap.png`
- `ranking_reversal_by_sample_size.png`

The calibrated profiles produce a non-tied deployment ranking and measurable empirical/deployment ranking reversals. Forced missing-tail support increases reversal probability in the tested settings.

The sample-size effect is bimodal, not monotone in one direction, and the mechanism is now fully characterized. Isolating pure sample-size effects (`missing_tail_fraction=0`, no forced missing domains) across the swept sample sizes `32, 128, 512, 2048, 8192`:

- When `exponent=0` (source prior equals the uniform deployment prior, no persistent mismatch), reversal probability shrinks toward 0 as sample size grows (e.g., alpha 0.50: `0.216 -> 0.031 -> 0.000 -> 0.0 -> 0.0`). This matches the naive intuition that more evidence reduces sampling-noise-driven disagreement.
- When `exponent>0` (a persistent long-tailed source prior), reversal probability instead rises toward 1 as sample size grows (e.g., exponent 0.5, alpha 0.50: `0.868 -> 0.970 -> 0.999 -> 1.0 -> 1.0`; exponent 1.0, alpha 0.75: `0.991 -> 1.0 -> 1.0 -> 1.0 -> 1.0`).

This is not sampling noise being resolved; with `exponent>0` the empirical prior converges toward the fixed long-tailed source prior rather than toward the uniform deployment prior, so more samples make the empirical ranking converge to a systematically wrong answer with increasing confidence. This is a stronger and more useful result than a simple monotonic-disagreement claim: additional source data does not correct persistent support mismatch, and can instead make the incorrect empirical conclusion increasingly confident.

## Priority 6 Lambda Audit

The CPU-only lambda audit in `CMNIST/analyze_lambda_sensitivity.py` checked the existing 418 lambda rows. It found 38 checkpoint/type combinations, 11 lambda values per checkpoint, and no duplicate checkpoint/lambda keys. Current coverage is IRO, INF-TASK, and IRM; ERM and GroupDRO are not present in the lambda result root.

**Paper placement:** the E4 lambda-sensitivity evidence used for reporting (`results/cmnist_lambda_prediction_eval_v2/`) is supplement-only: still exactly one IRO checkpoint and one INF-TASK checkpoint, both seed 0. Keep at most one main-text sentence describing checkpoint-level lambda sensitivity and deployment-wide pseudo-regret (not true operator regret), and do not present this as algorithm-wide or multi-seed evidence.

Outputs are in `results/cmnist_lambda_audit_v1/`:

- `lambda_records_deduplicated.jsonl`
- `lambda_sensitivity_summary.csv`
- `lambda_coverage.csv`

The current evaluator uses `lambda_eval` both as the alpha supplied to alpha-conditioned predictors and as the CVaR aggregation alpha. Separate `lambda_model` and `alpha_eval` sweeps remain unavailable. Predictor disagreement is now measured by the prediction-level re-evaluation below; true operator regret still requires fixed-lambda reference models.

The prediction-level re-evaluation is now available under `results/cmnist_lambda_prediction_eval_v2/`. It used one IRO and one INF-TASK checkpoint, five fixed test environments, and 11 lambda values. It saved per-example predictions and metrics in `*_predictions.npz`, `prediction_lambda_metrics.csv`, and `prediction_lambda_sensitivity_summary.csv`. The selected IRO checkpoint had a mean-environment accuracy range of about 0.25 percentage points and maximum disagreement from lambda 0 of about 3.9%; INF-TASK had about 3.19 percentage points and 15.9%, respectively. These are checkpoint-level results, not a claim over all CMNIST runs. Operator regret remains unavailable because no fixed-lambda retrained reference models were trained.

The clean prediction-level lambda figure is `results/cmnist_lambda_prediction_eval_v2/plots/lambda_prediction_accuracy_curve.png`, with tabular data in `lambda_prediction_accuracy_summary.csv`.

Deployment-wide pseudo-regret logging is available without changing the CMNIST training pipeline:

```text
bash CMNIST/job_scripts/bashes/run_regret_logging.sh
```

This creates deployment-wide same-checkpoint pseudo-regret in `results/cmnist_lambda_pseudoregret_v2/`. For each checkpoint, one oracle lambda is selected by mean accuracy across all deployment/test environments. Every used lambda is compared with that same oracle, so the result is not an environment-specific oracle and is not called empirical operator regret. It measures the cost of choosing another lambda for the same trained checkpoint, not the gap to a separately trained fixed-lambda expert.

The earlier `results/cmnist_lambda_pseudoregret_v1/` output used an environment-specific oracle and is superseded for reporting. True operator regret remains optional: it would require a separate fixed-lambda reference training study. If that study is later run, a reference CSV with columns `algorithm`, `seed`, `test_env`, `lambda_reference`, and `accuracy` can be supplied using `--fixed_reference_csv`.

### Optional True-Operator-Regret Study

The preferred optional strengthening is a reduced fixed-lambda reference study, not a requirement for the current AISTATS analysis. Train fixed-lambda reference models at `lambda in {0, 0.5, 1}` and compare them with the existing lambda-conditioned models.

Minimal feasibility version:

$$
3\text{ lambdas}
	imes 3\text{ seeds}
	imes 1\text{ algorithm (IRO)}
= 9\text{ training runs}.
$$

Reasonable report version:

$$
3\text{ lambdas}
	imes 3\text{ algorithms (ERM, GroupDRO, IRO)}
	imes 3\text{ seeds}
	imes 5\text{ conditions (E0 + 4 E3b conditions)}
= 135\text{ training runs}.
$$

The phrase “5 conditions” means **E0 plus the four E3b conditions**: `balanced_visible`, `long_tail_visible`, `near_missing_tail`, and `missing_tail`. It must not be described as “5 E3b conditions.” This study requires fixed-lambda training support and is optional; the current deployment-wide pseudo-regret and model-level lambda sensitivity analysis remain the primary low-cost analysis.

The matched-budget comparison is now complete using `CMNIST/job_scripts/groupdro_controlled_seed012_v2.txt`. Across three seeds, GroupDRO has observed-source worst loss `0.118 +/- 0.032` versus ERM `0.597 +/- 0.033`, but held-out target worst accuracy `0.600 +/- 0.014` versus ERM `0.607 +/- 0.016`. In this run, GroupDRO improves its observed-source objective without transferring that improvement to unseen target accuracy. The per-seed table is `results/cmnist_groupdro_control_v2/analysis/source_target_by_seed.csv`, and the two-panel source-loss-versus-target-accuracy figure is `results/cmnist_groupdro_control_v2/analysis/source_target_by_seed.png` (submission copy: `results_submit/figures/GroupDRO_control/source_target_by_seed.png`).

The corrected E1 domain-count study ran from `CMNIST/job_scripts/domain_count_clean.txt` into `results/cmnist_domain_count_clean_v1/`, using the resumable runner `CMNIST/job_scripts/run_groupdro_controlled_seed012.ps1`.

The corrected E1 study is complete: all 100 commands succeeded across seeds 0-4, algorithms ERM/IRM/GroupDRO/IRO/INF-TASK, and source-domain counts 2/4/6/8. Raw results include 8 duplicate seed-3 records from pre-fix launcher retries; `CMNIST/dedup_and_refresh_e1_e3.py` deduplicates these to the reported 100-record set. Cross-seed analysis is in `results/cmnist_domain_count_clean_v1/analysis_seed0-4/`:

- `domain_count_by_algorithm.csv`: mean, standard deviation, and count across seeds.
- `domain_count_accuracy_by_test_env.csv`: per-seed accuracy by test environment.
- `domain_count_accuracy_by_test_env.png`: four-panel accuracy curves for 2, 4, 6, and 8 source domains.
- `E1_report_table_verification.csv`: 60 mean/std checks against the deduplicated clean E1 records; all checks pass.

The original non-nested E1 results remain exploratory and should not be pooled with this corrected ablation. The corrected study uses one fixed source-environment set for each domain count; variability across multiple alternative subsets is not measured.

The corrected E3 summary preserves semantic schedule labels: `balanced`, `first_heavy_mild`, `last_heavy_mild`, `first_heavy_strong`, and `last_heavy_strong`. Each schedule has 25 seed-level result rows (5 seeds x 5 algorithms). The cross-seed summary is `results/imbalance_clean_v1/analysis_seed0-4/imbalance_by_algorithm_cross_seed.csv`, and its 75 mean/std/count values pass verification in `analysis_seed0-4/E3_report_table_verification.csv`.

To verify the E1 summary independently:

```text
dgil_env\Scripts\python.exe CMNIST\verify_report_tables.py results\cmnist_domain_count_clean_v1\results results\cmnist_domain_count_clean_v1\analysis_seed0-4\domain_count_by_algorithm.csv --phase domain_count --output results\cmnist_domain_count_clean_v1\analysis_seed0-4\report_table_verification_check.csv
```

Note: `verify_report_tables.py` reads raw (non-deduplicated) records, so it will flag the 8 duplicate seed-3 E1 configs; use `CMNIST/dedup_and_refresh_e1_e3.py` as the source of truth for the deduplicated 100-record verification.

The cross-experiment artifact index is `results/cmnist_artifact_index.csv`, generated by `CMNIST/build_artifact_index.py`. It records each CMNIST result root, manifest, and available JSONL/CSV/plot counts without modifying experiment outputs.

A controlled comparison manifest is available at `CMNIST/job_scripts/groupdro_controlled_seed012.txt`. It runs ERM and GroupDRO for seeds 0-2 with the same four source environments, balanced source sizes, test environments, 1,000-step budget, 400-step ERM pretraining, cosine schedule, batch size, and checkpoint policy. Its output root is `results/cmnist_groupdro_control_v1/`. Run it only after the active E3b jobs finish, then use the new GroupDRO source-loss fields to compare observed-source worst loss against held-out-target worst accuracy.

Use the resumable Windows runner to execute this comparison with per-command progress logging:

```powershell
.\CMNIST\job_scripts\run_groupdro_controlled_seed012.ps1
```

The runner writes `runner_logs/progress.csv`, `runner_logs/summary.txt`, and separate stdout/stderr files under `results/cmnist_groupdro_control_v1/`. It continues after a failed command. To backfill only unfinished commands after an interruption, rerun with:

```powershell
.\CMNIST\job_scripts\run_groupdro_controlled_seed012.ps1 -Resume
```

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

Run the selected missing condition:

```text
cd CMNIST
for /f "usebackq delims=" %i in ("job_scripts\e3b_tail_support_near_missing_seed012.txt") do @cmd /c "%i"
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
