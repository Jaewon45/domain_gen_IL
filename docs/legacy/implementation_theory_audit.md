# Implementation-Theory Audit

Date: 2026-08-31

## A. Synthetic Reversal Boolean Condition

Source: `CMNIST/priority7_theory.py`.

The synthetic profiles are:

- `head_favored`: `0.20 + tradeoff * position`, with default `tradeoff = 0.5`.
- `tail_favored`: `0.15 + tradeoff * (1.0 - position)`, equivalent to `0.65 - 0.50(a-1)/9` when `K = 10`.

Under the uniform deployment law, `tail_favored` is preferred by exact discrete CVaR. The implemented reversal flag is now explicitly:

```python
deployment_scores["tail_favored"] < deployment_scores["head_favored"]
and empirical_scores["head_favored"] < empirical_scores["tail_favored"]
```

This matches the requested orientation:

`deployment_cvar_g < deployment_cvar_f AND source_cvar_f < source_cvar_g`.

Before this audit edit, the code used `empirical_winner != deployment_winner`. With two candidates and no ties this was not the opposite event, but it did not spell out the paper orientation.

Focused test added: `CMNIST/tests/test_theory_audit.py::TheoryAuditTests.test_synthetic_reversal_condition_orientation`.

## B. Exact Source-Support Removal Formula / Code Behavior

Source: `CMNIST/priority7_theory.py::source_prior`.

For `K = 10`, the code computes:

```python
probabilities = ranks ** (-float(exponent))
missing_count = int(np.floor(n_domains * float(missing_tail_fraction)))
probabilities[-missing_count:] = 0.0
probabilities /= probabilities.sum()
```

Thus the mathematical form is:

`pi_src(a; gamma, epsilon) proportional to a^(-gamma) * 1[a <= K - floor(K epsilon)]`, renormalized over supported domains. For the requested epsilon grid, `K * epsilon` is integral, so this equals `1[a <= K - K epsilon]`. For `epsilon = 0.3`, exactly domains `1,...,7` remain supported.

Source vectors from the implementation, rounded to 6 decimals:

| gamma | epsilon | missing | support | probabilities |
|---:|---:|---:|---|---|
| 0.0 | 0.0 | 0 | 1-10 | [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1] |
| 0.0 | 0.1 | 1 | 1-9 | [0.111111, 0.111111, 0.111111, 0.111111, 0.111111, 0.111111, 0.111111, 0.111111, 0.111111, 0.0] |
| 0.0 | 0.2 | 2 | 1-8 | [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.0, 0.0] |
| 0.0 | 0.3 | 3 | 1-7 | [0.142857, 0.142857, 0.142857, 0.142857, 0.142857, 0.142857, 0.142857, 0.0, 0.0, 0.0] |
| 0.5 | 0.0 | 0 | 1-10 | [0.199164, 0.14083, 0.114987, 0.099582, 0.089069, 0.081308, 0.075277, 0.070415, 0.066388, 0.062981] |
| 0.5 | 0.1 | 1 | 1-9 | [0.21255, 0.150296, 0.122716, 0.106275, 0.095055, 0.086773, 0.080336, 0.075148, 0.07085, 0.0] |
| 0.5 | 0.2 | 2 | 1-8 | [0.228758, 0.161756, 0.132073, 0.114379, 0.102304, 0.09339, 0.086462, 0.080878, 0.0, 0.0] |
| 0.5 | 0.3 | 3 | 1-7 | [0.248887, 0.17599, 0.143695, 0.124444, 0.111306, 0.101608, 0.094071, 0.0, 0.0, 0.0] |
| 1.0 | 0.0 | 0 | 1-10 | [0.341417, 0.170709, 0.113806, 0.085354, 0.068283, 0.056903, 0.048774, 0.042677, 0.037935, 0.034142] |
| 1.0 | 0.1 | 1 | 1-9 | [0.353486, 0.176743, 0.117829, 0.088371, 0.070697, 0.058914, 0.050498, 0.044186, 0.039276, 0.0] |
| 1.0 | 0.2 | 2 | 1-8 | [0.367937, 0.183968, 0.122646, 0.091984, 0.073587, 0.061323, 0.052562, 0.045992, 0.0, 0.0] |
| 1.0 | 0.3 | 3 | 1-7 | [0.385675, 0.192837, 0.128558, 0.096419, 0.077135, 0.064279, 0.055096, 0.0, 0.0, 0.0] |
| 2.0 | 0.0 | 0 | 1-10 | [0.645258, 0.161314, 0.071695, 0.040329, 0.02581, 0.017924, 0.013169, 0.010082, 0.007966, 0.006453] |
| 2.0 | 0.1 | 1 | 1-9 | [0.649449, 0.162362, 0.072161, 0.040591, 0.025978, 0.01804, 0.013254, 0.010148, 0.008018, 0.0] |
| 2.0 | 0.2 | 2 | 1-8 | [0.654698, 0.163674, 0.072744, 0.040919, 0.026188, 0.018186, 0.013361, 0.01023, 0.0, 0.0] |
| 2.0 | 0.3 | 3 | 1-7 | [0.661464, 0.165366, 0.073496, 0.041342, 0.026459, 0.018374, 0.013499, 0.0, 0.0, 0.0] |

Focused test added: `CMNIST/tests/test_theory_audit.py::TheoryAuditTests.test_synthetic_source_prior_removes_final_tail_domains`.

## C. CVaR Implementation Details and Unit-Test Results

The exact weighted discrete CVaR rule used for synthetic/post-hoc audit paths now integrates exactly the upper `1 - alpha` mass, splitting the boundary atom when needed.

Touched helpers:

| Helper | Use | Status |
|---|---|---|
| `CMNIST/priority7_theory.py::cvar` | synthetic ranking reversal and identification-width import | corrected to fractional upper-tail integration |
| `CMNIST/analyze_tail_support.py::weighted_cvar` | CMNIST tail-support post-hoc empirical/deployment CVaR summaries | corrected to fractional upper-tail integration |
| `run_posthoc_and_theory_sims.py::cvar_discrete` | theorem-aligned 0-1 post-hoc bounds | already used fractional upper-tail integration |
| `CMNIST/lib/iro_utils.py::aggregation_function.cvar` | differentiable neural training objective and lambda-grid `aggregated_risk` | threshold-tail mean, not fractional discrete integration |
| `CMNIST/lib/misc.py::cvar` | printed auxiliary final/best diagnostics via `aggregation_function.cvar` | threshold-tail mean through `aggregation_function` |
| `IMAGENET_C/evaluate_lambda_grid.py::weighted_cvar` | ImageNet-C auxiliary lambda summaries | threshold-tail mean |
| `IMAGENET_C/eval_repeatability.py::weighted_cvar` | ImageNet-C repeatability lambda curve | threshold-tail mean with alpha clipped into `[0, 1]` |
| `IMAGENET_C/train_head.py::cvar_objective` | ImageNet-C training objective for IRO smoke/head model | threshold-tail mean |

Unit tests added in `CMNIST/tests/test_theory_audit.py`:

- `risks = [0, 1]`, `weights = [0.9, 0.1]`, `alpha = 0.9`, expected CVaR `1.0`.
- Fractional-boundary case: `risks = [0, 10]`, `weights = [0.75, 0.25]`, `alpha = 0.5`, expected CVaR `5.0`.
- Uniform-deployment synthetic gap for `alpha in {0.5, 0.75, 0.9}`: `CVaR(f) - CVaR(g) = 0.05`.

Validation command:

```powershell
Set-Location c:\Users\320257223\PycharmProjects\domain_gen_IL\CMNIST
..\dgil_env\Scripts\python.exe -m unittest discover -s tests -p test_theory_audit.py -v
```

Result: 4 tests ran and passed.

## D. lambda = 1 Handling

CMNIST training/evaluation uses `CMNIST/lib/iro_utils.py::aggregation_function.cvar` for the `cvar` aggregate:

```python
var = torch.quantile(risks, alpha, interpolation='linear')
cvar = risks[risks >= var].mean()
```

At `lambda = 1`, this does not divide by zero. It computes the quantile at 1.0, selects losses equal to the maximum loss, and returns the mean over max-tied losses. Therefore `lambda = 1` is treated as a maximum-loss / essential-supremum endpoint for the active finite risk vector.

Important caveat: `cvar_full` in the same file contains `1 / (1 - alpha)` and would divide by zero at `alpha = 1`, but the active CMNIST IRO/INF-TASK and lambda-grid paths instantiate `aggregation_function(name='cvar')`, not `cvar-full`.

## E. IRO Beta-Update Specification

Sources: `CMNIST/algorithms.py::IRO` and `CMNIST/lib/iro_utils.py::Pareto_distribution`.

- Initial Beta parameters: `a = 1.0`, `b = 1.0` from `self.dist_param = torch.tensor([1.0, 1.0], requires_grad=True, ...)`.
- Update frequency: every IRO `update()` call after ERM pretraining. During pretraining (`update_count < erm_pretrain_iters`) IRO delegates to ERM and does not update Beta.
- Statistic used: gradient norm of model parameters under an average CVaR-difference objective. `Pareto_distribution.update()` computes `avg_cvar`, differentiates it w.r.t. model parameters, forms the total L2 norm of those gradients, then differentiates that norm w.r.t. the Beta parameters.
- Internal objective samples per Beta update: `Pareto_distribution.aggregated_objective(..., num_samples=5)` samples 5 uniform values and maps them through the Beta inverse CDF.
- Exact update: `updated = (self.dist_param - 0.000001 * obj_grad).clamp(min=1e-4)`.
- Bounds: lower clamp `1e-4`; no explicit upper clamp.
- Number of lambda samples for the IRO training loss after the Beta update: `np.random.beta(a, b, size=10)`. This confirms ten lambda values are sampled per IRO update.

INF-TASK comparison: `np.random.beta(1.0, 1.0, size=5)` per update.

## F. Method Hyperparameter Table

Source: `CMNIST/job_scripts/gen_exps.py` clean domain-count, imbalance, and tail-support generators, plus `CMNIST/train_sandbox.py` optimizer construction.

| algorithm | optimizer before pretraining | optimizer after pretraining | learning rate | weight decay | pretraining steps | main training steps | cosine schedule | IRM penalty coefficient | GroupDRO eta | sampled lambdas | IRO-specific coefficients | INF-TASK-specific coefficients |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---|---|
| ERM | AdamW | n/a | 1e-4 | 0 | 0 | 600 | no | n/a | n/a | 0 | n/a | n/a |
| IRM | AdamW | Adam | 1e-4 | 0 | 400 | 600 | yes, after pretraining | 1000 | n/a | 0 | n/a | n/a |
| GroupDRO | AdamW | Adam | 1e-4 | 0 | 400 | 1000 | yes, after pretraining | n/a | 0.1 | 0 | n/a | n/a |
| INF-TASK | AdamW | Adam | 1e-4 | 0 | 400 | 600 | yes, after pretraining | n/a | n/a | 5 per update | n/a | Beta(1,1), no adaptive coefficients |
| IRO | AdamW | Adam | 1e-4 | 0 | 400 | 600 | yes, after pretraining | n/a | n/a | 10 per update | Beta update step size 1e-6; min clamp 1e-4; 5 internal samples for Beta update | n/a |

`lr_factor_reduction = 1` in the generated runs, so the post-pretraining Adam learning rate remains `1e-4`.

## G. Effective Batching Behavior

Source: `CMNIST/train_sandbox.py`, `CMNIST/datasets.py`, and `CMNIST/lib/fast_data_loader.py`.

Training constructs one `FastDataLoader` per active source environment and zips them. `FastDataLoader` uses `RandomSampler(replacement=False)`, `BatchSampler(..., drop_last=False)`, and an infinite wrapper. Since reported `batch_size = 25000` exceeds every explicit tail-support per-domain size, each active source environment contributes its full available dataset in one optimization step. Zero-sized source domains are dropped before loaders are built.

Effective examples per optimization step:

| condition family | source counts | active loaders | effective examples per step |
|---|---:|---:|---:|
| clean domain count, 2 envs | about 25000 each from 50000 split | 2 | 50000 |
| clean domain count, 4 envs | about 12500 each from 50000 split | 4 | 50000 |
| clean domain count, 6 envs | about 8333/8334 each from 50000 split | 6 | 50000 |
| clean domain count, 8 envs | 6250 each from 50000 split | 8 | 50000 |
| tail support: balanced_visible | [2000, 2000, 2000, 2000] | 4 | 8000 |
| tail support: long_tail_visible | [5000, 2000, 800, 200] | 4 | 8000 |
| tail support: near_missing_tail | [5800, 1800, 350, 50] | 4 | 8000 |
| tail support: missing_tail | [6000, 1500, 500, 0] | 3 | 8000 |
| visible imbalance: balanced | [2000, 2000, 2000, 2000] | 4 | 8000 |
| visible imbalance: last_heavy_mild | [1500, 1500, 1500, 3500] | 4 | 8000 |
| visible imbalance: first_heavy_mild | [3500, 1500, 1500, 1500] | 4 | 8000 |
| visible imbalance: last_heavy_strong | [1000, 1000, 1000, 5000] | 4 | 8000 |
| visible imbalance: first_heavy_strong | [5000, 1000, 1000, 1000] | 4 | 8000 |

## H. Final-vs-Best Checkpoint Audit

| analysis_name | metric_source | final_or_best |
|---|---|---|
| clean domain-count summary | `CMNIST/analyze_domain_count.py` uses `worst_domain_acc_final`, `avg_domain_acc_final`, `best_domain_acc_final`, and per-env `*_acc_final` | final |
| visible-imbalance summary | `CMNIST/analyze_imbalance_summary.py` exports both `*_final` and `*_best` summary columns; per-env long CSV uses `*_acc_final` | mixed export; manuscript should use final columns |
| tail-support raw/summary | `CMNIST/analyze_tail_support.py::build_raw_rows_from_main` uses `*_acc_final` and `*_loss_final`; verified `analysis_4anchor_seed0-4/raw_results.csv` has `source = main_eval` only | final |
| theorem-aligned 0-1 post-hoc | `run_posthoc_and_theory_sims.py` reads tail-support raw `test_accuracy`; that raw file is from final main-eval metrics | final |
| lambda-grid risk audit | `CMNIST/evaluate_lambda_grid.py` records `checkpoint_type` from checkpoint filename; can evaluate either `_final` or `_best` depending input | explicit checkpoint_type, not intrinsically final-only |
| lambda prediction sensitivity | `CMNIST/evaluate_lambda_predictions.py` evaluates whichever checkpoint paths are provided; runner uses `results/cmnist_exp/ckpts` and records checkpoint paths | explicit checkpoint path, not intrinsically final-only |
| legacy domain-stress plots | `CMNIST/plot_domain_stress.py` defaults to `worst_domain_acc_best` and uses `_acc_best` for E0 and imbalance curves | best default; avoid for final-checkpoint claims |
| legacy table collection | `CMNIST/collect_results.py` CLI metric default is `worst_domain_acc_best`, but `--model_selection_type` default is `final` for table building | mixed defaults |

Conclusion: the newer principal clean domain-count, tail-support, and theorem-aligned 0-1 paths are final-checkpoint based. The visible-imbalance summary exports both final and best, so reported manuscript values must be taken from the `*_final` columns. Legacy plotting/table defaults still contain best-checkpoint defaults and should not be used for revised final-checkpoint claims.

## I. Theorem-Aligned 0-1 Construction

Source: `run_posthoc_and_theory_sims.py::run_task1` and `results/E3b_tail_support/analysis_4anchor_seed0-4/raw_results.csv`.

For `condition == "missing_tail"`, the code uses:

- `R_e = 1 - final_accuracy(e)` through `test_accuracy` values in final-derived raw rows.
- Observed anchors: `{0.1, 0.2, 0.5}`.
- Missing anchor: `0.9`.
- `P_obs` equal conditional mass `1/3` on observed anchors, implemented equivalently as total mass `0.75` split into `[0.25, 0.25, 0.25]`.
- `epsilon = 1/4`.
- `P_minus = 3/4 P_obs + 1/4 delta_0`, implemented as support `[r_01, r_02, r_05, 0.0]` and weights `[0.25, 0.25, 0.25, 0.25]`.
- `P_deploy` uniform mass `1/4` on all four anchors, support `[r_01, r_02, r_05, r_09]`.
- `P_plus = 3/4 P_obs + 1/4 delta_1`, support `[r_01, r_02, r_05, 1.0]`.
- Original exported alpha: `0.9`.

Additional mean bounds computed from the same final-derived raw table:

| alpha | algorithm | lower_cvar_mean | deployment_cvar_mean | upper_cvar_mean | id_width_mean |
|---:|---|---:|---:|---:|---:|
| 0.5 | erm | 0.335990 | 0.610550 | 0.726830 | 0.390840 |
| 0.5 | groupdro | 0.334950 | 0.605860 | 0.725610 | 0.390660 |
| 0.5 | inftask | 0.301010 | 0.473160 | 0.693430 | 0.392420 |
| 0.5 | irm | 0.319280 | 0.482250 | 0.694870 | 0.375590 |
| 0.5 | iro | 0.301810 | 0.477000 | 0.692490 | 0.390680 |
| 0.75 | erm | 0.453660 | 0.767440 | 1.000000 | 0.546340 |
| 0.75 | groupdro | 0.451220 | 0.760500 | 1.000000 | 0.548780 |
| 0.75 | inftask | 0.386860 | 0.559460 | 1.000000 | 0.613140 |
| 0.75 | irm | 0.389740 | 0.574760 | 1.000000 | 0.610260 |
| 0.75 | iro | 0.384980 | 0.569020 | 1.000000 | 0.615020 |
| 0.9 | erm | 0.453660 | 0.767440 | 1.000000 | 0.546340 |
| 0.9 | groupdro | 0.451220 | 0.760500 | 1.000000 | 0.548780 |
| 0.9 | inftask | 0.386860 | 0.559460 | 1.000000 | 0.613140 |
| 0.9 | irm | 0.389740 | 0.574760 | 1.000000 | 0.610260 |
| 0.9 | iro | 0.384980 | 0.569020 | 1.000000 | 0.615020 |

## J. Pairwise Ranking-Ambiguity Results

Using the theorem-aligned 0-1 final-derived results, each algorithm/seed has a `[lower_cvar, upper_cvar]` interval. For every unordered pair `(f, g)` within each seed:

- `f` robustly better than `g` if `upper_f < lower_g`.
- `g` robustly better than `f` if `upper_g < lower_f`.
- Otherwise ambiguous.

There are 5 algorithms and 5 seeds, hence 10 pairs per seed and 50 pairs per alpha.

| alpha | robustly ordered pairs | ambiguous pairs | total pairs |
|---:|---:|---:|---:|
| 0.5 | 0 | 50 | 50 |
| 0.75 | 0 | 50 | 50 |
| 0.9 | 0 | 50 | 50 |

This confirms that interval inclusion of deployment CVaR is not an informative validation criterion here; pairwise identified rankings are fully ambiguous under these intervals.

## K. Preference-Sensitivity Evaluation Details

Sources: `CMNIST/evaluate_lambda_predictions.py`, `CMNIST/analyze_regret.py`, and `CMNIST/job_scripts/bashes/run_lambda_prediction_eval.ps1`.

- Five deployment environments for the preference-sensitivity figure: `0.0, 0.1, 0.5, 0.9, 1.0`.
- Lambda grid: `0.0:1.0:0.1`, i.e. `[0.0, 0.1, ..., 1.0]`.
- Runner filters algorithms to `iro,inftask`, with `--max_checkpoints 2`, `--distinct_algorithms`, and CPU evaluation.
- Predictions are recomputed separately for each lambda for IRO and INF-TASK by passing an alpha tensor to `algorithm.predict(x, alpha)`. For non-conditional algorithms, if included, predictions would not depend on lambda.
- The same loaded checkpoint is used for all lambda values within `evaluate_checkpoint`.
- Prediction disagreement definitions:
  - `disagreement_from_lambda_0 = mean(prediction_lambda != prediction_lambda_0)` within the same environment/checkpoint.
  - `neighbor_disagreement = mean(prediction_lambda_i != prediction_lambda_{i-1})`; lambda 0 has neighbor disagreement 0.
- Same-checkpoint pseudo-regret definition from `CMNIST/analyze_regret.py`:
  - For each checkpoint/algorithm/seed, compute deployment mean accuracy for every lambda.
  - Oracle lambda is the same-checkpoint lambda with maximum deployment mean accuracy.
  - `deployment_wide_pseudo_regret = oracle_deployment_mean_accuracy - used_deployment_mean_accuracy`.
  - Per-env `pseudo_regret` stores this same deployment-wide regret value for each environment row.

The existing lambda audit coverage file confirms 11 lambda values where present.

## No-Retrain Decision

The synthetic reversal event was not implemented with the opposite orientation; it is now explicit. The principal neural analyses inspected here use final-derived fields where expected, with noted legacy best-default scripts to avoid. No neural retraining was performed.

## Refreshed Analysis Artifacts

After correcting the discrete CVaR helper, the affected analysis artifacts were regenerated on 2026-08-31 without retraining neural models.

Commands run:

```powershell
.\dgil_env\Scripts\python.exe CMNIST\priority7_theory.py --output_dir results\cmnist_priority7_theory_v1
.\dgil_env\Scripts\python.exe CMNIST\analyze_identification_width.py --output_dir results\cmnist_priority7_theory_v1
.\dgil_env\Scripts\python.exe -c "from run_posthoc_and_theory_sims import run_task4; run_task4()"
.\dgil_env\Scripts\python.exe CMNIST\analyze_tail_support.py results\E3b_tail_support\results --output_dir results\E3b_tail_support
.\dgil_env\Scripts\python.exe CMNIST\analyze_tail_support.py results\E3b_tail_support\results --output_dir results\E3b_tail_support\analysis_4anchor_seed0-4 --eval_envs 0.1,0.2,0.5,0.9
.\dgil_env\Scripts\python.exe CMNIST\analyze_tail_support.py results\E3b_tail_support\results --output_dir results\E3b_tail_support\analysis_seed0-4
```

Regenerated synthetic/theory outputs:

- `results/cmnist_priority7_theory_v1/ranking_reversal_summary.csv`
- `results/cmnist_priority7_theory_v1/trial_diagnostics.csv`
- `results/cmnist_priority7_theory_v1/ranking_reversal_heatmap.png`
- `results/cmnist_priority7_theory_v1/ranking_reversal_by_sample_size.png`
- `results/cmnist_priority7_theory_v1/identification_width_by_alpha.csv`
- `results/cmnist_priority7_theory_v1/identification_width_by_alpha.png`

Regenerated missing-mass ranking-reversal outputs from `results/cmnist_priority7_theory_v1/ranking_reversal_summary.csv`:

- `ranking_reversal_by_missing_mass.csv`
- `ranking_reversal_by_missing_mass.pdf`
- `ranking_reversal_by_missing_mass.png`
- `results/ranking_reversal_by_missing_mass.csv`
- `results/ranking_reversal_by_missing_mass.pdf`
- `results/ranking_reversal_by_missing_mass.png`
- `results_submit/additional/ranking_reversal_by_missing_mass.csv`
- `results_submit/additional/ranking_reversal_by_missing_mass.pdf`
- `results_submit/additional/ranking_reversal_by_missing_mass.png`
- `results_submit/tables/P7_theory/ranking_reversal_by_missing_mass.csv`
- `results_submit/figures/ranking_reversal_by_missing_mass.pdf`
- `results_submit/figures/ranking_reversal_by_missing_mass.png`
- `results_submit/figures/P7_theory/ranking_reversal_by_missing_mass.pdf`
- `results_submit/figures/P7_theory/ranking_reversal_by_missing_mass.png`

The refreshed missing-mass probabilities are `[0.636200, 0.944983, 0.998067, 0.999950]` for epsilon `[0.0, 0.1, 0.2, 0.3]`.

Refreshed synthetic/theory submission copies:

- `results_submit/tables/P7_theory/ranking_reversal_summary.csv`
- `results_submit/tables/P7_theory/identification_width_by_alpha.csv`
- `results_submit/figures/P7_theory/ranking_reversal_heatmap.png`
- `results_submit/figures/P7_theory/ranking_reversal_by_sample_size.png`
- `results_submit/figures/P7_theory/identification_width_by_alpha.png`

Regenerated 11-environment tail-support outputs:

- `results/E3b_tail_support/raw_results.csv`
- `results/E3b_tail_support/summary_by_condition.csv`
- `results/E3b_tail_support/slide_table.csv`
- `results/E3b_tail_support/tail_accuracy_by_condition.png`
- `results/E3b_tail_support/worst_accuracy_by_condition.png`
- `results/E3b_tail_support/head_tail_gap_by_condition.png`
- `results/E3b_tail_support/cvar_gap_by_condition.png`
- `results/E3b_tail_support/analysis_seed0-4/raw_results.csv`
- `results/E3b_tail_support/analysis_seed0-4/summary_by_condition.csv`
- `results/E3b_tail_support/analysis_seed0-4/slide_table.csv`
- `results/E3b_tail_support/analysis_seed0-4/tail_accuracy_by_condition.png`
- `results/E3b_tail_support/analysis_seed0-4/worst_accuracy_by_condition.png`
- `results/E3b_tail_support/analysis_seed0-4/head_tail_gap_by_condition.png`
- `results/E3b_tail_support/analysis_seed0-4/cvar_gap_by_condition.png`

Regenerated 4-anchor tail-support outputs used by the theorem-aligned check:

- `results/E3b_tail_support/analysis_4anchor_seed0-4/raw_results.csv`
- `results/E3b_tail_support/analysis_4anchor_seed0-4/summary_by_condition.csv`
- `results/E3b_tail_support/analysis_4anchor_seed0-4/slide_table.csv`
- `results/E3b_tail_support/analysis_4anchor_seed0-4/tail_accuracy_by_condition.png`
- `results/E3b_tail_support/analysis_4anchor_seed0-4/worst_accuracy_by_condition.png`
- `results/E3b_tail_support/analysis_4anchor_seed0-4/head_tail_gap_by_condition.png`
- `results/E3b_tail_support/analysis_4anchor_seed0-4/cvar_gap_by_condition.png`

Refreshed tail-support submission copies:

- `results_submit/tables/E3b_tail_support/raw_results.csv`
- `results_submit/tables/E3b_tail_support/summary_by_condition.csv`
- `results_submit/tables/E3b_tail_support/slide_table.csv`
- `results_submit/tables/E3b_tail_support/raw_results_4anchor.csv`
- `results_submit/tables/E3b_tail_support/summary_by_condition_4anchor.csv`
- `results_submit/tables/E3b_tail_support/slide_table_4anchor.csv`
- `results_submit/figures/E3b_tail_support/tail_accuracy_by_condition.png`
- `results_submit/figures/E3b_tail_support/worst_accuracy_by_condition.png`
- `results_submit/figures/E3b_tail_support/head_tail_gap_by_condition.png`
- `results_submit/figures/E3b_tail_support/cvar_gap_by_condition.png`
- `results_submit/figures/E3b_tail_support/tail_accuracy_by_condition_4anchor.png`
- `results_submit/figures/E3b_tail_support/worst_accuracy_by_condition_4anchor.png`
- `results_submit/figures/E3b_tail_support/head_tail_gap_by_condition_4anchor.png`
- `results_submit/figures/E3b_tail_support/cvar_gap_by_condition_4anchor.png`