# [ImgC] ImageNet-C Extension Plan and Status

This directory contains the separate ImageNet-C extension. The synthetic smoke pipeline is implemented and has produced artifacts; the real-data path remains a scaffold and has not produced report-grade results.

## Goals

- Add an ImageNet-C extension without changing CMNIST behavior.
- Keep all new outputs under fresh `results/imagenet_c_*` roots.
- Start with deterministic evaluation, then held-out corruption training, then support-stress experiments.

## Planned Folder Layout

| Path | Purpose |
| --- | --- |
| `IMAGENET_C/README.md` | This planning document |
| `IMAGENET_C/datasets.py` | Dataset wrappers for clean ImageNet and ImageNet-C corruptions |
| `IMAGENET_C/features.py` | Frozen backbone feature extraction utilities |
| `IMAGENET_C/models.py` | Trainable 1000-class head and optional lambda-conditioned head |
| `IMAGENET_C/train_head.py` | Main training entry point for ERM, GroupDRO, and IRO on frozen features |
| `IMAGENET_C/eval_repeatability.py` | Deterministic evaluation-only pipeline for Experiment Set 1; currently supports a synthetic smoke mode |
| `IMAGENET_C/evaluate_lambda_grid.py` | Post-training lambda-grid evaluation on saved checkpoints |
| `IMAGENET_C/analyze_imagenet_c.py` | CSV export, aggregation, and plot generation |
| `IMAGENET_C/folds.json` | Fixed held-out corruption fold specification |
| `IMAGENET_C/job_scripts/` | Generated command manifests for pilot and full runs |

The table above is now partially implemented:

- implemented: `README.md`, `datasets.py`, `features.py`, `models.py`, `train_head.py`, `eval_repeatability.py`, `evaluate_lambda_grid.py`, `analyze_imagenet_c.py`, `folds.json`
- still pending: any real-data job-script generation under `IMAGENET_C/job_scripts/`

## [ImgC] Fixed Fold Specification

The held-out corruption folds should be fixed and versioned rather than redefined inside ad hoc scripts.

| Fold | Held-Out Corruptions |
| --- | --- |
| `fold_a` | `gaussian_noise`, `defocus_blur`, `glass_blur`, `snow`, `contrast` |
| `fold_b` | `shot_noise`, `motion_blur`, `frost`, `fog`, `elastic_transform` |
| `fold_c` | `impulse_noise`, `zoom_blur`, `brightness`, `pixelate`, `jpeg_compression` |

## [ImgC] Planned and Existing Output Roots

Use a new root for each phase. Do not reuse an existing root if it already contains files.

| Phase | Output Root |
| --- | --- |
| Smoke validation | `results/imagenet_c_eval_repeatability_smoke_v1/` |
| Experiment Set 1 | `results/imagenet_c_eval_repeatability_v1/` |
| Experiment Set 2 pilot | `results/imagenet_c_fold_generalization_v1/` |
| Experiment Set 3 pilot | `results/imagenet_c_support_stress_v1/` |

If a root already exists, increment the suffix (`_v2`) or append a date stamp instead of overwriting files.

## [ImgC] Planned Artifact Structure

### Experiment Set 1: Evaluation Repeatability

| Relative Path | Purpose |
| --- | --- |
| `raw/repetition_0.jsonl` | Per-condition metrics for repetition 0 |
| `raw/repetition_1.jsonl` | Per-condition metrics for repetition 1 |
| `raw/repetition_2.jsonl` | Per-condition metrics for repetition 2 |
| `summary/repeatability_summary.csv` | Aggregate comparison across the three repetitions |
| `summary/corruption_severity_matrix.csv` | Accuracy/loss matrix over corruption x severity |
| `plots/corruption_severity_heatmap.png` | Heatmap over 15 corruptions and 5 severities |
| `plots/corruption_accuracy_bar.png` | Mean accuracy per corruption type |
| `plots/lambda_sensitivity_curve.png` | Metric variation over lambda grid |
| `manifests/eval_repeatability_repetitions.txt` | Exact evaluation commands executed |

### Experiment Set 2: Held-Out Corruption Generalization

| Relative Path | Purpose |
| --- | --- |
| `raw/train_runs.jsonl` | One row per completed training run |
| `raw/lambda_eval.jsonl` | Post-training lambda-grid results |
| `summary/by_fold_seed.csv` | Metrics grouped by fold and seed |
| `summary/by_fold_algorithm.csv` | Mean/std across seeds per fold and algorithm |
| `summary/held_out_only.csv` | Primary held-out corruption results only |
| `summary/all_corruptions.csv` | Secondary evaluation on all 15 corruptions |
| `plots/held_out_worst_domain.png` | Worst held-out corruption metric by fold |
| `plots/held_out_mean_accuracy.png` | Mean held-out corruption accuracy by fold |
| `plots/severity_profile_by_algorithm.png` | Severity-wise curves |
| `plots/corruption_severity_heatmap_<algorithm>_<fold>.png` | Corruption x severity heatmap |
| `plots/lambda_sensitivity_<fold>.png` | Fold-level lambda sensitivity plot |
| `manifests/fold_a_seed0_smoke.txt` | Smoke-test command list |
| `manifests/pilot_seed012.txt` | Pilot command list |
| `manifests/final_seed01234.txt` | Final command list |

### Experiment Set 3: Corruption-Support Stress

| Relative Path | Purpose |
| --- | --- |
| `raw/train_runs.jsonl` | One row per completed stress-test run |
| `raw/lambda_eval.jsonl` | Lambda-grid outputs |
| `summary/by_condition_seed.csv` | Condition-level metrics by seed |
| `summary/by_condition_algorithm.csv` | Mean/std across seeds by condition and algorithm |
| `plots/tail_accuracy_by_condition.png` | Tail-domain accuracy comparison |
| `plots/worst_accuracy_by_condition.png` | Worst-domain comparison |
| `plots/head_tail_gap_by_condition.png` | Head-tail gap comparison |
| `plots/cvar_gap_by_condition.png` | CVaR gap relative to balanced condition |
| `plots/lambda_tail_accuracy.png` | IRO lambda curve for tail accuracy |
| `plots/lambda_cvar.png` | IRO lambda curve for aggregated risk |
| `manifests/pilot_seed012.txt` | Pilot command list |
| `manifests/final_seed01234.txt` | Final command list |

## Planned Run Naming Convention

| Field | Convention |
| --- | --- |
| Dataset tag | `imagenet_c` |
| Experiment tags | `eval_repeatability`, `fold_generalization`, `support_stress` |
| Fold tags | `fold_a`, `fold_b`, `fold_c` |
| Condition tags | `balanced_visible`, `long_tail_visible`, `near_missing_tail`, `missing_tail` |
| Seed tags | `seed0`, `seed1`, `seed2`, `seed3`, `seed4` |
| Algorithm tags | `erm`, `groupdro`, `iro` |

Recommended run ID pattern:

`imagenet_c__<experiment>__<fold_or_condition>__<algorithm>__seed<k>`

Examples:

- `imagenet_c__fold_generalization__fold_a__iro__seed0`
- `imagenet_c__support_stress__missing_tail__groupdro__seed2`

## [ImgC] Implementation Status and Checklist

1. [x] Add fixed fold specification file (`folds.json`).
2. [x] Implement synthetic deterministic evaluation smoke pipeline.
3. [x] Run repeatability smoke validation and verify its artifact layout.
4. [ ] Confirm repeated evaluations in the real-data path.
5. [x] Implement the frozen-feature/head scaffold.
6. [x] Implement one-fold synthetic smoke test (`fold_a`, `seed0`) for ERM, GroupDRO, and IRO.
7. [x] Add post-training lambda-grid smoke evaluation.
8. [ ] Complete real-data held-out-corruption analysis before scaling to 3 seeds.
9. [ ] Implement and run the support-stress experiment.
