# Dataset Setup Comparison: [CMN] and [ImgC]

This document compares the dataset setups that are either already present in this repository or have been discussed as extension candidates.

Status labels used below:

- Existing: code or experiment assets already exist in this repository.
- Planned: design or scaffold exists, but the corresponding report-grade experiment is not complete.

## Existing Repository Datasets

| Dataset | Status | Task Type | Label Space | Domain / Environment Definition | Train / Test Setup | Shift Mechanism | Binary Labeling | Current Entry Points |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CMNIST | Existing | Classification | Binary | Each environment is defined by a color-label correlation parameter `e` | Multiple source environments for training; multiple target environments for evaluation | Spurious correlation changes across environments | Yes. Digits `0-4 -> 1`, `5-9 -> 0`, then label noise and color flips are applied | `CMNIST/datasets.py`, `CMNIST/train_sandbox.py`, `CMNIST/evaluate_lambda_grid.py`, `CMNIST/analyze_tail_support.py` |
| UCI Bike Rental | Existing | Regression | Continuous target (`cnt`) | Each environment is a `(season, year)` combination | First 4 environments used for training, remaining 4 for testing | Distribution shift across season and year combinations | No. This setup is naturally regression, not binary classification | `UCI-Bike-Rental/data.py`, `UCI-Bike-Rental/iro.py`, `UCI-Bike-Rental/final.ipynb` |

## CMNIST Details

| Aspect | Current Repo Setup |
| --- | --- |
| Base images | Standard MNIST digit images |
| Target construction | Binary target from original 10-way labels |
| Exact mapping | `labels = (labels < 5).float()` so digits `0-4` map to `1` and digits `5-9` map to `0` |
| Label noise | Labels are flipped with probability `0.25` |
| Domain variable | Environment parameter `e` controlling color-label mismatch |
| Observed shift | Color is spuriously correlated with the binary label, and that correlation changes by environment |
| Why binary works here | The benchmark is intentionally simplified so the environment shift is the main difficulty |

## Proposed Extension Datasets

| Dataset | Status | Task Type | Label Space | Natural Domain Definition | Typical Measurement Structure | Binary Labeling | Fit With Current CMNIST Pipeline | Recommended First Use |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ImageNet-C | Planned/scaffolded | Classification | 1000-class | Corruption type, with severity as a secondary axis | 50 validation images per class, repeated across 15 corruptions and 5 severities | Usually no. Binary relabeling is possible only with a defensible superclass split | Low direct fit. Reuse CVaR-style evaluation, but not the CMNIST data-generation logic | Synthetic smoke validation completed; real-data evaluation pending |
| ImageNet-R | Proposed | Classification | 200-class subset of ImageNet classes | Domain shift from renditions / artistic style | Different dataset, not corruption copies of the same validation images | Usually no. Keep original multiclass task unless a strong binary grouping is justified | Low direct fit. Better treated as a separate dataset loader and evaluation path | Hold-out style robustness evaluation |
| ImageNet-O | Proposed | OOD / robustness evaluation | Not a standard in-distribution classification setup | Out-of-distribution image set | Separate OOD-style benchmark, not a standard DG training environment split | No for most uses | Very low direct fit to current CMNIST training scripts | Use only if the goal is explicit OOD analysis, not standard domain generalization |

## ImageNet-C Structure

| Quantity | Value |
| --- | --- |
| Number of classes | 1000 |
| Clean ImageNet validation images per class | 50 |
| Corruption types | 15 |
| Severity levels | 5 |
| Images per class for one corruption at one severity | 50 |
| Images per class for one corruption across all severities | 250 |
| Total corrupted images per class across all corruptions and severities | 3750 |

This means ImageNet-C is not a single-measurement-per-class setup. Each class is observed repeatedly under many corruption conditions.

## Binary Labeling Guidance

| Dataset | Should You Reuse CMNIST-Style Binary Labeling? | Reason |
| --- | --- | --- |
| CMNIST | Yes | Binary relabeling is part of the benchmark design |
| UCI Bike Rental | No | The task is regression, so binary relabeling would change the problem entirely |
| ImageNet-C | Usually no | The native task is 1000-class classification; arbitrary binarization weakens the benchmark |
| ImageNet-R | Usually no | The native task is multiclass classification on rendition shift |
| ImageNet-O | No | This is better framed as OOD evaluation than binary classification |

## Recommended Extension Strategy

| Priority | Dataset | Suggested First Experiment | Seeds |
| --- | --- | --- | --- |
| 1 | ImageNet-C | Evaluate a pretrained model across a small set of corruptions and compute average, worst-domain, and CVaR-style metrics | 1 seed is enough if there is any random subsampling; otherwise seed is mostly irrelevant for pure evaluation |
| 2 | ImageNet-R | Evaluate a pretrained model on rendition shift as a separate held-out dataset | 1 seed |
| 3 | ImageNet-O | Only add if you want an explicit OOD benchmark rather than a standard DG benchmark | 1 seed |

## Adopted ImageNet-C Plan

| Aspect | Adopted Direction |
| --- | --- |
| Status | Separate [ImgC] scaffold implemented; only synthetic smoke runs completed |
| Task | Native 1000-class ImageNet classification |
| Primary domain variable | Corruption type |
| Secondary analysis variable | Severity level `1-5` |
| Total evaluation conditions | `15 x 5 = 75` |
| Backbone | Frozen pretrained ResNet-50 |
| Head | Trainable 1000-class classifier, with optional lambda-conditioned variant for post-training evaluation |
| Core algorithms | `ERM`, `GroupDRO`, `IRO` |
| First validation step | Synthetic deterministic repeatability completed; real fixed-checkpoint repeatability pending |
| Main training study | Planned held-out corruption generalization across 3 fixed folds; current trainer accepts one fold per invocation |
| Stress follow-up | Four-domain corruption-support stress mirroring CMNIST tail-support logic |
| Seed policy | `0-2` pilot, expand to `0-4` only after pilot interpretation is stable |
| Output policy | Use fresh `results/imagenet_c_*` roots and never overwrite existing CMNIST / E3b artifacts |

## Stability Criteria For ImageNet-C

| Stability Level | Meaning | Required Evidence |
| --- | --- | --- |
| Evaluation repeatability | Same checkpoint, same data, repeated evaluation | Identical or numerically negligible differences across 3 repeated runs |
| Training stability across seeds | Same conclusion across stochastic training runs | Mean, standard deviation, individual seed values, and count of seeds beating baseline |
| Domain and severity stability | Robustness is not isolated to one or two favorable corruptions | Average, worst-case, CVaR, severity-wise analysis, heatmap, and improvement/degradation counts by corruption |

## Practical Difference From Existing CMNIST Experiments

| Aspect | CMNIST | ImageNet-C Extension |
| --- | --- | --- |
| Training setup | Multi-environment training benchmark | Best started as evaluation-only |
| Domain definition | Synthetic color-label correlation | Corruption type and/or severity |
| Label space | Binary by design | Multiclass by default |
| Reusable component | IRO / CVaR-style aggregation ideas | Same aggregation ideas can be reused |
| Non-reusable component | CMNIST dataset generation and job scripts | Needs a separate loader and evaluation pipeline |

## Source Anchors In This Repository

- CMNIST binary labeling and environment construction: `CMNIST/datasets.py`
- CMNIST training / evaluation entry point: `CMNIST/train_sandbox.py`
- CMNIST stress plots and exports: `CMNIST/plot_domain_stress.py`, `CMNIST/export_results_csv.py`, `CMNIST/analyze_tail_support.py`
- UCI Bike Rental environment construction: `UCI-Bike-Rental/data.py`
- UCI Bike Rental IRO-style code: `UCI-Bike-Rental/iro.py`
