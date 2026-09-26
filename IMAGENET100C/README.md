# ImageNet-100-C domain-generalization experiments

This directory is a self-contained ImageNet-100 experiment suite. It transfers
the intent of the CMNIST E0-E4 domain-count, sample-support, imbalance, missing-
support, and preference-sensitivity studies to 100-class natural images.

It does **not** modify CMNIST and does not claim that clean ImageNet-100 has
natural domains. Domains here are constructed with the 15 standard ImageNet-C
corruption mechanisms.

## Dataset contract

The loader uses `clane9/imagenet-100` at the immutable revision recorded in
`configs/experiments.json`.

- ImageNet-100 `train` supplies source training and a non-overlapping,
  source-only checkpoint-selection subset.
- ImageNet-100 `validation` is final evaluation only. It is never used for
  training, tuning, early stopping, or checkpoint selection.
- Labels remain integer ImageNet-100 class IDs in `[0,99]`; the task is never
  binarized.
- Images are converted to RGB and processed by the explicit
  `ResNet50_Weights.IMAGENET1K_V2` transform. The mutable `DEFAULT` alias is not
  used.
- Every run writes `manifest.json`, including the dataset revision, full class
  mapping, exact train/source-validation indices, corruption configuration,
  per-environment and per-class counts, seeds, and transform specification.

The Hugging Face dataset is approximately 8.4 GB. Install the optional suite
dependencies with:

```powershell
python -m pip install -r IMAGENET100C/requirements.txt
```

## Two protocols

### Primary: corruption-mechanism support

A domain is one corruption type. Within a source type, severity is sampled
uniformly from `1..5`. Sampling and corruption randomness use two distinct
hashes:

```text
severity = 1 + hash(global_seed, split, image_index, corruption_name, epoch) % 5
corruption_rng_seed = hash(global_seed, split, image_index, corruption_name, severity, epoch)
```

Source assignments are class-stratified and non-overlapping. Unobserved types
are missing-support deployment domains. E0, E1, E2, E3, and E3b use this
protocol.

This protocol supports claims about transfer across missing corruption
mechanisms. It does not establish robustness to naturally occurring domains,
geographies, acquisition devices, or populations.

### Secondary: severity support

A domain is a severity. Corruption type is sampled uniformly from all 15 types
inside each domain. Severities `1,2,3` are observed; `4,5` are held out. Severity
4 is never used for checkpoint selection.

This protocol supports a severity-extrapolation claim. It must not be described
as missing-corruption-type generalization.

Both protocols finally evaluate clean validation and every one of the `15 x 5`
corruption-type/severity conditions.

## Experiments

- **E0:** four-source smoke/pilot with ERM, GroupDRO, INF-TASK, and IRO.
- **E1:** nested source-type counts `2,4,8,12`, fixed total budget 40,000.
- **E2:** fixed four types with `1,000`, `5,000`, or `10,000` images/type.
- **E3:** balanced, mild, and strong visible-frequency imbalance.
- **E3b:** balanced, long-tail, near-missing, and missing support, each with a
  total budget of 40,000.
- **E4:** the same conditional checkpoint evaluated at lambda
  `0,0.1,...,1`; target performance is not used to select lambda.
- **severity_support:** the separate severity-extrapolation protocol.

Zero-count environments are recorded in `requested_domain_counts` but omitted
from `active_domain_counts` and from data loaders. ERM receives per-environment
batch sizes proportional to configured counts. GroupDRO, INF-TASK, and IRO
calculate a mean loss for every active environment.

The E1 type sets are pre-registered and nested. The first four contain one
noise (`gaussian_noise`), one blur (`defocus_blur`), one weather (`snow`), and
one digital (`contrast`) corruption. The next groups expand representation
across those families where the standard ImageNet-C family sizes permit.

> For E3b identification analysis, define a pre-registered four-type anchor set and use deployment weights 1/4 on each anchor type. Balanced, long-tail, near-missing, and missing conditions vary only the source counts for these four anchors. In the missing condition, epsilon=1/4. Evaluation over all 15 types remains a separate external robustness analysis with its own uniform 15-type deployment law.

The registered anchors are `gaussian_noise`, `defocus_blur`, `snow`, and
`contrast`. Identification uses 0-1 risks measured on the held-out
source-validation subset of the ImageNet-100 training split. Final ImageNet-100
validation results supply only the realized deployment CVaR, explicitly labeled
as a descriptive plug-in result.

The primary robustness CVaR is always computed from 15 severity-averaged
corruption-type losses under uniform 15-type deployment weights. Aggregation
over the 75 individual type/severity conditions is supplementary. At high
alpha, identification intervals can be vacuous: whenever missing mass
`epsilon >= 1 - alpha`, the unrestricted upper endpoint is 1. This is expected,
particularly at `alpha=0.9`, and is flagged in every interval record.

## Models and the pretrained-backbone caveat

`frozen_feature_pilot` freezes the full ResNet-50 backbone and trains a
lambda-conditioned FiLM classifier. It is a pretrained-feature robustness
study: the backbone was trained on ImageNet-1k, which contains the ImageNet-100
classes and training images. It is not an end-to-end representation-learning
result.

`finetune_last_stage` unfreezes ResNet layer 4 and trains it together with the
conditional classifier. This is the stronger main setting, but it remains
ImageNet-1k-pretrained transfer learning—not training a representation from
scratch.

ERM and GroupDRO call the conditional network at lambda 0. INF-TASK samples
preferences from `Beta(1,1)`. IRO ports the CMNIST adaptive-Beta update and then
samples preferences from the updated distribution. `num_lambda_samples` is an
explicit parameter with supported ablation values `2,4,8`. An average over
fixed CVaR levels is not labeled IRO anywhere in this suite.

## Validation order and commands

Run unit checks first; they do not download ImageNet-100:

```powershell
python -m unittest discover -s IMAGENET100C/tests -v
```

Run the required 100-source-image loader/training smoke test:

```powershell
python -m IMAGENET100C.train `
  --experiment E0 --algorithm erm --backbone_mode frozen_feature_pilot `
  --max_source_images 100 --steps 2 --eval_every 1 --batch_size 16 `
  --output_dir results/imagenet100c_e0_loader_smoke_seed0
```

Run one-seed E0, first ERM versus GroupDRO and then the conditional methods:

```powershell
python -m IMAGENET100C.train --experiment E0 --algorithm erm      --seed 0 --output_dir results/imagenet100c_e0_erm_seed0
python -m IMAGENET100C.train --experiment E0 --algorithm groupdro --seed 0 --output_dir results/imagenet100c_e0_groupdro_seed0
python -m IMAGENET100C.train --experiment E0 --algorithm inftask --seed 0 --num_lambda_samples 4 --output_dir results/imagenet100c_e0_inftask_seed0
python -m IMAGENET100C.train --experiment E0 --algorithm iro      --seed 0 --num_lambda_samples 4 --output_dir results/imagenet100c_e0_iro_seed0
```

Use `--backbone_mode finetune_last_stage` for the main scientific runs. The
small frozen pilot should pass before starting those runs.

Run an E3b condition:

```powershell
python -m IMAGENET100C.train `
  --experiment E3b --condition missing --algorithm iro `
  --backbone_mode finetune_last_stage --seed 0 --num_lambda_samples 4 `
  --output_dir results/imagenet100c_e3b_missing_iro_seed0
```

Run nested E1 or E2 configurations by changing the relevant single factor:

```powershell
python -m IMAGENET100C.train --experiment E1 --source_count 8 --algorithm groupdro --seed 0 --output_dir results/imagenet100c_e1_8types_groupdro_seed0
python -m IMAGENET100C.train --experiment E2 --samples_per_type 5000 --algorithm groupdro --seed 0 --output_dir results/imagenet100c_e2_5000_groupdro_seed0
```

Repeat report-grade E3b/E1/E2 runs with at least seeds `0,1,2`.

Training uses a fixed step budget and the **final checkpoint** by default. This
matches the newer paper draft and performs no checkpoint selection. The file is
`checkpoints/final.pt`. An optional, explicitly named alternative is available:

```powershell
--checkpoint_selection source_val_uniform_lambda_cvar
```

It selects `best_source_val.pt` using one fixed source-only objective: the mean,
over the predeclared lambda grid `0,0.1,...,1`, of source-domain CVaR
cross-entropy at the corresponding lambda. It never accesses ImageNet-100
validation. Do not mix final- and source-selected checkpoints in one comparison.

Final 75-condition evaluation:

```powershell
python -m IMAGENET100C.evaluate `
  results/imagenet100c_e0_iro_seed0/checkpoints/final.pt `
  --output_dir results/imagenet100c_e0_iro_seed0/evaluation_lambda0
```

E4 evaluates the same checkpoint, without retraining or target-based selection:

```powershell
python -m IMAGENET100C.evaluate `
  results/imagenet100c_e0_iro_seed0/checkpoints/final.pt `
  --lambda_grid 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1 `
  --output_dir results/imagenet100c_e0_iro_seed0/evaluation_e4
```

Aggregate completed evaluations:

```powershell
python -m IMAGENET100C.analyze `
  results/imagenet100c_e0_iro_seed0/evaluation_e4/evaluation.jsonl `
  --output_dir results/imagenet100c_analysis_v1
```

Every output directory must be new. The tools intentionally refuse to write
into an existing run or analysis directory.

## Artifacts and metrics

A training directory contains:

```text
manifest.json
history.jsonl
checkpoints/final.pt
```

`checkpoints/best_source_val.pt` is additionally present only when the optional
source-validation selection policy was requested.

An evaluation directory contains `evaluation.jsonl`, with:

- clean top-1, top-5, and cross-entropy;
- per-type/per-severity top-1, top-5, and cross-entropy;
- severity-averaged results for each corruption domain;
- mean, worst, and CVaR domain error at alpha `0.5,0.75,0.9`;
- the worst individual type/severity condition;
- mean accuracy by severity;
- exact environment and class counts;
- the explicit uniform deployment law;
- bounded 0-1 partial-identification intervals, widths, and epsilon;
- realized held-out CVaR labeled as a descriptive plug-in result.

`analyze.py` writes `summary.csv` and pairwise `robust_ranking.csv`. An algorithm
is robustly ranked below another in loss only when their identification
intervals are disjoint. Overlapping intervals are reported as
`not_robustly_ranked`.

