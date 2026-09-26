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
.\dgil_env\Scripts\python.exe -m unittest discover -s IMAGENET100C/tests -v
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

## Execution status, seeds, and local GPU plan

The implementation, experiment design, and seed-0 training bundle are present
in this repository:

- Implementation and protocol: `IMAGENET100C/`
- Seed-sweep launcher: `IMAGENET100C/run_seed_all.ps1`
- Submission-style diagnostic bundle: `results_submit_img100/`
- Completion metadata: `results_submit_img100/metadata/completion_summary.json`

### Completed local run

The first full training sweep completed on the local GPU using
`finetune_last_stage`, a fixed 1,000-step budget, and the final checkpoint
policy. It covers **seed 0 only** and contains **64/64 completed training
runs**:

| Experiment | Configurations | Algorithms | Runs at one seed |
| --- | ---: | ---: | ---: |
| E0 baseline | 1 | 4 | 4 |
| E1 domain count | 2, 4, 8, 12 types | 4 | 16 |
| E2 sample support | 1,000, 5,000, 10,000 images/type | 4 | 12 |
| E3 visible imbalance | balanced, mild, strong | 4 | 12 |
| E3b tail support | balanced, long-tail, near-missing, missing | 4 | 16 |
| severity support | severities 1, 2, 3 observed | 4 | 4 |
| **Total** |  |  | **64** |

The algorithm set is `erm`, `groupdro`, `inftask`, and `iro`. IRM is not part
of this ImageNet-100-C implementation, so this extension must not be presented
as a five-method reproduction of the CMNIST tables.

The seed-0 bundle contains manifests, histories, final checkpoints, source-loss
summaries, and source-training diagnostic figures. It also contains a completed
**reduced pilot** for the 28 E3/E3b checkpoints: 100 class-balanced validation
images (one per class), the four registered anchors, and all five severities.
Those pilot outputs are explicitly named `evaluation_pilot_anchor100` and the
derived submission artifacts contain `pilot_anchor100` in their filenames.

No checkpoint has yet received the full 5,000-image, 15-type by 5-severity
deployment evaluation. Training diagnostics must not be described as held-out
performance, and pilot figures must not be described as report-grade or as
five-seed evidence.

### Run the local seed sweep

The launcher is resumable: it skips a run only when its `checkpoints/final.pt`
exists, and it stops if it finds an incomplete output directory. Run it from
the repository root after activating `dgil_env`:

```powershell
.\dgil_env\Scripts\python.exe -m unittest discover -s IMAGENET100C/tests -v

# Seed 0: results/imagenet100c_seed0 (64 training runs)
.\IMAGENET100C\run_seed_all.ps1 -Seed 0 -BackboneMode finetune_last_stage

# Later, use a separate output root per additional seed.
.\IMAGENET100C\run_seed_all.ps1 -Seed 1 -BackboneMode finetune_last_stage `
  -ResultsRoot results\imagenet100c_seed1
```

For an offline rerun after the dataset and weights are cached, add `-Offline`.
The current documented local device is an NVIDIA RTX A1000 6GB Laptop GPU.
Use the 100-image loader/training smoke test before launching a new seed or a
different machine configuration.

### Measured runtime and five-seed wall-clock budget

All timings in this section assume the machine used for seed 0: one NVIDIA RTX
A1000 6GB Laptop GPU, Windows, `finetune_last_stage`, batch size 64, zero data
loader workers, cached/offline ImageNet-100, and sequential execution. IRO uses
CPU activation offloading to fit its second-order adaptive-Beta calculation.
The larger IRO configurations used up to approximately 18.4 GB of host RAM.

Measured values and projections are intentionally separated:

| Work item | Measured time | Projected time | Status / interpretation |
| --- | ---: | ---: | --- |
| Seed-0 training, all 64 runs | 46 h 27 min calendar time | about 40-44 productive GPU hours for a clean rerun | Completed. Calendar time included an OOM retry, a cache-permission stall, and an explicit pause. |
| Four-anchor pilot, one checkpoint | 25.9 s | about 25-30 s/checkpoint | Measured on 100 class-balanced images, 4 anchors, 5 severities. |
| Four-anchor E3/E3b pilot, 28 checkpoints | about 12.5 min total | under 20 min | Completed. Single seed and high variance; not report-grade. |
| Full evaluation, one checkpoint | not yet run end-to-end | about 2-3 h | Extrapolated from a measured 56.6 s benchmark using 20 images over all 75 corruption conditions. |
| Plot/table rendering after evaluation JSONL exists | seconds | under a few minutes | Model-free aggregation; evaluation is the expensive stage. |

The full evaluation estimate is large because one checkpoint processes clean
validation plus `15 x 5 = 75` corrupted conditions, or approximately 380,000
validation-image instances at 5,000 images per condition. Corruptions are
generated online for every checkpoint and every lambda. The measured pilot
showed that corruption generation is a material bottleneck; GPU inference alone
does not determine runtime.

#### Training-only budget for five seeds

One seed contains 64 runs. Seeds `0-4` contain 320 runs. At approximately
40-44 productive hours per seed, five-seed training is projected to require:

```text
5 seeds x 40-44 h = 200-220 h = about 8.3-9.2 days sequentially
```

Seed 0 is already complete. Training the remaining four seeds on the same
single GPU is therefore approximately 160-176 hours, or 6.7-7.3 days, assuming
no failures or manual pauses.

The seed-0 training tree occupies approximately 19.5 GiB. Retaining equivalent
checkpoints, histories, and manifests for five seeds is approximately 97.5 GiB,
before logs or evaluation outputs. Submission figures and CSV tables are small
by comparison and raw checkpoints are not copied into `results_submit_img100`.

#### Targeted CMNIST-equivalent figure/table budget

The closest ImageNet-100-C counterpart to the principal CMNIST submission
figures uses E1, E3, and E3b at lambda 0, plus an E4 lambda-sensitivity study on
one IRO and one INF-TASK checkpoint per seed:

| Component | Checkpoint-equivalent full evaluations per seed | Five-seed projection |
| --- | ---: | ---: |
| E1 domain count | 16 | 160-240 h |
| E3 visible imbalance | 12 | 120-180 h |
| E3b tail support and identification | 16 | 160-240 h |
| E4 extra lambda values | 20 | 200-300 h |
| **Targeted evaluation total** | **64** | **640-960 h = 26.7-40 days** |

The E4 row counts only the ten additional lambda values after lambda 0 for two
selected conditional checkpoints: `2 checkpoints x 10 extra lambdas = 20`
checkpoint-equivalent passes per seed. The synthetic ranking-reversal figure is
model-free and takes seconds to minutes; it is not included in the GPU total.

Combining five-seed training with this targeted report evaluation gives a
current-code wall-clock estimate of approximately:

```text
8.3-9.2 training days + 26.7-40 evaluation days
= about 35-49 days, or roughly 5-7 weeks sequentially
```

Starting from the already completed seed-0 training run, the remaining work is
approximately 33-47 days under the same assumptions. This estimate includes
all five seeds in evaluation because the existing pilot cannot substitute for
5,000-image report-grade metrics.

#### Explicit remaining schedule from the current state

The current state is: seed 0 has all 64 training checkpoints, and seed 0 has
only the reduced four-anchor E3/E3b pilot evaluation. Seeds 1-4 have not been
trained. The following estimates make every remaining stage explicit.

For **one additional seed**, training and evaluation are projected as follows:

| Stage for one new seed | Work | Projected time |
| --- | ---: | ---: |
| Train all E0/E1/E2/E3/E3b/severity configurations | 64 training runs | 40-44 h |
| Full lambda-0 evaluation for E1/E3/E3b only | 44 checkpoint passes | 88-132 h |
| E4 on one IRO and one INF-TASK checkpoint | 20 additional lambda passes | 40-60 h |
| **Targeted train + evaluation total per new seed** | 64 training runs + 64 evaluation-equivalent passes | **168-236 h = 7.0-9.8 days** |
| Full lambda-0 evaluation for every configuration | 64 checkpoint passes | 128-192 h |
| **Exhaustive train + evaluation + E4 total per new seed** | training + 84 evaluation-equivalent passes | **208-296 h = 8.7-12.3 days** |

For the **four untrained seeds 1-4**, the stages are:

| Remaining seeds 1-4 stage | Calculation | Projected time |
| --- | ---: | ---: |
| Train all four seeds | `4 x 40-44 h` | 160-176 h = 6.7-7.3 days |
| Targeted E1/E3/E3b lambda-0 evaluation | `4 x 44 x 2-3 h` | 352-528 h = 14.7-22 days |
| E4 extra lambdas | `4 x 20 x 2-3 h` | 160-240 h = 6.7-10 days |
| **Seeds 1-4 targeted train + evaluation** | sum of the three rows above | **672-944 h = 28-39.3 days** |
| Exhaustive lambda-0 evaluation of all 64 checkpoints | `4 x 64 x 2-3 h` | 512-768 h = 21.3-32 days |
| **Seeds 1-4 exhaustive train + evaluation + E4** | training + exhaustive evaluation + E4 | **832-1,184 h = 34.7-49.3 days** |

Seed 0 still needs report-grade validation/evaluation even though its training
is complete:

| Remaining seed-0 stage | Projected time |
| --- | ---: |
| Targeted E1/E3/E3b lambda-0 evaluation | 88-132 h = 3.7-5.5 days |
| E4 extra lambdas | 40-60 h = 1.7-2.5 days |
| **Seed-0 targeted evaluation total** | **128-192 h = 5.3-8 days** |
| Exhaustive lambda-0 evaluation of all 64 checkpoints | 128-192 h = 5.3-8 days |
| **Seed-0 exhaustive evaluation plus E4** | **168-252 h = 7-10.5 days** |

Therefore, starting from the current repository state:

```text
Targeted five-seed report path:
  seed 0 targeted evaluation       128-192 h
  seeds 1-4 train + evaluation     672-944 h
  aggregation/plotting             minutes
  total remaining                  800-1,136 h = 33.3-47.3 days

Exhaustive five-seed path:
  seed 0 full evaluation + E4      168-252 h
  seeds 1-4 train + evaluation     832-1,184 h
  aggregation/plotting             minutes
  total remaining                  1,000-1,436 h = 41.7-59.8 days
```

The stages can be ordered seed-by-seed (train seed 1, evaluate seed 1, then
continue) or phase-by-phase (train seeds 1-4, then evaluate all seeds). On one
GPU the total sequential time is effectively the same. Seed-by-seed execution
produces usable intermediate results earlier and limits the number of
unevaluated checkpoints waiting on disk.

#### Faster five-seed pilot-only alternative

If the immediate goal is only a five-seed version of the existing four-anchor
E3/E3b pilot figures, the expensive full evaluation can be deferred. Assuming
all 64 configurations are still trained for each new seed:

```text
train seeds 1-4                         160-176 h
repeat the 28-checkpoint pilot 4 times  about 50 min total
aggregate five-seed pilot figures       under a few minutes
total remaining                         about 161-177 h = 6.7-7.4 days
```

This fast path can provide five-seed pilot means and standard deviations, but
it remains based on one validation image per class and four anchors. It cannot
support claims about all 15 corruption types, full-validation accuracy, or
report-grade CVaR. It should retain `pilot_anchor100` in every filename and
caption.

#### Exhaustive all-configuration budget

Evaluating all 64 checkpoints at lambda 0 costs approximately 128-192 hours per
seed. Across five seeds that is 640-960 hours, or 26.7-40 days. Adding the two-
checkpoint E4 grids contributes another 200-300 hours across five seeds.
Together with five-seed training, an exhaustive end-to-end run is approximately
1,040-1,480 hours, or **43-62 days (about 6-9 weeks)** of sequential wall time.

These projections are deliberately conservative and apply to the current
online-corruption evaluator. Caching each corrupted validation image once and
reusing it across models, or distributing checkpoints across multiple GPUs,
could reduce elapsed time substantially. Neither optimization is implemented
in the report-grade path yet, so it must not be assumed when scheduling runs.
Running multiple evaluation processes concurrently on the same 6GB GPU is not
recommended without a separate memory and throughput benchmark.

#### What the completed pilot does and does not provide

The completed pilot took about 12.5 minutes for all 28 E3/E3b checkpoints and
produced:

- designated-tail accuracy across E3b conditions;
- worst-anchor accuracy across E3b conditions;
- the missing-support alpha-0.9 identification interval panel;
- visible-imbalance worst-anchor accuracy for E3; and
- corresponding CSV tables in `results_submit_img100`.

It uses seed 0 only, one validation image per class, and only the four registered
anchors. Consequently it has no cross-seed standard deviation, has high sampling
variance, and its worst-domain metric means worst among four anchors rather than
worst among all 15 ImageNet-C corruption types.

### Seed policy and report-grade status

Seed 0 establishes that the full configuration matrix trains successfully. It
is not sufficient for mean/standard-deviation reporting. For report-grade E1,
E2, E3, E3b, and severity-support comparisons, run at least seeds `0,1,2`
under the same backbone, fixed-step budget, and final-checkpoint policy. Use
seeds `0` through `4` only when the additional precision is needed and the
local-GPU budget permits it.

After training each checkpoint, run `IMAGENET100C.evaluate` once at lambda 0
for deployment results. For IRO and INF-TASK, additionally run the E4 lambda
grid on the same final checkpoint. Then pass all evaluation JSONL files for an
experiment to `IMAGENET100C.analyze`.

This produces the raw deployment fields for clean accuracy, 15-by-5 corruption
metrics, mean/worst/CVaR domain error, and identification intervals. It does
not yet automatically produce multi-seed mean/SD report tables or the CMNIST-
style Figures 2-6; those require a report aggregation/plotting layer over the
completed evaluation files. The synthetic ranking-reversal Figure 1 is a
separate theory simulation and is not an ImageNet-100-C experiment output.

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

