# ImageNet-100-C external replication

This directory is an optional natural-image supplement to the AISTATS theory,
synthetic ranking-reversal, and Colored-MNIST evidence. It does not alter the
CMNIST experiments and does not claim that clean ImageNet-100 has natural
domains. Domains are constructed from ImageNet-C corruption mechanisms.

The current paper claim does not require a full ImageNet-C benchmark. The
ImageNet result should be labelled **external constructed-domain replication**
in the supplement, or omitted from the submission if it delays the core paper.
It must not be presented as a new central theorem test or as a five-method
reproduction of the CMNIST tables.

## Focused scientific question

The retained ImageNet question is whether the support-removal pattern survives
on natural images:

- observed corruption mechanisms: `gaussian_noise`, `defocus_blur`, `snow`,
  and `contrast`;
- deployment law: uniform over those four registered anchors;
- source conditions: `balanced`, `long_tail`, and `missing`;
- algorithms: ERM, GroupDRO, and IRO;
- seeds: `0,1,2`;
- evaluation: 1,000 fixed class-stratified validation images per
  type/severity condition, shared across models and seeds.

This is `3 conditions x 3 algorithms x 3 seeds = 27 checkpoints`. Each
checkpoint evaluates the four anchors at severities `1..5`, with severity
averaging within each anchor. Clean validation is reported separately.

The primary report contains per-anchor error, domain CVaR at
`alpha = 0.5, 0.75, 0.9`, the partial-identification interval, interval width,
and whether pairwise rankings are certified. The held-out ImageNet validation
results are descriptive plug-in deployment results; identification intervals
use the held-out source-validation subset declared in each manifest.

The `near_missing` condition is dropped. INF-TASK is dropped from this focused
extension. E1 domain-count, E2 sample-support, E3 visible-imbalance,
`severity_support`, and E4 lambda-grid studies are stopped as paper runs.
They remain available as internal controls if a reviewer specifically requests
them.

## Protocol and dataset contract

The loader uses `clane9/imagenet-100` at the immutable revision in
`configs/experiments.json`.

- ImageNet-100 `train` supplies source training and a non-overlapping,
  source-only validation subset.
- ImageNet-100 `validation` is final evaluation only.
- Labels remain integer class IDs in `[0,99]`; the task is never binarized.
- Images are converted to RGB and use the explicit
  `ResNet50_Weights.IMAGENET1K_V2` transform.
- Each run writes its dataset revision, class mapping, assignments, counts,
  seeds, corruption configuration, and transform specification to
  `manifest.json`.

For the focused extension, source domains are the four registered anchors.
The missing condition has one zero-count anchor and therefore missing mass
`epsilon = 1/4`; zero-count environments are omitted from data loaders but
remain recorded in requested counts. ERM receives proportional per-environment
batch sizes. GroupDRO and IRO receive a loss for every active environment.

This protocol supports a claim about transfer across missing constructed
corruption mechanisms. It does not establish robustness to geography,
acquisition devices, populations, or naturally occurring domains.

## Models and caveat

`finetune_last_stage` is the main setting: ResNet layer 4 and the conditional
classifier are trained while the remaining backbone and BatchNorm statistics
stay frozen. The backbone is ImageNet-1k pretrained, and ImageNet-1k contains
the ImageNet-100 classes and training images. This is transfer learning, not
representation learning from scratch.

The focused extension uses:

- ERM at lambda 0;
- GroupDRO at lambda 0;
- IRO with the existing adaptive-Beta update and
  `num_lambda_samples=4`.

The old frozen-feature pilot remains an internal smoke test only.

## Commands

Run unit checks first:

```bash
conda run -n domgen python -m unittest discover -s IMAGENET100C/tests -v
```

The required loader smoke test is:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n domgen python -m IMAGENET100C.train \
  --experiment E0 --algorithm erm --backbone_mode frozen_feature_pilot \
  --max_source_images 100 --steps 2 --eval_every 1 --batch_size 16 \
  --output_dir results/imagenet100c_e0_loader_smoke_linux
```

The historical full launcher is retained for reproducibility, but do not use it
for the paper extension:

```bash
nohup bash IMAGENET100C/run_seed_all.sh \
  --seed 1 --results-root /home/ra95tig/imagenet100c_results/seed1 \
  --python /home/ra95tig/anaconda3/envs/domgen/bin/python \
  --gpus 0,1 --offline \
  > logs/imagenet100c_seed1_legacy_full.log 2>&1 &
```

For each of the 27 selected checkpoints, use the focused evaluator command:

```bash
conda run -n domgen python -m IMAGENET100C.evaluate \
  /path/to/run/checkpoints/final.pt \
  --corruption_types gaussian_noise,defocus_blur,snow,contrast \
  --max_eval_images 1000 --batch_size 64 --workers 0 \
  --output_dir /path/to/run/evaluation_minimal_anchor1000
```

Use a new output directory. The evaluator writes `evaluation.jsonl`; it refuses
to overwrite an existing evaluation directory. The current
`run_evaluations.sh` is a full 15-type lambda-0 launcher and is intentionally
not the focused paper command.

## Current execution status

The historical local Windows RTX A1000 run completed the full 64-checkpoint
training matrix for seed 0. That matrix is retained as an internal diagnostic
bundle, not as the paper target.

On the remote Linux server, the `domgen` environment uses PyTorch 2.6 with
CUDA 12.4 on two RTX A6000 GPUs. Seed 1 and seed 2 each completed all 64
training checkpoints. No seed 3 or 4 training should be launched for this
submission plan.

The obsolete seed-1 full 15-type evaluation was stopped after 8 completed
checkpoints. Those JSONL files are preserved but are not part of the focused
27-checkpoint result set. Seed 2 has no evaluation outputs yet.

The existing `results_submit_img100/` figures are seed-0 pilot/training
artifacts. They are explicitly marked `pilot_anchor100` where appropriate and
must not be described as report-grade evidence. No focused multi-seed
visualizations have been generated yet.

## Remaining work and estimate

Training remaining for the focused plan: **none**, assuming the nine selected
seed-0 checkpoints are available from the completed local bundle. Seed 1 and
seed 2 already contain the nine selected training checkpoints each. If the
seed-0 checkpoints are not transferred to the remote server, transfer or rerun
only those nine runs; do not rerun the other 55 configurations.

Evaluation remaining:

- seed 0: 9 focused evaluations;
- seed 1: 9 focused evaluations, regardless of the 8 obsolete broad evaluations;
- seed 2: 9 focused evaluations;
- total: **27 focused evaluations**.

The completed broad evaluations show that a full 15-type, 5,000-image
checkpoint evaluation takes approximately four hours on this remote setup.
The focused evaluation reduces the work from 75 to 20 corrupted conditions and
from 5,000 to 1,000 validation images. A first focused checkpoint should be
benchmarked before finalizing the budget; a planning range is **10-30 minutes
per focused checkpoint**.

With two independent evaluator processes, the 27-checkpoint focused extension
is approximately **3-7 wall-clock hours**, plus startup and any queue imbalance.
This is a planning estimate, not a completed benchmark. It is substantially
smaller than the abandoned full-matrix plan and is the only ImageNet evaluation
that should be scheduled for this submission.

After focused JSONL files exist, aggregation can be run with:

```bash
conda run -n domgen python -m IMAGENET100C.analyze \
  '/home/ra95tig/imagenet100c_results/seed*/selected/*/evaluation.jsonl' \
  --output_dir /home/ra95tig/imagenet100c_results/analysis_minimal
```

The glob above is illustrative: pass the actual focused evaluation JSONL paths
or a shell-expanded list. `analyze.py` writes `summary.csv`, robust pairwise
rankings, and `analysis_manifest.json`. A separate multi-seed plotting layer is
still required for paper figures; the existing pilot plotting script is not a
substitute because it assumes the old seed-0, 100-image pilot.

## Artifacts and interpretation

A training directory contains:

```text
manifest.json
history.jsonl
checkpoints/final.pt
```

A focused evaluation directory contains `evaluation.jsonl` with clean metrics,
per-anchor severity metrics, severity-averaged domain losses, CVaR values,
identification intervals, counts, and the explicit four-anchor deployment law.

Do not combine the obsolete 15-type evaluations with focused four-anchor
results in one table. Do not call the old full matrix report-grade evidence.
The central AISTATS evidence remains the exact synthetic ranking reversals,
Colored-MNIST support removal, plug-in intervals, and failed robust-ranking
certificates.
