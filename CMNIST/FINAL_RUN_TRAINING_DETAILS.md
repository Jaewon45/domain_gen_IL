# CMNIST final-run training details

This is the self-contained configuration record for the E3b CMNIST support
comparison on branch `reproducibility` (commit `ce0304e`).  It was extracted
from the 140 completed runtime manifests under `results/CMNIST/` and checked
against the frozen command generator and trainer.  The runtime manifests are
intentionally gitignored, so this file records their final-run values rather
than referring to repository defaults.

## Data, environments, and evaluation

- **Dataset:** the standard MNIST training and test splits loaded through
  `torchvision.datasets.MNIST(root, train=..., download=True)`.  The final-run
  manifests and logs identify the data root as
  `/home/ra95tig/qrm_cmnist_data`.  Its raw files were verified against the
  following MD5 fingerprints (the standard torchvision MNIST release):

  | File | MD5 |
  | --- | --- |
  | `train-images-idx3-ubyte.gz` | `f68b3c2dcbeaaa9fbdd348bbdeb94873` |
  | `train-labels-idx1-ubyte.gz` | `d53e105ee54ea40749a09fcbcd1e9432` |
  | `t10k-images-idx3-ubyte.gz` | `9fb629c4189551a2d022fa330f9573f3` |
  | `t10k-labels-idx1-ubyte.gz` | `ec29112dd5afa0611ce80d1b7f02629c` |

  These four checksums identify the dataset bytes used by the final runs.
  `CMNIST/requirements.txt` records lower bounds (`torch>=1.7.1`,
  `torchvision>=0.8.2`), rather than the exact runtime package versions.
- **CMNIST construction:** binarize the digit label as `digit < 5`, flip that
  label independently with probability 0.25, and flip its colour independently
  with probability `e` in environment `e`.  Duplicate the greyscale image to
  two channels, zero the channel not selected by the colour, divide by 255,
  and downsample 28x28 to 14x14 by stride-two spatial subsampling.  There is no
  other augmentation.
- **Split construction:** for training-time data, seed-specific `torch.randperm`
  shuffles the 60,000 MNIST training images; the first 50,000 are divided into
  four disjoint stride-four source pools (12,500 available images per source)
  and the remaining 10,000 form the training-time test pool.  Requested source
  counts are then selected by a second seeded random permutation.  Final
  reporting instead uses the shuffled 10,000-image MNIST test split, transformed
  separately for every evaluation environment.  The 10,000-image training-time
  pool is used only for periodic and final evaluation/logging.  In the final E3b
  runs it is not used for optimization, validation, hyperparameter tuning, early
  stopping, checkpoint selection, or any other model-selection decision:
  `checkpoint_selection=final` always reports the final-update checkpoint.
  The transformed evaluation datasets are constructed inside each method/seed
  job rather than loaded from a shared persisted cache, so they are regenerated
  separately for each method; matching seeds and configuration make those
  independently constructed datasets reproducibly equivalent.
- **Declared source anchors:** `(0.1, 0.2, 0.5, 0.9)`; final evaluation anchors:
  `(0.0, 0.1, ..., 1.0)` in increments of 0.1.  The source budget is always
  8,000 examples in total across the four declared anchors, not 8,000 per
  anchor.  Thus the balanced condition is `(2000, 2000, 2000, 2000)`:

  | Support condition | Counts at `(0.1, 0.2, 0.5, 0.9)` | Present source anchors |
  | --- | ---: | --- |
  | balanced | `(2000, 2000, 2000, 2000)` | all four |
  | long_tail | `(5000, 2000, 800, 200)` | all four |
  | scarce_tail | `(5800, 1800, 350, 50)` | all four |
  | missing_tail | `(6000, 1500, 500, 0)` | 0.1, 0.2, 0.5; 0.9 declared but withheld from training |

  In `missing_tail`, the zero-count 0.9 dataset is constructed but removed from
  the active training-loader list; it remains in the manifest and evaluation
  declaration.  The manifest therefore reports active counts `(6000,1500,500)`
  for that condition and the requested vector above.

- **Seeds and checkpoint:** seeds are `{0,1,2,3,4}` for Python `random`, NumPy,
  and PyTorch; cuDNN deterministic mode is false.  The reported model is the
  final update checkpoint, never a target-selected or validation-selected one.
  Checkpoints every 100 updates are recovery-only and are deleted after a
  successful final result and manifest write.
- **Fixed conditional evaluation input:** for IRO and INF-TASK, every final
  evaluation input is `lambda_eval = 0.9`.  This is neither EQRM's training
  quantile nor the post-training deployment-CVaR audit level.  The latter uses
  `alpha in {0.5, 0.75, 0.9}` in the interval table (and a wider displayed
  sensitivity grid from 0.0 through 0.95); it does not retrain a model.

## Shared model and optimization configuration

| Item | Final-run value |
| --- | --- |
| Model | `FiLMedMLP`, one 392-dimensional flattened 2x14x14 input, two 390-unit hidden layers, one-logit output, FiLM conditioning dimension 1 |
| Initialization | Xavier-uniform weights and zero biases for input, hidden, and output linear layers; FiLM gamma/beta generators retain PyTorch's constructor initialization |
| Dropout | 0.2 after input and after each hidden linear layer |
| Loss | binary cross-entropy with logits (`nll` option) |
| Loader batch size | 25,000 per active source loader; since each active source has <=6,000 examples, each update consumes its whole active source dataset (8,000 examples total) |
| Learning rate / weight decay | `1e-4` / `0` |
| Optimizer | AdamW before any transition; non-ERM methods reset to Adam at the ERM-to-method transition.  No non-default optimizer arguments are supplied. |
| Preprocessing / augmentation | stated in the preceding section; `full_resolution=false`, no geometric or photometric augmentation beyond CMNIST's stochastic label/colour generation |

The learning-rate factor is 1.  ERM has no warm-up and no cosine schedule: 600
AdamW updates.  All non-ERM methods do 400 ERM warm-up updates and then reset
the optimizer.  IRM, VREx, EQRM, IRO, and INF-TASK have 600 total updates (200
post-transition); GroupDRO has 1,000 total updates (600 post-transition).  For
each non-ERM run, cosine decay starts after the transition and is
`1e-4 * 0.5 * (1 + cos(pi*t/T))`, where `T` is the number of post-transition
updates (200 or 600).  It reaches zero at the final update.

## Method-specific final values

| Method | Final-run settings |
| --- | --- |
| ERM | 600 total updates; no warm-up, reset, or cosine decay. |
| IRM | 400 ERM warm-up + 200 IRM updates; optimizer reset; cosine decay; penalty weight 1,000 after the transition. |
| VREx | 400 ERM warm-up + 200 VREx updates; optimizer reset; cosine decay; variance-penalty weight 1,000 after the transition. |
| EQRM | 400 ERM warm-up + 200 EQRM updates; optimizer reset and cosine decay; training quantile `alpha=0.9` (a probability on the observed source-risk distribution). |
| GroupDRO | 400 ERM warm-up + 600 GroupDRO updates; optimizer reset and cosine decay; group-weight exponentiated-gradient step size `eta=0.1`. |
| INF-TASK | 400 ERM warm-up + 200 updates; optimizer reset and cosine decay; fixed preference `Beta(1,1)` and 5 sampled lambdas per method update; no adaptive sampler or sampler learning rate. |
| IRO | 400 ERM warm-up + 200 updates; optimizer reset and cosine decay; adaptive preference initially `Beta(1,1)`; 10 sampled lambdas for the model update.  Its distribution update uses 5 reparameterized uniform draws and a manual sampler step of `1e-6`, clamped to Beta parameters >= `1e-4`. |

`alpha=0.4`, `groupdro_eta=1.0`, and `penalty_weight=1000` may still appear
in irrelevant fields of some manifests because the trainer records all parsed
arguments.  They are not method settings unless stated in the table: EQRM's
executed `alpha` is 0.9, GroupDRO's executed `eta` is 0.1, and IRM/VREx's
executed penalty is 1,000.

## Selection provenance

The principal support vectors, five seeds, final-checkpoint rule, and shared
training values were fixed by the controlled E3b protocol before the
support-condition comparison.  IRM/VREx's 1,000 penalty and GroupDRO's 0.1
step size are inherited E3b/DGIL-lineage settings.  The IRO/INF-TASK
distributions, 10/5 Monte Carlo counts, and IRO's `1e-6` manual sampler step
are inherited from the later Singh CMNIST implementation; the sampler step is
an implementation setting, not an algorithm-theory constant.  EQRM's 0.9
training quantile was fixed for the missing-method replication to align with
the primary tail-risk level.  There was no target-data tuning and this record
does not claim a tuning protocol.

## Evidence and remaining checklist status

The concrete final-run artifacts are the 100 manifests in
`results/CMNIST/e3b_tail_support/` (ERM, IRM, GroupDRO, IRO, INF-TASK) and the
40 manifests in `results/CMNIST/e3b_tail_support_eqrm_vrex/` (VREx, EQRM).
They are runtime artifacts ignored by `.gitignore`; this self-contained record
is the version-controlled substitute.  The values are generated by
`CMNIST/job_scripts/gen_exps.py`, executed by `CMNIST/train_sandbox.py`, with
data construction in `CMNIST/datasets.py`, model definitions in
`CMNIST/networks.py`, and objectives in `CMNIST/algorithms.py` and
`CMNIST/lib/iro_utils.py`.

Accordingly, this closes the CMNIST training-details gap.  The checklist answer
for **All training details** is **Yes**: this file supplies a self-contained
final-run configuration record and identifies the exact MNIST source bytes.
Exact package versions were not logged by the final runs; they should be pinned
for bitwise reruns, but are not a reason to mark this training-details item
No.
