# ImageNet-100-C final-run training details

This is the self-contained configuration record for the ImageNet-100-C E3b
support comparison. Values below are taken from the canonical runner,
configuration, and model/data code. Where the executed environment did not
record an exact software version or class-name list, that limitation is stated
explicitly rather than inferred from package lower bounds.

## Data, splits, and evaluation

- **Dataset and class subset:** Hugging Face `clane9/imagenet-100`, revision
  `0519dc2f402a3a18c6e57f7913db059215eee25b`. The dataset feature must contain
  exactly 100 class names and labels in `[0,99]`; the manifest records the
  resulting integer-to-class-name mapping. The literal class-name list is not
  checked into this branch.
- **Splits:** source images come from the dataset `train` split. Final
  evaluation comes from the dataset `validation` split. Source assignments are
  class-stratified and non-overlapping; the E3b source budgets are
  `(10000,10000,10000,10000)` for `balanced`, `(25000,10000,4000,1000)` for
  `long_tail`, `(29000,9000,1800,200)` for `scarce_tail`, and
  `(30000,7500,2500,0)` for `missing`, in the anchor order
  `(gaussian_noise, defocus_blur, snow, contrast)`.
- **Source validation:** manifests reserve a class-stratified held-out subset
  of the source `train` split (`validation_per_domain=100` by default). It is
  recorded for diagnostics and descriptive identification calculations, not
  used for the canonical checkpoint choice: `checkpoint_selection=final` and
  `target_validation_used_for_selection=false`.
- **Input and preprocessing:** the pretrained ResNet-50 transform is
  `resize=232`, center crop `224x224`, bilinear interpolation, RGB, followed by
  ImageNet normalization with mean `(0.485,0.456,0.406)` and standard deviation
  `(0.229,0.224,0.225)`. No additional random geometric or photometric
  augmentation is configured.
- **Corruptions and severity:** source mechanism domains are the four anchor
  ImageNet-C corruption types above. For each source sample, severity is chosen
  deterministically and uniformly from `{1,2,3,4,5}` using the seed, split,
  image index, corruption type, and epoch. Final evaluation uses the same
  fixed class-stratified validation indices across all five severities and
  averages severity within corruption type; the canonical runner evaluates
  `max_eval_images=1000` per condition.
- **Sampling replacement:** each source `DataLoader` uses `shuffle=True` and
  therefore samples without replacement within a loader pass. When the loader
  is exhausted and restarted during the fixed 1,000 updates, samples can recur;
  across the full run the effective protocol is with replacement across passes.

## Model and optimization

- **Backbone and weights:** torchvision ResNet-50 initialized with
  `ResNet50_Weights.IMAGENET1K_V2`; the final fully connected layer is replaced
  by identity. The canonical runner uses `finetune_last_stage`: all backbone
  parameters are frozen except `layer4`, whose BatchNorm statistics remain in
  evaluation mode. A lambda-conditioned 100-class head maps 2048 features to a
  512-unit hidden layer and then to 100 logits, with FiLM gamma/beta conditioning
  on the scalar lambda.
- **Shared settings:** batch size `64`, `AdamW` before the transition, learning
  rate `3e-4`, weight decay `1e-4`, total `1000` updates, and `400` ERM warm-up
  updates for non-ERM methods. At the transition, non-ERM methods reset to Adam.
  The post-transition schedule is exactly
  `lr(t) = 3e-4 * 0.5 * (1 + cos(pi*t/600))`, for `t=1,...,600`, reaching zero
  at the final update. ERM has zero warm-up and no cosine schedule.
- **Method settings:** IRM and VREx use penalty weight `0.3` in the canonical
  runner; EQRM uses training quantile `alpha=0.9`; GroupDRO uses exponentiated
  gradient step size `eta=0.1`; INF-TASK samples 4 lambdas per update from
  Beta(1,1); IRO samples 4 lambdas per update, starts from Beta(1,1), and uses
  adaptive sampler learning rate `1e-6`. The four-sample budget is shared by
  IRO and INF-TASK for this ImageNet protocol.
- **Checkpoint policy:** final checkpoint after 1,000 updates; no target
  validation or target-data model selection.

## Software-version status

The dataset revision and model-weight enum are recorded exactly. The
version-controlled requirements specify only `torch>=2.0`, `torchvision>=0.15`,
`datasets>=2.18,<4`, and `imagecorruptions>=1.1.2`; the exact Torchvision,
PyTorch, Hugging Face Datasets, Pillow, NumPy, SciPy, and ImageNet-C
(`imagecorruptions`) versions used by the final runs were not captured. The
ImageNet-C backend includes a compatibility path documented for
`imagecorruptions 1.1.2`, but this is not evidence of the installed runtime
version. The protocol nevertheless supplies the backbone, preprocessing,
optimization, batch, schedule, corruption-severity, and method-specific
settings needed for the paper-wide **All training details** checklist, so its
answer is **Yes**. Exact runtime versions remain a reproducibility limitation
and should be supplied by a final environment lockfile or package manifest
when available.

The generated method table is
`results/reports/E3b/ImgNet/imagenet100c_hyperparameters.tex`; regenerate it
with `python IMAGENET100C/export_hyperparameters.py` from the repository root.
