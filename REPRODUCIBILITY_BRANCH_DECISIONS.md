# Reproducibility Branch Decisions

This document records the controlled extension of branch `reproducibility` from the Singh-era reference point `fbebfac` (`Update README.md`). It separates algorithm implementation behavior from dataset-specific experiment protocol settings.

## Branch Scope

The working branch is `reproducibility`, checked out at:

```text
/home/ra95tig/lrz_mount/domgen_ai
```

The branch is built from `fbebfac`, not by blindly merging the newest `aistat` tree. Changes are selected by effective behavioral scope and committed in small, reviewable units.

The accepted CMNIST commits are:

```text
2b788c0  Implement paper-faithful IRO full-gradient sampler
108607f  Fix VREx CUDA losses and log GroupDRO diagnostics
54c3ef0  Add controlled CMNIST environment subsampling
9a884e2  Implement controlled CMNIST principal-study trainer
92f308e  Generate only declared CMNIST source-budget vectors
9294b57  Summarize CMNIST results under fixed final policy
ba391b9  Record complete CMNIST protocol arguments in manifest
```

ImageNet100C source, protocol configuration, and tests were added separately in commit `45e1c2c`. GPU-only pipelines, launcher families, checkpoints, logs, generated evaluations, and large artifacts are not part of the reproducibility source branch.

## Singh Reference Lineage

### Earlier Singh implementation

The earlier implementation at `44acefd` used:

```text
IRO:       Beta(0.5, 0.5), sampler Adam learning rate 1e-5, 5 samples/update
INF-TASK:  Beta(1, 1), 5 samples/update
```

This is historical provenance, not the current reference baseline.

### Later Singh implementation

The later optimization commit `13624f4` (`fixed optimization`) changed the lineage to:

```text
IRO:       Beta(1, 1), manual sampler step 1e-6, 10 samples/update
INF-TASK:  Beta(1, 1), 5 samples/update
```

The paper-level distinction is:

```text
INF-TASK: fixed uniform preference distribution, Beta(1,1)
IRO:      adaptive preference distribution Q_t, initialized at Beta(1,1)
```

The later code's `1e-6` sampler step is retained as inherited implementation behavior. It is not treated as a theoretically prescribed IRO constant.

The full-gradient correction is accepted because the later Singh code accidentally kept only the first result of `autograd.grad(...)`. The corrected implementation uses the full parameter-gradient tuple, handles unused gradients, and keeps Beta parameters positive and differentiable across updates.

## CMNIST Decisions

### Accepted implementation changes

- VREx losses are accumulated as model-device tensors and stacked, avoiding the old CPU tensor/device mismatch.
- GroupDRO records per-domain losses and group weights without changing its objective.
- IRO computes the full parameter-gradient tuple for its adaptive sampler update.
- IRO handles missing gradients and keeps Beta parameters valid.
- `train_env_sizes` provides a generic per-domain sample-support mechanism.
- The declared source-domain order is preserved.
- A zero-count source domain is excluded from training but retained in protocol metadata and evaluation declarations.
- The trainer parses actual command-line arguments rather than forcing an empty argument list.
- All parsed protocol arguments are written into the run manifest.
- Final-checkpoint evaluation is separated from model and hyperparameter selection.
- The evaluation lambda is fixed by protocol and is not selected using target data.
- A final checkpoint is always written.
- Periodic checkpoints are recovery checkpoints, not candidate checkpoints. They are retained if a run is interrupted and removed only after final evaluation, result writing, and manifest writing succeed.
- The result collector defaults to final-checkpoint metrics.

### Principal CMNIST source budgets

The manuscript commits the principal study to a total source budget of 8,000 and exactly these vectors:

```text
balanced:    (2000, 2000, 2000, 2000)
long_tail:   (5000, 2000, 800, 200)
scarce_tail: (5800, 1800, 350, 50)
missing:     (6000, 1500, 500, 0)
```

The generator was restricted to these four vectors. The following exploratory vectors are not valid principal-study commands and are excluded from the controlled principal generator:

```text
(2000, 2000, 2000, 4000)
(8000, 8000, 8000, 8000)
(2000, 2000, 2000, 10000)
```

## ImageNet100C Provenance

`IMAGENET100C/` was added directly to this repository in commit `7af2157` (`Add imagenet 100 C experiments`). It was not cloned from an external code repository.

External data and library dependencies are separate from code provenance:

```text
Dataset:   clane9/imagenet-100
Revision:  0519dc2f...
Weights:   torchvision IMAGENET1K_V2
Corruption implementation: imagecorruptions package
```

The ImageNet implementation is an external constructed-domain replication authored in this repository. Its deployment laws are locally defined over corruption types and registered E3b anchors.

## ImageNet Algorithm and Protocol Separation

The ImageNet algorithm implementation contains ERM, IRM, VREx, EQRM, GroupDRO, INF-TASK, and IRO. The implementation should expose protocol settings rather than silently defining them as algorithm theory.

### `num_lambda_samples`

The later CMNIST reference uses different counts:

```text
IRO:       10
INF-TASK:  5
```

ImageNet instead exposes `num_lambda_samples` in `IMAGENET100C/train.py`, and its launch scripts explicitly pass:

```text
--num_lambda_samples 4
```

The code and call-site audit found:

- the setting is passed separately from evaluation lambda;
- both IRO and INF-TASK receive the same value, 4;
- no deployment-law or evaluation setting uses the number 4;
- deployment laws contain corruption-type weights, not Monte Carlo sample counts;
- ImageNet evaluation uses a separate lambda grid, whose manuscript value is 0.9.

Therefore, 4 is specific to the **ImageNet experiment invocation**. It is not a property of IRO, INF-TASK, or ImageNet-C theoretically, and it was not inferred from the deployment law.

Keeping 4 is reasonable if the protocol:

- fixes the count within each dataset across IRO and INF-TASK;
- states CMNIST's and ImageNet-100-C's values explicitly;
- does not compare raw training cost or claim identical optimization fidelity across datasets;
- keeps the count fixed across ImageNet support conditions and seeds.

The more notable inconsistency is within the later CMNIST lineage itself: IRO uses 10 while INF-TASK uses 5. CMNIST's 5 versus ImageNet-100-C's 4 is a minor, defensible port-level compute choice.

The ImageNet protocol now records:

```text
num_lambda_samples = 4
rationale = fixed equal Monte Carlo budget for IRO and INF-TASK under ImageNet compute constraints
```

This belongs in the ImageNet manifest/config commit, not as a theoretical algorithm constant.

### IRO sampler learning rate

The later CMNIST ancestor uses the manual sampler step `1e-6`, and the ImageNet port copies that value. Preserve it for implementation comparability, but record it as a protocol setting:

```text
iro_sampler_learning_rate = 1e-6
```

The paper specifies adaptive `Q_t` initialized at Beta(1,1), not this particular learning rate.

### Cosine schedule

The cosine schedule is a training protocol setting. It is enabled only when the ImageNet manifest/launcher requests it. It is not a property of IRO, INF-TASK, or the deployment law.

The ERM schedule must be declared explicitly because inherited CMNIST behavior excluded ERM from the robust-method warm-up/cosine path. An ERM shared-schedule control was added separately for ImageNet comparison.

## IRO and INF-TASK Interpretation

Do not describe IRO and INF-TASK as theoretically equivalent.

```text
INF-TASK:
  intentionally fixed Beta(1,1)

IRO:
  adaptive Beta distribution Q_t
  initialized at Beta(1,1)
  inherited sampler step 1e-6
  may remain near-uniform in the observed runs
```

The correct empirical statement is:

> Under the inherited ImageNet implementation and sampler learning rate of 1e-6, IRO's observed sampling distribution remained near Beta(1,1), making it operationally similar to INF-TASK in these runs. This is an implementation/optimization outcome, not a theoretical equivalence.

The implementation should log for IRO and INF-TASK:

- sampler type;
- sampled-lambda quantiles;
- IRO alpha and beta values;
- sampler update count;
- IRO sampler learning rate.

These logging changes do not alter the training objective.

## Completed ImageNet Tests

The ImageNet test extension proves:

1. IRO's sampler update uses the full parameter-gradient tuple, not only the first parameter gradient.
2. A toy IRO case can depart from Beta(1,1) when given an intentionally larger test learning rate.
3. The production value `1e-6` is supplied by protocol configuration rather than silently hardcoded as algorithm theory.
4. `num_lambda_samples=4` is passed identically to IRO and INF-TASK in the ImageNet protocol.

## Excluded Artifacts

Do not add these to the reproducibility source branch:

- `.pt` checkpoints;
- `.pyc` bytecode;
- launch logs;
- generated evaluation outputs;
- failed-startup artifacts;
- old collapsed ImageNet runs;
- generated figures and report tables.

Those artifacts remain useful locally for diagnosis, but the controlled branch should contain source, tests, manifests/configuration, and concise protocol documentation.
