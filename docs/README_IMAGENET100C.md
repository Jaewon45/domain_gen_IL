# ImageNet-100-C

This is the only active ImageNet extension in this repository. The older
non-100 ImageNet-C plan is archived at
[docs/legacy/README_IMAGENET_C_LEGACY.md](legacy/README_IMAGENET_C_LEGACY.md) and
should not receive new experiments.

## Purpose

ImageNet-100-C is an optional external constructed-domain replication of the
support-removal argument. It is not central evidence for the paper and must not
be described as a natural-domain benchmark.

The retained design uses four registered corruption anchors:

- `gaussian_noise`
- `defocus_blur`
- `snow`
- `contrast`

A source condition is `balanced`, `long_tail`, or `missing`; `near_missing` is
dropped. Severity `1..5` is averaged within each corruption type. Evaluation
uses 1,000 fixed class-stratified validation images per type/severity condition
and a uniform deployment law over the four anchors.

## Candidate methods

The current candidate set is:

```text
ERM, IRM, GroupDRO, IRO, INF-TASK, EQRM, VREx
```

The target matrix is `3 conditions x 7 methods x 3 seeds = 63 checkpoints`.
Seeds are `0,1,2`. Existing legacy ImageNet runs outside this matrix are
retained for diagnostics but are not part of the focused report.

## Status

- Seed 1 and seed 2 have the original ERM, GroupDRO, IRO, and INF-TASK training
  checkpoints for all three conditions.
- Shared-protocol EQRM and VREx runs have been added for the three conditions
  and seeds `0,1,2` in the current workspace.
- IRM/EQRM/VREx ImageNet candidates were added to the trainer; IRM, EQRM, and
  VREx seed-1/2 completion is tracked under the active result roots.
- Seed-0 selected checkpoints still need to be transferred from the historical
  local bundle before the complete 63-checkpoint evaluation can be claimed.

For exact commands, checkpoint layouts, environment assumptions, and current
runtime estimates, use [IMAGENET100C/README.md](../IMAGENET100C/README.md).

## Output policy

Use fresh roots under `results/imagenet100c_*` or
`/home/ra95tig/imagenet100c_results/`. Do not mix the old 15-type evaluation
outputs with four-anchor focused outputs. Focused evaluation artifacts use
`evaluation_minimal_anchor1000` and remain descriptive external replication
results.
