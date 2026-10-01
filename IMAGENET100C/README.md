# ImageNet-100-C support-removal replication

This is an external constructed-domain replication. It does not claim that ImageNet-C corruptions are natural populations, and it is supplementary to the CMNIST evidence.

## Fixed protocol

- Anchors: `gaussian_noise`, `defocus_blur`, `snow`, and `contrast`.
- Deployment law: uniform over the four anchors.
- Conditions: `balanced`, `long_tail`, `near_missing`, and `missing`.
- Methods: ERM, IRM, VREx, EQRM, GroupDRO, INF-TASK, and IRO.
- Seeds: `0,1,2`.
- Matrix: `4 conditions × 7 methods × 3 seeds = 84` final checkpoints.
- Evaluation: 1,000 fixed class-stratified validation images per anchor/severity condition, with severity averaged within anchor.
- Checkpoint policy: final checkpoint after 1,000 updates; no target validation is used for selection.

`near_missing` has positive source support at every anchor. `missing` has a zero count at the final anchor and missing mass `epsilon = 1/4` under the stated deployment law.

## Shared transition policy

IRM, VREx, and EQRM use the CMNIST-style transition policy in the shared trainer:

- 400 plain-ERM warm-up updates;
- an AdamW-to-Adam optimizer reset at the objective transition;
- cosine decay over the post-warm-up updates;
- `penalty_weight = 1000` and `(risk + penalty_weight × penalty) / penalty_weight` for IRM and VREx;
- EQRM at quantile `alpha = 0.9`, with its CMNIST-style gradient-ratio scaling after warm-up.

These settings are fixed across conditions and seeds. Check `history.jsonl` for `training_phase`, `optimizer_lr`, IRM/VREx penalties, and the EQRM gradient ratio before interpreting a run. Any checkpoint produced before this transition policy is legacy diagnostic output and must not be combined with the canonical matrix.

## Setup and canonical runner

From the repository root:

```bash
python -m pip install -r IMAGENET100C/requirements.txt
python -m unittest discover -s IMAGENET100C/tests -v
bash IMAGENET100C/run_e3b.sh --seed 0 --results-root results/imagenet100c --gpus 0,1
bash IMAGENET100C/run_evaluate_e3b.sh --seed 0 --results-root results/imagenet100c --gpus 0,1
```

Repeat once per seed. The only canonical layout is:

```text
results/imagenet100c/
  seed<seed>/
    E3b_<condition>_<algorithm>/
      manifest.json
      history.jsonl
      checkpoints/final.pt
    logs/
```

The runner refuses to reuse incomplete run directories and safely skips a directory that already contains `checkpoints/final.pt`.

The old non-100 ImageNet-C implementation and its plans are archival material under `docs/legacy/`. The GPU-only benchmark is performance instrumentation, not a source of paper checkpoints.
