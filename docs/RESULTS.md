# Results Guide

This document is the entry point for generated experiment artifacts. Raw
training outputs and derived tables/figures are separate: regenerate derived
outputs only from a known raw-result root and never overwrite an existing run.

## Completeness Snapshot

Status is a workspace snapshot, not a claim that every historical experiment
is publication-ready.

| Area | Current status |
| --- | --- |
| CMNIST E1 | Derived five-seed tables and figures exist. |
| CMNIST E3 | Derived five-seed tables and figures exist. |
| CMNIST E3b | Historical full artifacts exist; focused four-anchor artifacts include the shared-protocol candidate comparisons. |
| CMNIST E4 | Supplementary selected-checkpoint lambda artifacts exist; this is not a full method-by-seed study. |
| QRM control | Ten upstream-protocol runs exist: EQRM and VREx, seeds `0..4`. |
| Img100C seed 0 | Selected checkpoints are not present on the remote server; historical local training exists. |
| Img100C seed 1 | `21/21` selected candidate training checkpoints exist; `9/21` focused evaluations exist. |
| Img100C seed 2 | `19/21` selected candidate training checkpoints exist; `9/21` focused evaluations exist; two candidate runs are active. |
| Img100C seeds 3-4 | Not in the current focused scope. |

The current Img100C candidate matrix is `3 conditions x 7 methods x 3 seeds =
63 checkpoints`, but the remote workspace is not complete for that matrix.
Legacy full-matrix and 15-corruption outputs are retained for diagnostics and
must not be silently combined with focused four-anchor results.

The available Img100C focused report bundle is
`results_submit_img100/tables/focused_seed12/` and
`results_submit_img100/figures/focused_seed12/`. It contains 18 records for
seeds 1–2 and methods ERM, GroupDRO, and IRO only.

## Submission Priority

The deadline-critical evidence is intentionally narrower than the available
experiment code:

| Priority | Component | Decision |
| --- | --- | --- |
| Main text | Synthetic 240-configuration study | Keep; directly visualizes intervals, ambiguity, and ranking reversals. |
| Main text | CMNIST E3b support removal | Keep; direct empirical counterpart of the support/evidence result. |
| Main comparison | CMNIST ERM, IRM, VREx, EQRM, GroupDRO, INF-TASK, IRO | Keep when run under the shared protocol. |
| Appendix control | CMNIST E3 visible imbalance | Keep as one compact contrast showing nonzero support is different from missing support. |
| Appendix or drop | E1 domain count | Not deadline-critical; indirect coverage proxy. |
| Appendix or drop | E2 sample support | Finite-sample control, not the structural theorem. |
| Appendix or drop | E4 lambda sensitivity | Supplementary IRO behavior, not identifiability evidence. |
| Optional appendix | Img100C | External constructed-domain sanity check only; never block the main submission. |

After the current Img100C focused evaluation finishes, the planned CMNIST
follow-up for seeds 3 and 4 is limited to E3b support removal plus the compact
E3 visible-imbalance control, using the seven candidate methods. No E1, E2, or
E4 expansion is required for that follow-up.

## CMNIST

### Current Phase Names

The active report vocabulary is:

| Legacy label | Canonical phase | Meaning |
| --- | --- | --- |
| `E0` | `baseline` | Baseline/reproduction configuration |
| `E1` | `domain_count` | Number of observed training domains |
| `E2` | `sample_support` | Balanced samples per observed domain |
| `E3` | `visible_imbalance` | Unequal but nonzero source-domain support |
| `E3b` | `support_removal` | Balanced, long-tail, near-missing, and missing support |
| `E4` | `lambda_sensitivity` | Post-training lambda-grid evaluation |

These names are documentation/report labels. Existing command generators and
result directories may retain legacy E-tags for compatibility; no migration of
historical paths is required.

The principal CMNIST stress evidence is the support-removal analysis:

- E3b conditions: balanced-visible, long-tail-visible, near-missing, and
  missing-tail in the historical full study;
- the focused four-anchor artifacts are named with `_4anchor`;
- supported methods in the current report bundle include ERM, GroupDRO,
  INF-TASK, IRM, IRO, EQRM, and VREx where their protocols match;
- QRM upstream-protocol controls use two source environments `(0.1, 0.2)` and
  must remain labeled as reproduction/control results.

Other CMNIST analyses are secondary:

- E1: training-domain count;
- E3: visible imbalance;
- E4: lambda sensitivity, limited to selected checkpoints;
- P7 theory: identification width and ranking-reversal simulations.

Existing selected artifacts are under [results_submit](../results_submit/):

```text
results_submit/tables/
results_submit/figures/
results_submit/metadata/
```

The four-anchor E3b report files are the appropriate place for shared-protocol
candidate comparisons. Do not silently merge QRM's upstream two-source
protocol into those tables.

## ImageNet-100-C

ImageNet-100-C is an optional external constructed-domain replication. Its
active scope and status are maintained in
[docs/README_IMAGENET100C.md](README_IMAGENET100C.md) and
[IMAGENET100C/README.md](../IMAGENET100C/README.md).

The focused evaluation output is stored per checkpoint as:

```text
evaluation_minimal_anchor1000/evaluation.jsonl
```

Legacy full 15-type outputs and seed-0 pilot figures are diagnostics, not
focused report-grade results.

## Regeneration rules

- Keep raw runs immutable and write derived outputs into a new or explicitly
  versioned directory.
- Record the input root, seeds, algorithms, phase, and checkpoint policy in the
  output metadata.
- Do not combine final-checkpoint and validation-selected runs in one table.
- Do not combine four-anchor deployment laws with all-15-corruption laws.
- Label external QRM/SWAD controls separately from the main CMNIST protocol.
