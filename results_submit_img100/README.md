# ImageNet-100-C Submission Results

This bundle was generated from `C:/Users/320257223/PycharmProjects/domain_gen_IL/results/imagenet100c_seed0`.

## Scope

- Completed training runs: **64/64**
- Seeds: **0** (one seed only)
- Backbone: ImageNet-1k-pretrained ResNet-50 V2 with the last stage fine-tuned
- Checkpoint policy: fixed 1,000-step budget, final checkpoint
- Evaluation JSONL files discovered: **0**

## Critical interpretation

The current tables and figures are **training diagnostics**, derived from source-domain
losses logged during optimization. They are not held-out ImageNet-100-C deployment
results and must not be reported as clean accuracy, corruption accuracy, worst-domain
error, CVaR, or partial-identification evidence.

Final scientific tables require running `IMAGENET100C.evaluate` on the checkpoints.
That evaluation covers clean validation plus all 15 corruption types at five severity
levels. `IMAGENET100C.analyze` can then create deployment `summary.csv` and
`robust_ranking.csv` tables.

## Layout

- `tables/training_run_summary.csv`: one row per completed run.
- `tables/training_history.csv`: long-form 64,000-step diagnostic table.
- `tables/<experiment>/training_summary.csv`: experiment-specific tables.
- `figures/<experiment>/source_training_diagnostic.png`: source-loss diagnostics.
- `figures/E0_baseline/convergence.png`: smoothed E0 training objectives.
- `metadata/artifact_index.csv`: paths and sizes of all raw training artifacts.
- `metadata/completion_summary.json`: completeness and provenance summary.
- `metadata/evaluation_inventory.csv`: discovered deployment evaluation files, if any.

Raw checkpoints are referenced rather than copied, avoiding a duplicate of roughly
20 GiB of model files.
