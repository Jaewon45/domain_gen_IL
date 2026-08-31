# CMNIST Submission Results

This directory contains selected CMNIST paper artifacts. Raw outputs remain under `results/`.

## Paper Roles

- E3b tail support: principal CMNIST evidence.
- E3 visible imbalance: supporting evidence for visible imbalance versus missing support.
- E1 domain count: secondary or appendix evidence; one source set per domain count.
- E4 lambda: **supplement only**. Still one IRO checkpoint and one INF-TASK checkpoint, both seed 0. Keep at most one main-text sentence (checkpoint-level lambda sensitivity and deployment-wide pseudo-regret, not true operator regret) and do not present it as algorithm-wide evidence.
- Priority 7: mechanism-level theory simulation.
- GroupDRO control: **supplement only**. Still three seeds (0-2). Fine as a control; keep at most one main-text sentence: source-objective robustness does not necessarily transfer to target-domain robustness.

## Layout

- `tables/`: E1, E3, E3b, E4, GroupDRO, and Priority 7 tables.
- `figures/`: E1, E3, E3b, E4, GroupDRO, and Priority 7 figures.
- `metadata/`: manifests, artifact index, run summaries, and table verification.

## Provenance

- E1: 100/100 successful runs, seeds 0-4, domain counts 2/4/6/8. Raw result directory contains 108 jsonl files (8 duplicate reruns from pre-fix launcher attempts); tables/figures are built from a deduplicated 100-record set keeping the most recently written record per (seed, algorithm, train_envs, train_env_sizes).
- E3: 125/125 successful runs, seeds 0-4, five fixed-budget schedules.
- E3b: 100 records, seeds 0-4, four support conditions (no duplicates). Includes a post-hoc 4-anchor CVaR variant (`*_4anchor.csv`/`*_4anchor.png`) restricted to the 4 source anchors `{0.1,0.2,0.5,0.9}`, giving the tightest link between the missing-support theorem and the experiment; the original 11-test-environment files remain the broader deployment-generalization measure.
- E4: two checkpoints (one IRO, one INF-TASK), both seed 0 only, 11 lambda values, five fixed test environments. Not yet extended to multiple seeds.
- GroupDRO control: matched-budget comparison still uses only seeds 0-2 (`CMNIST/job_scripts/groupdro_controlled_seed012_v2.txt`, 6 commands); not extended to seeds 3-4.
- Priority 7: 240 conditions and 1,000 repetitions per condition; synthetic, seed-independent.

All principal CMNIST stress experiments (E1, E3, E3b) are now five-seed (0-4). E4 lambda sensitivity and the GroupDRO matched-budget control remain limited to seed 0 and seeds 0-2 respectively — do not describe these as five-seed evidence.

Do not combine exploratory `domain_stress` results with corrected E1/E3 results. Alternative E1 subset variability and true operator regret are not included.