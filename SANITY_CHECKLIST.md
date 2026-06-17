# CMNIST Stress Test and Lambda Study Sanity Checklist

Use this checklist before scaling to large runs. Keep checked boxes in version control so progress is auditable.

## Logic Validation and Fix Status (2026-06-17)

- [x] Critical fix applied: IRO Pareto update now uses full parameter-gradient tuple and safe Beta-parameter update path.
- [x] High fix applied: `loss_fn` CLI choices are valid (`nll`, `cross_ent`) and no longer malformed.
- [~] Medium behavior check in progress: IRO domain-count robustness red flag is improved in tiny E1 smoke run, but full multi-seed E1 confirmation is still pending.
  - [x] Tiny smoke signal collected (seed 0, short run): IRO worst-domain accuracy no longer collapsed near zero.
  - [ ] Full-status closure pending: rerun E1 at longer schedule and across seeds before marking fully resolved.

## 0) Scope Lock

- [x] Phase labels are explicit and unique across runs: E0, E1, E2, E3, E4.
- [ ] Phase objectives are documented consistently in code comments and README:
  - [x] E0 reproduction
  - [x] E1 number of domains
  - [x] E2 samples/domain
  - [x] E3 imbalance
  - [x] E4 lambda-sensitivity
- [x] Dataset is CMNIST.
- [x] Initial train environments are fixed and documented.
- [x] Test environments are fixed and documented.
- [x] Initial algorithm set is fixed: ERM, GroupDRO, IRO.
- [x] Optional later algorithms are marked separately: INF-TASK, IRM.
- [x] Seed policy is fixed for this stage (1 seed first; 3 seeds if runtime allows).
- [x] Training steps are fixed and labeled as default or reduced reproduction.
- [x] Output root directory is fixed and writable.

## 1) Pre-Run Infrastructure Checks

- [x] `CMNIST/train_sandbox.py` accepts and logs:
  - [x] `--train_envs`
  - [x] `--train_env_sizes`
  - [x] `--train_env_size_mode`
  - [x] `--seed`
  - [x] `--deterministic` (if used)
- [x] `CMNIST/datasets.py` applies per-domain subsampling correctly.
- [x] `CMNIST/job_scripts/domain_stress_small.txt` exists and is readable.
- [x] `CMNIST/job_scripts/domain_stress.txt` exists and is readable.
- [x] `CMNIST/collect_results.py` runs without schema errors.
- [x] `CMNIST/export_results_csv.py` runs end-to-end.
- [x] `CMNIST/plot_domain_stress.py` runs end-to-end for E0-E3.
- [x] `CMNIST/evaluate_lambda_grid.py` is executable.

## 1.1) Dataset Construction Logic Checks

- [x] `train_envs` changes actual training-domain composition, not only printed labels.
- [x] `train_env_size_mode=random` and `train_env_size_mode=first` produce different sampled subsets.
- [x] With fixed seed and deterministic mode, repeated runs produce identical subsampling outcomes.
- [x] Imbalance naming matches semantics in code (which domain is heavy/light).
- [x] Test-domain set is held fixed whenever an experiment claims single-factor isolation.

## 1.2) Algorithm Coverage Checks

- [x] Small sweep uses intended subset for this stage (ERM, GroupDRO, IRO, plus INF-TASK where intended).
- [x] Full sweep uses intended five algorithms (ERM, IRM, GroupDRO, IRO, INF-TASK).
- [x] Any phase-level algorithm omission is intentional and documented.
- [x] Evaluation code keeps fixed baselines and lambda-conditioned methods in compatible but not conflated schemas.

## 2) Result Schema Sanity (Must Pass Before Any Claim)

Check at least one fresh JSONL output record manually and via script.

- [ ] Core identity fields present:
  - [x] `algorithm`
  - [x] `seed`
  - [ ] `phase`
  - [x] `train_envs`
  - [x] `train_env_sizes`
  - [x] `test_envs`
  - [x] `steps`
  - [x] output directory path
- [x] Stress metadata present or derivable:
  - [x] `phase`
  - [x] `n_train_domains`
  - [x] `sample_size_per_domain`
  - [x] `imbalance_type`
  - [x] `train_env_sizes`
- [x] Metrics present:
  - [x] per-test-env accuracy
  - [x] loss/risk where available
- [ ] Failed runs are explicitly distinguishable from valid low-performance runs (status/error marker).
- [x] Lambda-eval fields present for E4 outputs:
  - [x] `lambda_eval` flag or equivalent marker
  - [x] `lambda` value
  - [x] per-test-env metric(s)
  - [x] aggregated risk / CVaR summary
- [ ] No silent missing values in grouping columns used by analysis scripts.

## 3) E0 Reduced Reproduction Gate

Run E0 with one seed and ERM, GroupDRO, IRO first.

- [x] Command(s) complete without crash.
- [x] Result files are created in expected output dir.
- [x] One E0 accuracy-vs-test-environment plot is generated.
- [ ] One E0 summary table is generated:
  - [ ] average accuracy
  - [ ] worst-domain accuracy
  - [ ] regret (exact or clearly labeled approximate)
- [x] Short written interpretation is added and honest about reduced/default setup.
- [x] Decision gate passed: E0 artifacts are understandable and reproducible.

## 4) E1 Domain-Count Stress Gate

- [x] Domain-count conditions are explicitly listed for this run.
- [x] Exact train environment sets used are archived in notes/results.
- [x] `phase=domain_count` grouping works.
- [x] `n_train_domains` values are correct in outputs.
- [x] Plot generated: worst-domain accuracy (or regret) vs number of training domains.
- [ ] Optional second plot generated: test-env curves per domain-count condition.
- [x] Interpretation written (does IRO need more source domains?).

## 5) E2 Sample-Size Stress Gate

- [x] Sample-size grid for this run is documented.
- [x] `train_env_sizes` are saved exactly in each run record.
- [x] `phase=sample_size` grouping works.
- [x] `sample_size_per_domain` is correct in outputs.
- [x] Plot generated: worst-domain accuracy (or regret) vs samples/domain.
- [ ] Runtime/perf tradeoff notes captured if runtime differs strongly by algorithm.
- [x] Interpretation written (stability under low per-domain sample size).

## 6) E3 Imbalance Stress Gate

- [x] Imbalance schedules for this run are explicitly listed.
- [x] Environment-position semantics are documented (which env is overweighted).
- [x] `phase=imbalance` grouping works.
- [x] `imbalance_type` labels are populated and consistent.
- [x] `train_env_sizes` are present for all imbalance runs.
- [x] Plot generated: worst-domain accuracy (or regret) vs imbalance condition.
- [ ] Plot generated: balanced vs strong-imbalance test-env curves.
- [x] Interpretation written with correct causal claim strength.

## 7) E4 Lambda-Sensitivity Gate

- [x] Lambda grid is fixed and documented (for example 0.0 to 1.0).
- [x] `evaluate_lambda_grid.py` runs on saved checkpoints.
- [x] Lambda result files are written to expected directory.
- [ ] For IRO and INF-TASK (if available), lambda-conditioned metrics are recorded.
- [ ] For ERM/GroupDRO, lambda-dependent aggregation over fixed predictions is recorded.
- [x] Plot generated: aggregated risk vs lambda.
- [ ] Optional plot generated: worst-domain accuracy vs lambda.
- [ ] Optional heatmap generated: test env (y) vs lambda (x).
- [ ] Robustness summary computed:
  - [ ] best lambda
  - [ ] worst lambda
  - [ ] performance range across lambda
  - [ ] sensitivity score (max-min)

## 8) Aggregation and Export Gate

- [x] `collect_results.py` can group by:
  - [x] `phase`
  - [x] `algorithm`
  - [x] `seed`
  - [x] `n_train_domains`
  - [x] `sample_size_per_domain`
  - [x] `imbalance_type`
  - [x] lambda-eval indicator
- [x] `export_results_csv.py` outputs all expected tables:
  - [x] run-level
  - [x] env-metric long format
  - [x] aggregated summary
- [x] CSV row counts are plausible and match expected run count.
- [ ] No duplicate-run inflation in final tables.
- [x] Worst-domain accuracy is computed as min accuracy across test domains.
- [x] Average-domain accuracy uses the same test-domain set across compared methods.
- [ ] Grouped summaries do not mix phases, seeds, or condition definitions.

## 9) Reproducibility and Reporting Gate

- [ ] README section includes:
  - [x] environment setup
  - [ ] small subset run instructions
  - [ ] full sweep run instructions
  - [ ] result collection commands
  - [ ] plotting commands
- [x] Reduced vs full settings are clearly labeled in text and filenames.
- [x] Runtime assumptions (GPU count, expected days) are documented.
- [x] Known caveats are disclosed (train-env set choices, imbalance semantics, etc.).
- [x] All figures and tables have clear titles and axis labels.

## 9.1) Figure/Table Trust Checks

- [x] E0-E3 figures are generated from the same aggregated source used for tables.
- [x] Figure names map unambiguously to phases (`e0`, `e1`, `e2`, `e3`, `e4`).
- [ ] Reported table values match CSV exports exactly.
- [ ] Missing phase/algorithm rows are called out explicitly in captions/notes (not silently dropped).

## 9.2) Paper/Report Alignment

- [x] Manuscript/report text matches implemented train-env sets and imbalance schedules exactly.
- [x] Claims like majority-heavy/minority-heavy match the actual schedules in code.
- [x] Narrative clearly distinguishes reduced sweep from publication-grade sweep.
- [x] Stated limitations are traceable to observed repository behavior/results.

## 10) Stop/Go Rules

- [x] STOP if E0 outputs are unclear or schema is inconsistent.
- [x] STOP if grouping fields are missing for any stress phase.
- [x] GO to larger sweep only after E0-E3 reduced outputs are valid and interpretable.
- [x] GO to multi-seed only after one-seed runs are stable and complete.
- [x] GO to full 600-command sweep only after storage/runtime budget is confirmed.

## 11) Recommended Validation Order

- [x] Validate E0 end-to-end first: train -> log -> aggregate -> plot -> table.
- [x] Validate E1, E2, E3 sequentially using the same schema checks.
- [x] Run E4 only after checkpoints and E0-E3 schema consistency are confirmed.
- [ ] Final pass: verify README/methods text documents exact reduced vs full sweep differences.

## Minimum Deliverables (If Time-Constrained)

- [ ] E0 reduced reproduction plot + table.
- [x] E1 domain-count plot + short interpretation.
- [x] E3 imbalance plot + short interpretation.
- [x] E4 lambda-sensitivity plot for available checkpoints.
- [x] One paragraph stating whether reproduction is approximate and what matched/did not match expected behavior.
