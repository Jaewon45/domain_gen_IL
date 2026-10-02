# Submission reproducibility and sanity checks

This is the release gate for the experiments in this repository. A result is **not submission-grade** unless every applicable blocking check below passes and the corresponding evidence is retained with the release artifacts. It is intentionally stricter than a smoke test: it checks that the reported number means what the paper says it means.

## 1. Freeze the protocol before running anything

Record one immutable JSON document for every `dataset × condition × method × seed` run. Store it beside the raw result and the checkpoint, and include at least:

```text
code_commit, repository_and_upstream_commit, experiment_id, dataset_manifest_hash, condition, seed, method, architecture_id, total_parameter_count, trainable_parameter_count, initialization_seed, data_shuffle_seed, source_domains, source_counts, deployment_domains, deployment_weights, evaluation_domains, loss, optimizer_and_state_reset_policy, learning_rate_schedule, batching_rule, updates_or_epochs, checkpoint_policy, lambda_training_distribution, lambda_evaluation_value, hardware, software_versions, checkpoint_path_and_hash, raw_result_path_and_hash
```

Treat a configuration change as a new experiment ID. Do not overwrite a failed run, replace it with a rerun under the same ID, or select a checkpoint using a source-absent test environment.

### Required common decisions

- [ ] Pin this repository commit and the upstream revisions used for baselines (DomainBed/QRM/DGIL as applicable).
- [ ] Lock Python, PyTorch, CUDA, cuDNN, GPU model, mixed-precision setting, and package versions (a lock file or `pip freeze` is acceptable).
- [ ] State the network precisely: layer widths, activation, dropout, FiLM placement, and whether a baseline receives no conditioning or a fixed conditioning value.
- [ ] Record Adam versus AdamW, learning rate, betas, epsilon, weight decay, cosine schedule, annealing/pretraining length, and every optimizer reset.
- [ ] State whether each update uses equal per-domain batches or pooled examples.  In the current CMNIST runner `--batch_size 25000` is passed to a loader for *each active domain*; this is not a pooled batch of 25,000.
- [ ] Seed Python, NumPy, PyTorch CPU and CUDA.  For deterministic CMNIST reruns pass `--deterministic`; record any remaining nondeterminism.
- [ ] Fix a final-checkpoint or source-only validation policy. The default `legacy_test_env_best` policy in `CMNIST/train_sandbox.py` is not acceptable for a deployment-absent test domain; use `--checkpoint_selection final` for submission runs unless a predeclared source-only alternative is implemented.
- [ ] Keep dataset-specific protocol decisions separate. CMNIST's 600-update budget is not automatically an ImageNet-100-C requirement.

## 2. CMNIST: data and run-level gates

The implementation in `CMNIST/datasets.py` currently constructs CMNIST as follows.  Cite this contract in the supplement and verify it with a small, deterministic run:

- MNIST train data are randomly permuted and split into 50,000 training and 10,000 test examples by default (`--use_test_set` is not exposed by `train_sandbox.py`).
- Digits `0--4` map to label 1 and `5--9` to label 0.  Labels are independently flipped with probability 0.25, then colours are independently flipped with the environment probability.
- Images are decimated by `[:, ::2, ::2]` from 28×28 to 14×14 unless `--full_resolution` is passed; this is slicing, not interpolated resizing.
- The two input channels begin as identical images and the non-colour channel is zeroed. Environment datasets share the requested test images but use independently sampled stochastic colour/label draws.
- A zero source count is removed from the training loaders, but its requested domain and zero count remain in tail-support metadata.

Before a full sweep:

- [ ] Run each intended method once with the final command template and inspect its JSONL record.  Confirm the algorithm, seed, source list, active source list, source-count map, test grid, steps, loss, and checkpoint policy.
- [ ] Re-run one deterministic configuration twice and require identical metadata, source counts, and final metrics (or record the justified nondeterminism).
- [ ] Confirm all four E3b count vectors and anchor order in the generated command manifest.  The declared anchors are `{0.1, 0.2, 0.5, 0.9}`.
- [ ] Confirm the paper’s mean/SD formula and use sample SD (`ddof=1`) for the five seeds; pandas `.std()` used by the current reporting scripts has this convention.
- [ ] Define worst-environment accuracy separately from the audit as `min_{e in {0,0.1,...,1}} accuracy_e`.  This predictive 11-point grid must not silently become the audit deployment law.
- [ ] Record the exact evaluation lambda for IRO and INF-TASK, and state the fixed lambda (if any) for every baseline.

Run the CMNIST tests from `domgen/CMNIST/` (the tests intentionally import the local modules directly):

```bash
cd CMNIST
python -m unittest discover -s tests -v
python -m py_compile train_sandbox.py datasets.py algorithms.py
```

Some legacy artifacts retain historical five-method wording. Regenerate and inspect the manifest rather than relying on those artifacts: submission E3b requires the declared seven methods and five seeds, or a clearly reduced and consistently reported scope.

## 3. CMNIST audit: release law and verification

**Release decision:** use the four-anchor uniform deployment law for the main CMNIST identification audit:

```text
A = {0.1, 0.2, 0.5, 0.9}
pi_star(a) = 1/4 for every a in A
M = 1, loss = 0--1, alpha = {0.50, 0.75, 0.90}
```

Under this law the missing mass must be:

| Condition | Observed anchors | epsilon |
| --- | --- | --- |
| Balanced | all four | 0 |
| Long-tail | all four | 0 |
| Near-missing | all four | 0 |
| Missing | `{0.1, 0.2, 0.5}` | 1/4 |

The 11-point grid `{0, 0.1, ..., 1}` remains valid for worst-environment accuracy only.  It is a different deployment universe.  In particular, the current `CMNIST/analyze_tail_support.py` creates deployment weights over its `--eval_envs` default (the 11-point grid), and the current `export_identification_audit.py` therefore reports `epsilon=7/11` for visible conditions and `8/11` for Missing.  Those outputs must not be used as the four-anchor submission audit.

Before release:

- [ ] Implement or configure a versioned audit input with **only the four anchors** and uniform weights; preserve 11-grid accuracy in a separately named artifact.
- [ ] Audit at run level, never after averaging losses across seeds.
- [ ] Verify that each condition has 7 methods × 5 seeds × 4 audit-domain rows, valid accuracies in `[0,1]`, and deployment weights summing to one.
- [ ] Compute discrete CVaR with fractional boundary mass.  For each method, seed, and alpha, use observed 0--1 losses with their deployment weights and complete missing mass once with loss 0 for the lower endpoint and once with loss 1 for the upper endpoint.
- [ ] Only compare intervals within the same seed.  Declare a ranking only when `upper_f + 1e-6 < lower_g` (or document another fixed tolerance).
- [ ] Save the audit source CSV, executable script/commit, run-level endpoints, summary, and a machine-readable check of the epsilon table above.

The present `export_identification_audit.py` is useful for checking the old 11-grid interpretation, but its assertions (`11` domains per run) make it unsuitable for this release audit without revision.

## 4. Table and artifact integrity

- [ ] Generate all tables/figures only from immutable raw JSONL/CSV artifacts; never manually copy a result into the paper.
- [ ] Hash every checkpoint and raw input.  Ensure every published summary row resolves to exactly the expected seed records; record missing and failed records explicitly.
- [ ] Use `CMNIST/verify_report_tables.py` for the E1/E3 summary families and require zero failures.  It recomputes means and sample SDs from JSONL.
- [ ] For E3b, verify the count, method, and seed cross-product before running `CMNIST/analyze_tail_support.py`; its output must carry the protocol label (`four_anchor_audit` versus `eleven_grid_predictive`) in its path and header.
- [ ] Remove or relabel all TODOs, historical five-method counts, pilot outputs, and legacy controls that conflict with the submitted tables.

Example E1/E3 table verification (adjust paths to the frozen bundle):

```bash
cd CMNIST
python verify_report_tables.py RESULTS_DIR SUMMARY.csv \
  --phase domain_count --output VERIFY_DIR/e1_table_check.csv
```

## 5. Method-specific checks

Publish a configuration table with one row per method.  Verify the code path, not just the command-line defaults, for each field.

| Method | Sanity requirement |
| --- | --- |
| ERM | State pooled-example versus equal-domain risk, optimizer and budget. |
| GroupDRO | Record initial group weights, multiplicative update, eta, normalization/clipping, pretraining and reset. |
| IRM | Record scale-variable penalty, coefficient, annealing point, and reset. |
| VREx | Record domain-risk variance, coefficient, aggregation, annealing point, and reset. |
| EQRM | Record empirical/KDE quantile estimator, alpha, bandwidth/smoothing, risk samples, objective direction, and annealing. |
| INF-TASK | Record conditioning input, lambda distribution, preferences per update, objective, and evaluation lambda. |
| IRO | Record conditioning architecture, tail/quantile estimator, adaptive-Beta update, outer/internal preference samples, step size, clamp, and evaluation lambda. |

Use **EQRM** for the implemented empirical method and reserve **QRM** for the population framework.  In `CMNIST/algorithms.py`, IRM, VREx, GroupDRO, and other robust methods have explicit ERM pretraining and optimizer-reset paths; check these against the pinned upstream implementation and make their budgets consistent with the reported protocol.

### Dataset-specific optimizer policies

The CMNIST 400/600 schedule is supported by the CMNIST protocol and is not
evidence that ImageNet-100-C must use 600 total updates. For ImageNet-100-C,
the current frozen comparison policy is:

```text
total_updates = 1000 for every retained method
erm_warmup_updates = 400 where the method uses the adopted warm-up policy
EQRM alpha = 0.9, frozen across seeds and support conditions
lambda_eval = 0.9 for IRO and INF-TASK
```

The ImageNet warm-up, optimizer reset, cosine schedule, and penalty rescaling
are adopted implementation choices that must be recorded and validated; they
are not claims that the seminar paper established these choices for ImageNet.
IRM and VREx do not intrinsically use EQRM's quantile alpha. Their manifests
must not treat a shared `alpha` field as evidence that they trained at an
EQRM quantile. EQRM's alpha must be recorded because it changes its objective.

### ImageNet IRM/VREx optimization alarm

The observed ImageNet IRM/VREx failure signature is a real optimization alarm:
source losses improve through the 400-update ERM warm-up, then deteriorate when
the invariant penalty activates and approach the uniform 100-class baseline,
`log(100)`. For VREx, equalizing all domain risks at a bad common value is a
natural variance-penalty degeneracy.

Do **not** describe removing a division by `1000` as a rebalancing fix. The
objectives `(R + 1000 P) / 1000` and `R + 1000 P` differ only by a global
factor, so their relative risk/penalty weighting is identical. With Adam/AdamW
this may change little; with SGD it mostly resembles a learning-rate change.

Before a full three-seed ImageNet matrix, use the same step-400 checkpoint and
minibatch to measure separately for IRM and VREx:

```text
g_R = grad(mean_risk)
g_P = grad(raw_penalty)
||g_R||
||g_P||
lambda_penalty * ||g_P|| / ||g_R||
cosine(g_R, g_P)
```

Run short 50--100-update source-only probes with method-specific coefficients
such as `lambda_penalty = 1, 10, 30, 100, 300, 1000`. IRM and VREx must not
automatically share a coefficient: their penalties have different scales and
gradients. Record per-domain risks, source/clean accuracy, penalty, gradient
norms/cosine, prediction entropy, and largest-class frequency.

Current chance-level IRM/VREx ImageNet checkpoints are failed optimization
runs, not meaningful baselines, until this diagnostic and a corrected probe
pass. The step-400 discontinuity is the primary signal; legacy alpha metadata,
lambda evaluation, and checkpoint selection are not its primary explanation.

### Frozen settings summary

The following settings are the intended final comparison protocols. `total`
means total optimizer updates, including warm-up; it is not added to the
warm-up count.

| Dataset / method | Total updates | ERM warm-up | Main updates | Optimizer policy | Method parameters |
| --- | ---: | ---: | ---: | --- | --- |
| CMNIST ERM | 600 | 0 | 600 | Adam, lr `1e-4`, weight decay `0`, no schedule | pooled ERM objective |
| CMNIST GroupDRO | 600 | 400 | 200 | Adam warm-up, reset at transition, cosine after transition | eta `0.1` |
| CMNIST IRM | 600 | 400 | 200 | Adam warm-up, reset at transition, cosine after transition | penalty `1000` |
| CMNIST VREx | 600 | 400 | 200 | Adam warm-up, reset at transition, cosine after transition | penalty `1000` |
| CMNIST EQRM | 600 | 400 | 200 | Adam warm-up, reset at transition, cosine after transition | `eqrm_alpha=0.9` |
| CMNIST IRO | 600 | 400 | 200 | Adam warm-up, reset at transition, cosine after transition | adaptive preference sampler; `lambda_eval=0.9` |
| CMNIST INF-TASK | 600 | 400 | 200 | Adam warm-up, reset at transition, cosine after transition | preference sampler; `lambda_eval=0.9` |
| ImageNet100C ERM | 1000 | 0 | 1000 | AdamW, lr `3e-4`, weight decay `1e-4` | lambda fixed at zero for ERM |
| ImageNet100C GroupDRO | 1000 | 400 | 600 | AdamW warm-up, reset at transition, cosine after transition | eta `0.1` |
| ImageNet100C IRM | 1000 | 400 | 600 | AdamW warm-up, reset at transition, cosine after transition | penalty `1000` |
| ImageNet100C VREx | 1000 | 400 | 600 | AdamW warm-up, reset at transition, cosine after transition | penalty `1000` |
| ImageNet100C EQRM | 1000 | 400 | 600 | AdamW warm-up, reset at transition, cosine after transition | `eqrm_alpha=0.9` |
| ImageNet100C IRO | 1000 | 400 | 600 | AdamW warm-up, reset at transition, cosine after transition | adaptive Beta preference sampler; `lambda_eval=0.9` |
| ImageNet100C INF-TASK | 1000 | 400 | 600 | AdamW warm-up, reset at transition, cosine after transition | preference sampler; `lambda_eval=0.9` |

Shared settings:

```text
CMNIST architecture       = FiLMedMLP, hidden width 390, dropout 0.2
CMNIST batch size         = 25000 per active source domain
ImageNet architecture     = ResNet-50 IMAGENET1K_V2, fine-tune layer4 + head
ImageNet batch size       = 64 per training update
ImageNet lambda samples   = 4 for conditional methods
checkpoint selection      = final
checkpoint interval       = every 100 updates for new runs
lambda_eval               = 0.9 for IRO and INF-TASK
audit_cvar_alpha          = 0.50, 0.75, 0.90, post-training only
```

The ImageNet 1000-update budget is an independent secondary-replication
choice. It is not inherited from the CMNIST 600-update budget. The ImageNet
400-update warm-up and optimizer transition are also adopted frozen settings;
they must be reported as implementation choices unless a specific ImageNet
reference implementation is cited.

## 6. ImageNet-100-C: separate release gate

ImageNet-100-C is an external constructed-domain supplement, not evidence to mix with CMNIST. Its active protocol declares four anchors `gaussian_noise`, `defocus_blur`, `snow`, and `contrast`; uniform four-anchor deployment; four conditions (`balanced`, `long_tail`, `scarce_tail`, `missing`); seven methods; and seeds `0,1,2`. Do not label pilot files under `results_submit_img100/` as report-grade evidence.

Choose exactly one outcome before submission:

- [ ] Complete the declared 4 × 7 × 3 checkpoint/evaluation matrix, including all focused four-anchor evaluations and a four-anchor audit (`epsilon=1/4` only for Missing); or
- [ ] Remove ImageNet-100-C from the main result claims and label it incomplete future/supplementary work.

For a completed protocol, retain hashes/manifests for the ImageNet-100 revision and class list, source assignments/counts, fixed class-stratified evaluation set, corruptions and severity aggregation, transforms, backbone checkpoint, frozen modules, head architecture, cross-entropy reduction, AMP/GPU settings, and all seed IDs.  Run its unit tests before a pilot:

```bash
python -m unittest discover -s IMAGENET100C/tests -v
```

IRM, VREx, or EQRM near chance in the Balanced condition is an optimization or protocol alarm, not missing-support evidence. Confirm the adopted 400-step ERM warm-up, optimizer reset, post-warm-up schedule, penalty scaling, EQRM alpha, and method-specific objective diagnostics in `history.jsonl` before interpreting a row. Do not describe these ImageNet choices as paper-mandated corrections without a source/reference-code citation.

## 7. Final sign-off

- [ ] Every reported result has a provenance record and hash chain from raw result to table cell.
- [ ] CMNIST deployment law, support mask, audit loss, alpha grid, CVaR boundary rule, and ranking tolerance are written in the paper/supplement.
- [ ] The audit epsilon checks pass exactly: `0, 0, 0, 1/4` for the four E3b conditions under the selected four-anchor law.
- [ ] All method configurations, lambda values, checkpoint rules, seed values, hardware, and software versions are published.
- [ ] ImageNet uses one common total-update budget across retained methods; no CMNIST 600-update assumption is silently transplanted into ImageNet.
- [ ] No result combines CMNIST’s 11-grid predictive metric with its four-anchor identification audit, or combines ImageNet pilot artifacts with focused results.
- [ ] A clean environment can run the unit tests, a deterministic CMNIST smoke run, analysis, table verification, and the audit from the frozen artifacts.

If any unchecked item changes the reported protocol or result interpretation, the correct action is to regenerate the affected artifacts and update the paper—not to mark the check as waived.
