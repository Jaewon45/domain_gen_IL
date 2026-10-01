# AISTATS 2027 Tail-Risk Paper: Revision and Reproduction Protocol

This README is the implementation checklist for revising the current main paper and supplement and for regenerating only the experiments that materially support the identification claim.

The source documents that this protocol targets are:

- `aistats2027 (4).tex`
- `supplement.tex`

The exact LaTeX replacement / deletion / insertion blocks are in `LATEX_REVISION_BLOCKS.md`.

---

## 1. Target paper after revision

The final empirical story should contain only:

1. a deterministic synthetic identification / ranking-reversal construction;
2. the five-seed Colored-MNIST support-removal experiment;
3. the visible-imbalance control;
4. an unrestricted post-training identification sensitivity map over `(alpha, epsilon)`;
5. a structured-completion sensitivity analysis showing how additional verified risk restrictions tighten the identified set.

Remove from the submitted paper:

- incomplete ImageNet-100-C results;
- the domain-count control;
- the single-seed lambda-sensitivity diagnostic;
- the separate matched-budget GroupDRO diagnostic after the principal GroupDRO runs are rerun at the common budget;
- the 240-configuration ranking-reversal frequency study;
- the old emphasis on `0/150` or `0/105` certification failures as if they were independent empirical discoveries.

The paper should present the unrestricted `[0,1]` completion as an assumption-free baseline and the structured analysis as a sensitivity exercise, not as a claim that a particular smoothness condition is true.

---

## 2. Non-negotiable conceptual definitions

### 2.1 Source anchors and support conditions

Use exactly:

```text
anchors = [0.1, 0.2, 0.5, 0.9]

balanced    = [2000, 2000, 2000, 2000]
long_tail   = [5000, 2000,  800,  200]
scarce_tail = [5800, 1800,  350,   50]
missing_tail= [6000, 1500,  500,    0]
```

Do not use the term `near-missing` in the final paper. Rename it `scarce-tail` everywhere.

Every condition has total source count `8000`.

A zero-count environment must not construct a training loader.

### 2.2 OOD performance grid versus identification deployment population

These are different objects and must never be conflated.

For realised worst-environment accuracy:

```text
eval_grid = [0.0, 0.1, 0.2, ..., 1.0]   # 11 environments
```

For the identification analysis:

```text
A_ID = [0.1, 0.2, 0.5, 0.9]
pi_star = [0.25, 0.25, 0.25, 0.25]
```

Therefore:

```text
balanced:     epsilon = 0
long_tail:    epsilon = 0
scarce_tail:  epsilon = 0
missing_tail: epsilon = 0.25
```

Do **not** use `epsilon = 7/11` or `8/11`. The 11-point grid is an OOD evaluation grid, not the deployment law for the identification theorem.

### 2.3 Fixed predictor for IRO and INF-TASK

For all principal performance and identification outputs, fix:

```text
lambda_eval = 0.9
```

for IRO and INF-TASK.

Rules:

- use the same `lambda_eval` for every seed;
- use the same `lambda_eval` for every support condition;
- use the same `lambda_eval` for every evaluation environment;
- never choose lambda from held-out target performance;
- do not identify `lambda_eval` with the post-training CVaR level `alpha`.

The post-training identification sweep varies `alpha` independently while the predictor remains fixed at `lambda_eval = 0.9`.

If the repository already has a fully documented, pre-registered fixed operator input used for every principal run, it may be retained instead of `0.9`, but then replace `0.9` consistently in **all** code, generated artefacts, main text, and supplement. Do not retain a value that was chosen from target performance.

---

## 3. Audit existing CMNIST runs before rerunning anything

The user already has full five-seed CMNIST runs. Reuse them wherever the following audit passes.

For every principal checkpoint, verify and export:

```text
seed
method
support_condition
source_counts
source_anchor_values
code_commit_or_archive_hash
architecture
initialization
optimizer_before_pretraining_boundary
optimizer_after_pretraining_boundary
learning_rate
weight_decay
pretraining_steps
annealing_steps_if_any
total_update_calls
method_specific_coefficients
preference_sampling_distribution
number_of_preference_samples
beta_update_parameters_if_IRO
checkpoint_path
```

Also verify whether the same dataset realisation / split was reused for all methods at the same `(seed, support_condition)`.

### Audit decision tree

**Case A: all metadata is recoverable, and data are paired across methods.**

Reuse all completed checkpoints except GroupDRO, which must be rerun at the common principal budget described below.

**Case B: VREx or EQRM hyperparameters cannot be recovered exactly.**

Do not reconstruct them from memory. Either:

1. rerun that method under a frozen, documented configuration for all `5 seeds x 4 conditions = 20` runs; or
2. remove that method from every main/supplement table and figure.

The theory does not require seven methods. A smaller fully reproducible comparison is preferable to an undocumented row.

**Case C: the same seed produced different data realisations across methods.**

The paper cannot claim paired method comparisons from the existing runs. Either omit paired language and recompute all post-processing without paired inference, or preferably rerun the principal method set with a shared dataset realisation per `(seed, condition)`. If fully rerunning, the total is:

```text
K methods x 5 seeds x 4 support conditions
```

where `K` is the final verified method set.

---

## 4. Mandatory training rerun

### 4.1 GroupDRO: common principal budget

The current paper uses `600` update calls for most principal methods and `1000` for GroupDRO. Remove this budget mismatch from the final comparison.

Use:

```text
total_update_calls = 600
pretraining_updates = 400   # if this is the existing verified GroupDRO protocol
algorithm_specific_updates = 200
```

Rerun GroupDRO for:

```text
5 seeds x 4 support conditions = 20 runs
```

Do not tune GroupDRO separately by support condition.

All GroupDRO hyperparameters other than the total budget must remain frozen to the verified principal configuration unless a bug is found. If a bug is found, document it and rerun all affected GroupDRO conditions/seeds with one corrected configuration.

After this rerun, delete the separate matched-budget GroupDRO diagnostic from the supplement.

### 4.2 Conditional reruns for undocumented methods

If exact VREx or EQRM settings cannot be recovered, rerun or drop as described in Section 3. Do not leave `[TODO]` hyperparameters in the supplement.

---

## 5. Mandatory evaluation refresh; no neural retraining required

Even when checkpoints are reused, recompute evaluation outputs from the saved final checkpoint.

For every principal run:

1. load the final checkpoint;
2. for IRO / INF-TASK, set `lambda_eval = 0.9`;
3. evaluate accuracy on every environment in `eval_grid`;
4. evaluate accuracy and `0-1` risk on every anchor in `A_ID`;
5. save per-environment predictions or at minimum `(n_correct, n_total, risk)` so results can be recomputed without loading the model again.

Output one tidy record per:

```text
method, seed, support_condition, eval_environment
```

Required columns:

```text
method
seed
support_condition
lambda_eval_or_NA
eval_environment
n_eval
accuracy
risk_01
checkpoint_id
```

Use the same final checkpoint rule for every method.

---

## 6. Common training protocol for the final paper

For every principal method retained in the final table:

```text
seeds = [0, 1, 2, 3, 4]
total_update_calls = 600
checkpoint = final
support-specific hyperparameter tuning = forbidden
target-based checkpoint selection = forbidden
target-based lambda selection = forbidden
```

The support condition is the intervention. A method's hyperparameters must not change between balanced, long-tail, scarce-tail, and missing-tail.

The paper currently states the following architecture; verify it against the executed code before retaining it:

```text
FiLMedMLP
input: flattened image
hidden layers: 390, 390
dropout: 0.2 immediately after input/hidden linear layers
activation: ReLU
output: linear logits
initialization: Xavier uniform weights, zero biases
```

Do not silently change architecture or initialisation during the rerun.

---

## 7. Exact discrete CVaR implementation used for all post-training analysis

Post-training identification must **not** use the neural thresholded tail-mean surrogate.

Implement one canonical routine for a discrete law `(values, weights)`:

1. assert non-negative weights and `sum(weights) == 1` up to numerical tolerance;
2. sort atoms by risk from largest to smallest;
3. set required tail mass `q = 1 - alpha`;
4. take atoms from the top until `q` mass is filled;
5. if the final atom crosses the boundary, include only the fractional boundary mass;
6. return the weighted tail sum divided by `q`.

This routine is used for:

- synthetic figure;
- unrestricted identification endpoints;
- realised deployment CVaR in the four-anchor law;
- structured-completion endpoints.

Do not use a quantile-thresholded unweighted average for paper post-processing.

Unit tests:

```text
CVaR_alpha(constant c) == c
CVaR_0(discrete law) == weighted mean
CVaR approaches max atom as alpha -> 1
for law [0,1] with weights [0.75,0.25], CVaR_0.75 == 1
```

---

## 8. Regenerate the deterministic synthetic experiment

Replace the old 240-configuration study with one exact deterministic construction.

Use:

```text
observed domains = 3
missing domains = 1
conditional observed weights = [1/3, 1/3, 1/3]

r_obs(f) = [0.10, 0.15, 0.20]
r_obs(g) = [0.30, 0.35, 0.40]
M = 1
alpha = 0.9
epsilon_grid = arange(0.0, 0.300 + 0.005, 0.005)

pi_obs_each(epsilon) = (1 - epsilon) / 3
pi_missing(epsilon) = epsilon

World A: r_missing(f)=0, r_missing(g)=1
World B: r_missing(f)=1, r_missing(g)=0
```

Generate:

```text
figures/identification_ranking.png
```

The figure must contain:

- lower and upper identified endpoints for `f` and `g` versus epsilon;
- the deployment-CVaR difference under World A and World B;
- no Monte Carlo error bars because the construction is deterministic.

Delete / stop generating:

```text
figures/ranking_reversal_by_missing_mass.pdf
```

unless it is used outside the paper.

---

## 9. Unrestricted `(alpha, epsilon)` identification sensitivity

Use **only final checkpoints from the missing-tail training condition** for this sensitivity analysis.

For each method `m` and seed `s`, construct:

```text
P_obs = uniform law over risks at anchors [0.1, 0.2, 0.5]
```

with predictor fixed as specified in Section 2.3.

Sweep:

```text
alpha_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
              0.6, 0.7, 0.75, 0.8, 0.9, 0.95]

epsilon_grid = [0.0, 0.05, 0.10, 0.15, 0.20,
                 0.25, 0.30, 0.40, 0.50]
```

For every `(alpha, epsilon)`:

```text
P_minus = (1-epsilon) * P_obs + epsilon * delta_0
P_plus  = (1-epsilon) * P_obs + epsilon * delta_1
L = exact_discrete_cvar(P_minus, alpha)
U = exact_discrete_cvar(P_plus, alpha)
width = U - L
```

For every same-seed unordered method pair `(m1, m2)`:

```text
certified = (U_m1 < L_m2) OR (U_m2 < L_m1)
```

Aggregate:

```text
mean_width(alpha, epsilon) = mean over methods and seeds
cert_rate(alpha, epsilon)  = certified pairs / (5 * choose(K,2))
```

where `K` is the final method count after the reproducibility audit.

Generate:

```text
figures/cmnist_identification_phase_diagram.pdf
```

Recommended two-panel layout:

- left: heatmap of mean width;
- right: heatmap of certification rate.

Overlay:

```text
alpha = 1 - epsilon
```

as the structural saturation boundary and:

```text
epsilon = 0.25
```

as the actual four-anchor missing-tail deployment slice.

Also generate the actual `epsilon=0.25` summary table at:

```text
tables/cmnist_missing_tail_intervals.tex
```

Include at least:

```text
alpha in [0.5, 0.75, 0.9]
method
mean lower endpoint over 5 seeds
mean realised deployment CVaR over 5 seeds
mean upper endpoint over 5 seeds
mean width over 5 seeds
```

Do not headline the number of failed certificates. At `alpha >= 0.75`, failure is structurally forced by `epsilon = 0.25` under unrestricted completion.

---

## 10. Structured-completion sensitivity

This analysis uses the same missing-tail checkpoints and the actual deployment mass:

```text
epsilon = 0.25
```

For each method / seed, on observed anchors `O = [0.1, 0.2, 0.5]`, compute:

```text
L_obs(method, seed) = max over e != e' in O:
    abs(risk[e] - risk[e']) / abs(e - e')
```

Define one common feasibility threshold:

```text
L0 = max over all final methods and all 5 seeds of L_obs(method, seed)
```

Sweep:

```text
L_grid = [L0, 1.5*L0, 2*L0, 4*L0, infinity]
alpha_structured = [0.5, 0.75, 0.9]
```

For finite `L`, compute the admissible risk interval for the missing anchor `e_m = 0.9`:

```text
lower = max(
    0,
    max_{e in O} (risk[e] - L * abs(0.9 - e))
)

upper = min(
    1,
    min_{e in O} (risk[e] + L * abs(0.9 - e))
)
```

For `L = infinity`, set:

```text
lower = 0
upper = 1
```

Then form the lower and upper four-anchor deployment laws using mass `0.25` on the missing anchor and `0.25` on each observed anchor, and compute exact CVaR.

Aggregate width and same-seed certification rate exactly as in the unrestricted analysis.

Generate:

```text
figures/cmnist_structured_completion_sensitivity.pdf
```

The figure should show both:

- interval width versus `L/L0` for each alpha;
- pairwise certification rate versus `L/L0` for each alpha.

Sanity check:

```text
result at L=infinity == unrestricted result at epsilon=0.25
```

Do not describe any finite `L` as empirically validated from the three observed anchors alone. It is a sensitivity assumption.

---

## 11. Regenerate the main CMNIST performance table

After the GroupDRO rerun and the fixed-lambda evaluation refresh, generate:

```text
tables/cmnist_support_performance.tex
```

The file should contain the complete LaTeX table environment and be included by the main paper with:

```latex
\input{tables/cmnist_support_performance.tex}
```

Required columns:

```text
Method | Balanced | Long-tail | Scarce-tail | Missing-tail
```

Entries:

```text
mean worst-environment accuracy over 5 seeds +/- standard deviation
```

Worst-environment accuracy is over the 11-point OOD evaluation grid.

Do not bold a winner and do not describe a method as strongest / best. The table is a realised-performance diagnostic, not an algorithm ranking claim.

---

## 12. Generate an exact hyperparameter table from the executed manifest

Create:

```text
tables/cmnist_hyperparameters.tex
```

It must be generated from the configurations that actually produced the retained runs.

At minimum, include:

```text
Method
Architecture
Initialization
Pretraining updates
Main optimizer
Learning rate
Weight decay
Schedule
Total updates
Method-specific coefficient(s)
Preference sampling, if applicable
Number of preference samples, if applicable
Beta / adaptive preference settings, if applicable
```

No `[TODO]` may remain for VREx, EQRM, or any other retained method.

If the exact setting cannot be recovered, rerun or drop the method.

---

## 13. Visible-imbalance control

Keep this control because all four anchors remain observed while frequency imbalance changes.

Do not rerun it unless one of the following is true:

- the current checkpoint metadata cannot be recovered;
- the data pairing audit fails and the paper wants to claim paired controls;
- IRO / INF-TASK were evaluated with target-selected or inconsistent lambda values.

If only the preference-conditioned evaluation input is the issue, re-evaluate the saved checkpoints at `lambda_eval = 0.9`; retraining is unnecessary.

Update the text to emphasize:

```text
visible imbalance can change realised performance while epsilon remains 0;
therefore imbalance and structural nonidentification are different mechanisms.
```

---

## 14. Optional but valuable: theory-aligned IRO domain-CVaR robustness run

This is optional and should not delay the core revision if compute is limited.

Run only:

```text
method variant: IRO-domainCVaR
conditions: [balanced, missing_tail]
seeds: [0,1,2,3,4]
number of runs: 10
```

Keep the same architecture, data, preference sampler, optimizer policy, total budget, and fixed evaluation input as the principal IRO run.

Change only the training risk aggregation:

For each sampled conditioning input `lambda`:

1. compute the mean loss separately in every active source domain;
2. build the vector of domain mean losses;
3. compute exact discrete CVaR across active domains with equal domain mass and fractional boundary mass;
4. backpropagate through that domain-level aggregation;
5. average across sampled lambda values as in the IRO update.

Label this implementation variant explicitly. Do not replace the reference IRO results silently.

If completed, generate:

```text
tables/iro_domain_cvar_robustness.tex
```

and add the conditional LaTeX block in `LATEX_REVISION_BLOCKS.md`.

---

## 15. Do not complete ImageNet for this submission unless committing to a full protocol

Current ImageNet results should be removed from the submission.

Do not retain two-seed results or failed IRM / VREx rows as a nominal replication.

Only reintroduce ImageNet if all of the following can be completed before submission:

```text
>= 5 seeds for every retained method
all four support conditions or a new clearly justified design
one frozen hyperparameter configuration per method across support conditions
validated non-degenerate optimisation for every retained baseline
complete dataset / corruption / severity / class-subset specification
same checkpoint-selection rule as CMNIST
complete reproducibility manifest
```

Otherwise archive the current ImageNet outputs outside the paper.

---

## 16. Required run manifest

For every retained run, save one machine-readable record, preferably JSONL or CSV.

Required fields:

```text
run_id
code_revision
seed
dataset
support_condition
source_anchors
source_counts
eval_grid
method
architecture
initialization
pretraining_steps
total_updates
optimizer_schedule
learning_rate
weight_decay
method_hyperparameters
lambda_training_scheme_or_NA
lambda_eval_or_NA
checkpoint_path
final_checkpoint_step
```

For evaluation outputs additionally record:

```text
eval_environment
n_eval
accuracy
risk_01
```

For post-processing outputs additionally record:

```text
alpha
epsilon
completion_type                 # unrestricted / structured
structured_L_or_NA
lower_cvar
upper_cvar
width
realised_deployment_cvar_or_NA
```

This manifest is the source of truth for all paper tables.

---

## 17. Validation checks before updating LaTeX

### Data / support invariants

```text
sum(source_counts) == 8000 for every principal condition
missing_tail count at e=0.9 == 0
all other principal conditions have positive counts at all 4 anchors
A_ID has exactly 4 anchors
pi_star sums to 1
actual missing_tail epsilon == 0.25
```

### Evaluation invariants

```text
all principal methods have 5 seeds
all IRO / INF-TASK principal evaluations use one fixed lambda_eval
no target-selected lambda
no target-selected checkpoint
```

### Budget invariants

```text
all retained principal methods have 600 update calls
GroupDRO no longer has a 1000-update principal result
```

### CVaR invariants

At actual `epsilon=0.25`:

```text
for alpha >= 0.75, unrestricted upper endpoint == 1
```

For structured sensitivity:

```text
L=infinity reproduces unrestricted epsilon=0.25 endpoints
lower <= upper for every run
```

### Paper consistency grep

Before submission, these strings should have no active-paper occurrences except where explicitly discussed historically:

```text
Near-missing
near-missing
epsilon=7/11
epsilon=8/11
ImageNet-100-C
240-configuration
150 unsuccessful
Domain-count control
Preference sensitivity across
[TODO
Problem Fromulation
crtify
interva lusing
unrestirected
```

Also check for the malformed tokens that currently exist in the main source:

```text
\\mathcal{U}N
\\mathsf{P}{f,u}
\\mathcal{P}N
r_a(f\\lambda)
```

---

## 18. Paper artefacts that must exist before compilation

Required:

```text
figures/identification_ranking.png
figures/cmnist_identification_phase_diagram.pdf
figures/cmnist_structured_completion_sensitivity.pdf

tables/cmnist_support_performance.tex
tables/cmnist_missing_tail_intervals.tex
tables/cmnist_hyperparameters.tex
```

Optional:

```text
tables/iro_domain_cvar_robustness.tex
```

Removed / no longer referenced by the paper:

```text
figures/ranking_reversal_by_missing_mass.pdf
ImageNet paper tables/figures
Domain-count paper figure/table
single-seed lambda-sensitivity paper figure
matched-budget GroupDRO diagnostic figure/table
```

---

## 19. Recommended execution order

1. **Freeze the conceptual protocol** in Sections 2 and 6 of this README.
2. **Audit existing checkpoints and configs**; decide the final method set.
3. **Verify paired data generation** across methods for each seed/condition.
4. **Rerun GroupDRO** at 600 total updates: 20 runs.
5. **Conditionally rerun or drop VREx/EQRM** if settings are not recoverable.
6. **Re-evaluate all final checkpoints**, fixing `lambda_eval = 0.9` for IRO / INF-TASK.
7. **Regenerate the CMNIST performance table**.
8. **Regenerate the deterministic synthetic figure**.
9. **Run unrestricted `(alpha, epsilon)` post-processing** and render the phase diagram + actual-slice table.
10. **Run structured-completion sensitivity** and render its figure.
11. **Optionally run the 10 IRO-domainCVaR robustness runs**.
12. **Generate the hyperparameter table from the executed manifest**.
13. **Apply `LATEX_REVISION_BLOCKS.md`**.
14. **Run the consistency grep / validation checks**.
15. **Compile main + supplement from a clean build directory** and inspect all references, captions, tables, and figure paths.

---

## 20. Minimal rerun count under the expected good case

If the existing five-seed CMNIST runs are paired and all non-GroupDRO configurations are recoverable:

```text
Mandatory new neural training:
    GroupDRO: 20 runs

Mandatory checkpoint re-evaluation:
    all retained methods x 5 seeds x 4 conditions
    (cheap relative to training)

Mandatory post-processing:
    no neural training

Optional theory-aligned IRO robustness:
    10 runs

ImageNet:
    0 runs; remove from submission
```

This is the preferred revision path.
