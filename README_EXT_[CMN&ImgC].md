# README Extension: [CMN] CMNIST and [ImgC] ImageNet-C Workflow

This file keeps workflow-specific instructions separate from the existing README files.

## Environment
Use the existing virtual environment:
- `dgil_env`

## Output Layout Policy
Write experiment folders under `results/` (not directly under repository root).

Recommended roots:
- Full sweep root: `results/cmnist_exp`
- Reduced sweep root: `results/cmnist_exp_small`
- Reduced multi-seed roots: `results/cmnist_exp_small_seed0`, `results/cmnist_exp_small_seed1`, `results/cmnist_exp_small_seed2`, `results/cmnist_exp_small_seed3`, `results/cmnist_exp_small_seed4`
- Export root: `results/export`

Artifact preservation rules:
- Never reuse an existing experiment output directory for a new dataset extension.
- Never point new ImageNet-C runs at CMNIST or prior E3b result roots.
- For new phases, create a fresh root such as `results/imagenet_c_eval_repeatability_v1`, `results/imagenet_c_fold_generalization_v1`, or `results/imagenet_c_support_stress_v1`.
- If a proposed root already exists from prior exploratory work, create a new suffixed root such as `_v2` or a date-stamped variant instead of overwriting files.

## Commands

### [ImgC] ImageNet-C Extension (Adopted Additional-Dataset Plan)

This section records the currently preferred additional-dataset direction. It does not replace the historical CMNIST workflows below.

Design summary:
- Native task: keep the original 1000-class ImageNet label space.
- Primary domain variable: corruption type.
- Severity levels `1-5` are retained as a secondary analysis axis rather than treated as unrelated domains.
- Initial architecture: frozen pretrained ResNet-50 backbone with a trainable 1000-class head.
- Core algorithms for the first comparison: `ERM`, `GroupDRO`, `IRO`.
- Lambda evaluation should follow the existing post-training grid style: `0.0:1.0:0.1`.

Stability must be checked at three levels:
- evaluation repeatability for a fixed checkpoint,
- training stability across seeds,
- stability across corruption domains and severity.

Recommended execution order:
1. Deterministic evaluation repeatability on one fixed pretrained checkpoint, repeated 3 times.
2. One-fold, one-seed smoke test for held-out corruption generalization.
3. Three-seed pilot across all 3 corruption folds.
4. Expand to 5 seeds only if the pilot is interpretable.
5. Run corruption-support stress only after the held-out-corruption pipeline is validated.

Planned output roots:
- `results/imagenet_c_eval_repeatability_smoke_v1`
- `results/imagenet_c_eval_repeatability_v1`
- `results/imagenet_c_fold_generalization_v1`
- `results/imagenet_c_support_stress_v1`

Planned implementation surface:
- `IMAGENET_C/README.md`
- `IMAGENET_C/datasets.py`
- `IMAGENET_C/features.py`
- `IMAGENET_C/models.py`
- `IMAGENET_C/train_head.py`
- `IMAGENET_C/eval_repeatability.py`
- `IMAGENET_C/evaluate_lambda_grid.py`
- `IMAGENET_C/analyze_imagenet_c.py`
- `IMAGENET_C/folds.json`
- `IMAGENET_C/job_scripts/`

Current implementation status:
- The separate [ImgC] code surface is implemented for synthetic smoke evaluation/training and real-data scaffolding.
- Synthetic repeatability, fold-generalization, and lambda-grid smoke artifacts exist under `results/imagenet_c_*`.
- No report-grade real-data ImageNet-C training or evaluation has been completed.
- `analyze_imagenet_c.py` exports summaries from training records; it does not yet produce every report artifact listed below.

Expected report-grade metrics:
- clean ImageNet accuracy,
- mean corruption accuracy,
- worst-corruption accuracy,
- mean and worst corruption loss,
- CVaR across corruption types,
- severity-wise performance,
- corruption-by-severity heatmap,
- lambda sensitivity,
- mean and standard deviation across seeds.

Planned artifact layout for Experiment Set 1 (`results/imagenet_c_eval_repeatability_v1/`):
- `raw/repetition_0.jsonl`
- `raw/repetition_1.jsonl`
- `raw/repetition_2.jsonl`
- `summary/repeatability_summary.csv`
- `summary/corruption_severity_matrix.csv`
- `plots/corruption_severity_heatmap.png`
- `plots/corruption_accuracy_bar.png`
- `plots/lambda_sensitivity_curve.png`
- `manifests/eval_repeatability_repetitions.txt`

Planned artifact layout for Experiment Set 2 (`results/imagenet_c_fold_generalization_v1/`):
- `raw/train_runs.jsonl`
- `raw/lambda_eval.jsonl`
- `summary/by_fold_seed.csv`
- `summary/by_fold_algorithm.csv`
- `summary/held_out_only.csv`
- `summary/all_corruptions.csv`
- `plots/held_out_worst_domain.png`
- `plots/held_out_mean_accuracy.png`
- `plots/severity_profile_by_algorithm.png`
- `plots/corruption_severity_heatmap_<algorithm>_<fold>.png`
- `plots/lambda_sensitivity_<fold>.png`
- `manifests/fold_a_seed0_smoke.txt`
- `manifests/pilot_seed012.txt`
- `manifests/final_seed01234.txt`

Implemented smoke commands and real-data command scaffolds:

The first executable run should be a smoke run that validates layout and repeatability plumbing without requiring ImageNet assets.

```cmd
set REPO_ROOT=%USERPROFILE%\PycharmProjects\domain_gen_IL
cd /d %REPO_ROOT%
dgil_env\Scripts\python.exe IMAGENET_C\eval_repeatability.py --smoke --output_dir results\imagenet_c_eval_repeatability_smoke_v1 --repetitions 3 --lambda_grid 0.0:1.0:0.1
```

Experiment Set 1 manifests should contain 3 repeated evaluation commands over the same fixed checkpoint and same output root subpaths.

```cmd
set REPO_ROOT=%USERPROFILE%\PycharmProjects\domain_gen_IL
cd /d %REPO_ROOT%
dgil_env\Scripts\python.exe IMAGENET_C\eval_repeatability.py --imagenet_root %IMAGENET_ROOT% --imagenet_c_root %IMAGENET_C_ROOT% --checkpoint_path %CHECKPOINT_PATH% --output_dir %REPO_ROOT%\results\imagenet_c_eval_repeatability_v1 --repetitions 3 --lambda_grid 0.0:1.0:0.1
```

Note: the real-data evaluation scaffold uses `--imagenet_val_root` and `--imagenet_c_root` rather than a single `--imagenet_root`.

The clean split is loaded with `torchvision.datasets.ImageNet(root=<imagenet_root>, split='val', ...)`.

```cmd
set REPO_ROOT=%USERPROFILE%\PycharmProjects\domain_gen_IL
cd /d %REPO_ROOT%
dgil_env\Scripts\python.exe IMAGENET_C\eval_repeatability.py --imagenet_val_root %IMAGENET_VAL_ROOT% --imagenet_c_root %IMAGENET_C_ROOT% --checkpoint_path %CHECKPOINT_PATH% --output_dir results\imagenet_c_eval_repeatability_v1 --repetitions 3 --lambda_grid 0.0:1.0:0.1
```

Preferred argument form:

```cmd
set REPO_ROOT=%USERPROFILE%\PycharmProjects\domain_gen_IL
cd /d %REPO_ROOT%
dgil_env\Scripts\python.exe IMAGENET_C\eval_repeatability.py --imagenet_root %IMAGENET_ROOT% --imagenet_c_root %IMAGENET_C_ROOT% --checkpoint_path %CHECKPOINT_PATH% --output_dir results\imagenet_c_eval_repeatability_v1 --repetitions 3 --lambda_grid 0.0:1.0:0.1
```

Experiment Set 2 smoke-test manifest should target one fold, one seed, and the 3 core algorithms.

```cmd
set REPO_ROOT=%USERPROFILE%\PycharmProjects\domain_gen_IL
cd /d %REPO_ROOT%
dgil_env\Scripts\python.exe IMAGENET_C\train_head.py --imagenet_root %IMAGENET_ROOT% --imagenet_c_root %IMAGENET_C_ROOT% --fold fold_a --algorithms erm,groupdro,iro --seeds 0 --output_dir %REPO_ROOT%\results\imagenet_c_fold_generalization_v1 --batch_size 256 --max_epochs 20 --patience 3 --learning_rate 3e-3 --weight_decay 1e-4
```

Current smoke command:

```cmd
set REPO_ROOT=%USERPROFILE%\PycharmProjects\domain_gen_IL
cd /d %REPO_ROOT%
dgil_env\Scripts\python.exe IMAGENET_C\train_head.py --smoke --fold fold_a --algorithms erm,groupdro,iro --seeds 0 --output_dir results\imagenet_c_fold_generalization_smoke_v1
```

Real-data training scaffold:

```cmd
set REPO_ROOT=%USERPROFILE%\PycharmProjects\domain_gen_IL
cd /d %REPO_ROOT%
dgil_env\Scripts\python.exe IMAGENET_C\train_head.py --fold fold_a --algorithms erm,groupdro,iro --seeds 0 --imagenet_train_corrupted_root %IMAGENET_TRAIN_C_ROOT% --imagenet_val_root %IMAGENET_VAL_ROOT% --imagenet_c_root %IMAGENET_C_ROOT% --output_dir results\imagenet_c_fold_generalization_v1 --batch_size 64 --max_epochs 20 --patience 3
```

Preferred argument form:

```cmd
set REPO_ROOT=%USERPROFILE%\PycharmProjects\domain_gen_IL
cd /d %REPO_ROOT%
dgil_env\Scripts\python.exe IMAGENET_C\train_head.py --fold fold_a --algorithms erm,groupdro,iro --seeds 0 --imagenet_train_corrupted_root %IMAGENET_TRAIN_C_ROOT% --imagenet_root %IMAGENET_ROOT% --imagenet_c_root %IMAGENET_C_ROOT% --output_dir results\imagenet_c_fold_generalization_v1 --batch_size 64 --max_epochs 20 --patience 3
```

Experiment Set 2 pilot manifest should expand only after the smoke test is verified.

```cmd
set REPO_ROOT=%USERPROFILE%\PycharmProjects\domain_gen_IL
cd /d %REPO_ROOT%
for %f in (fold_a fold_b fold_c) do @dgil_env\Scripts\python.exe IMAGENET_C\train_head.py --imagenet_root %IMAGENET_ROOT% --imagenet_c_root %IMAGENET_C_ROOT% --fold %f --algorithms erm,groupdro,iro --seeds 0,1,2 --output_dir %REPO_ROOT%\results\imagenet_c_fold_generalization_v1 --batch_size 256 --max_epochs 20 --patience 3 --learning_rate 3e-3 --weight_decay 1e-4
```

Lambda-grid evaluation on saved checkpoints:

```cmd
set REPO_ROOT=%USERPROFILE%\PycharmProjects\domain_gen_IL
cd /d %REPO_ROOT%
dgil_env\Scripts\python.exe IMAGENET_C\evaluate_lambda_grid.py results\imagenet_c_fold_generalization_smoke_v1\ckpts --output_dir results\imagenet_c_fold_generalization_smoke_lambda_v1 --lambda_grid 0.0:1.0:0.1
```

See `IMAGENET_C/README.md` for the concrete folder plan, output layout, and run naming convention.

### [ImgC] Preparing Kaggle ImageNet Localization Data

Goal:
- keep the original Kaggle dataset unchanged,
- build a separate torchvision-compatible clean ImageNet root,
- continue loading ImageNet-C corruptions with `ImageFolder`.

Expected roots:
- `KAGGLE_ROOT`: Kaggle ImageNet Localization Challenge download (unchanged source)
- `IMAGENET_ROOT`: prepared clean ImageNet root with `meta.bin` and `val/<wnid>/...`
- `IMAGENET_C_ROOT`: official ImageNet-C root with 15 corruption folders

Example Windows layout:

```text
D:\datasets\
|-- imagenet_kaggle\
|   |-- ILSVRC\
|   |-- LOC_val_solution.csv
|   `-- LOC_synset_mapping.txt
|
|-- imagenet_prepared\
|   |-- meta.bin
|   `-- val\
|       `-- <1000 wnid folders>
|
`-- imagenet_c\
		`-- <15 corruption folders>
```

PowerShell procedure:

```powershell
$REPO_ROOT = "C:\Users\320257223\PycharmProjects\domain_gen_IL"
$KAGGLE_ROOT = "D:\datasets\imagenet_kaggle"
$IMAGENET_ROOT = "D:\datasets\imagenet_prepared"
$IMAGENET_C_ROOT = "D:\datasets\imagenet_c"
$PY = "$REPO_ROOT\dgil_env\Scripts\python.exe"
```

Step A - dry run:

```powershell
& $PY `
	"$REPO_ROOT\IMAGENET_C\prepare_kaggle_imagenet.py" `
	--kaggle_root $KAGGLE_ROOT `
	--output_root $IMAGENET_ROOT `
	--link_mode auto `
	--dry_run
```

Step B - create prepared dataset and verify clean loader:

```powershell
& $PY `
	"$REPO_ROOT\IMAGENET_C\prepare_kaggle_imagenet.py" `
	--kaggle_root $KAGGLE_ROOT `
	--output_root $IMAGENET_ROOT `
	--link_mode auto `
	--verify
```

Step C - direct torchvision check:

```powershell
& $PY -c "from torchvision.datasets import ImageNet; d=ImageNet(r'$IMAGENET_ROOT', split='val'); print('images=',len(d)); print('wnids=',len(d.wnids)); print('mapping=',len(d.wnid_to_idx))"
```

Step D - ImageNet-C structure and mapping check:

```powershell
& $PY `
	"$REPO_ROOT\IMAGENET_C\validate_data_layout.py" `
	--imagenet_root $IMAGENET_ROOT `
	--imagenet_c_root $IMAGENET_C_ROOT
```

Optional exhaustive ImageNet-C check (loads all 75 conditions):

```powershell
& $PY `
	"$REPO_ROOT\IMAGENET_C\validate_data_layout.py" `
	--imagenet_root $IMAGENET_ROOT `
	--imagenet_c_root $IMAGENET_C_ROOT `
	--full_verify
```

Then inspect real evaluation CLI flags and run real repeatability:

```powershell
& $PY "$REPO_ROOT\IMAGENET_C\eval_repeatability.py" --help

$STAMP = Get-Date -Format "yyyyMMdd_HHmm"
$OUT_EVAL = "$REPO_ROOT\results\imagenet_c_eval_repeatability_$STAMP"

& $PY "$REPO_ROOT\IMAGENET_C\eval_repeatability.py" `
	--imagenet_root $IMAGENET_ROOT `
	--imagenet_c_root $IMAGENET_C_ROOT `
	--repetitions 3 `
	--output_dir $OUT_EVAL
```

Notes:
- The preparation script creates `prepare_summary.json` and `prepare_manifest.csv` under `IMAGENET_ROOT`.
- `link_mode auto` prefers hard links on same volume and falls back to copies if hard links are unavailable.
- The source Kaggle dataset is never modified.

### [CMN] E3b/E2 Tail-Support Stress Test (Main Presentation Experiment)

This experiment replaces the prior E2 sample-size stress test in the seminar narrative while keeping E2 code available.

- Experiment name options:
	- `e3b_tail_support` (preferred)
	- `e2_tail_support` (alternate slide numbering)
- Conditions (fixed source budget where possible):
	- `balanced_visible`: `2000,2000,2000,2000`
	- `long_tail_visible`: `5000,2000,800,200`
	- `near_missing_tail`: `5800,1800,350,50`
	- `missing_tail`: `6000,1500,500,0`

Generate command files:

```cmd
set REPO_ROOT=%USERPROFILE%\PycharmProjects\domain_gen_IL
cd /d %REPO_ROOT%\CMNIST
..\dgil_env\Scripts\python.exe job_scripts\gen_exps.py --exp_name e3b_tail_support --data_dir %REPO_ROOT%\data --output_dir %REPO_ROOT%\results\E3b_tail_support --seed_list 0,1,2
```

Run generated jobs:

```cmd
set REPO_ROOT=%USERPROFILE%\PycharmProjects\domain_gen_IL
cd /d %REPO_ROOT%\CMNIST
for /f "usebackq delims=" %i in ("%REPO_ROOT%\CMNIST\job_scripts\e3b_tail_support.txt") do @echo Running: %i & cmd /c "%i" >> "%REPO_ROOT%\e3b_tail_support_run.log" 2>&1
```

Evaluate lambda grid from checkpoints (IRO/INF-TASK + baseline-compatible risk aggregation):

```cmd
set REPO_ROOT=%USERPROFILE%\PycharmProjects\domain_gen_IL
cd /d %REPO_ROOT%\CMNIST
..\dgil_env\Scripts\python.exe evaluate_lambda_grid.py %REPO_ROOT%\results\E3b_tail_support\ckpts --output_dir %REPO_ROOT%\results\E3b_tail_support\lambda_results --lambda_grid 0.0:1.0:0.1
```

Build required CSV outputs and plots:

```cmd
set REPO_ROOT=%USERPROFILE%\PycharmProjects\domain_gen_IL
cd /d %REPO_ROOT%\CMNIST
..\dgil_env\Scripts\python.exe analyze_tail_support.py %REPO_ROOT%\results\E3b_tail_support\results --lambda_results_dir %REPO_ROOT%\results\E3b_tail_support\lambda_results --output_dir %REPO_ROOT%\results\E3b_tail_support
```

Expected artifacts under `results/E3b_tail_support/`:

- `raw_results.csv`
- `summary_by_condition.csv`
- `slide_table.csv`
- `tail_accuracy_by_condition.png`
- `worst_accuracy_by_condition.png`
- `head_tail_gap_by_condition.png`
- `cvar_gap_by_condition.png`
- `iro_lambda_tail_accuracy_by_condition.png`
- `iro_lambda_cvar_by_condition.png`

Important behavior for `missing_tail`:

- The missing source domain receives zero empirical prior mass.
- Zero-count source domains are removed from training loaders (to avoid empty-loader failures).
- The same domain remains in the fixed test/evaluation set and gets positive deployment prior mass.

### [CMN] Full Sweep (Generate + Run)

1. Generate full sweep command file (`domain_stress.txt`):
```cmd
set REPO_ROOT=%USERPROFILE%\PycharmProjects\domain_gen_IL
cd /d %REPO_ROOT%\CMNIST
..\dgil_env\Scripts\python.exe job_scripts\gen_exps.py --exp_name domain_stress --data_dir %REPO_ROOT%\data --output_dir %REPO_ROOT%\results\cmnist_exp
```

2. Run full sweep from generated command file (CMD, logs to `domain_stress_run.log`):
```cmd
set REPO_ROOT=%USERPROFILE%\PycharmProjects\domain_gen_IL
cd /d %REPO_ROOT%\CMNIST
for /f "usebackq delims=" %i in ("%REPO_ROOT%\CMNIST\job_scripts\domain_stress.txt") do @echo Running: %i & cmd /c "%i" >> "%REPO_ROOT%\domain_stress_run.log" 2>&1
```

3. Export CSVs:
```bash
cd CMNIST
..\dgil_env\Scripts\python.exe export_results_csv.py ../results/cmnist_exp/results --output_dir ../results/export --prefix cmnist_exp
```

4. Plot E0-E3:
```bash
cd CMNIST
..\dgil_env\Scripts\python.exe plot_domain_stress.py ../results/cmnist_exp/results --output_dir ../results/cmnist_exp/plots
```

5. Evaluate lambda grid (E4) on saved checkpoints:
```bash
cd CMNIST
..\dgil_env\Scripts\python.exe evaluate_lambda_grid.py ../results/cmnist_exp/ckpts --output_dir ../results/cmnist_exp/lambda_results --lambda_grid 0.0:1.0:0.1
```

### [CMN] Per-Seed Reduced Main Run (Regenerate + Run)

Use this when `domain_stress_main_seed*.txt` files were deleted or you want to refresh them.

1. Regenerate full master command file first:
```cmd
set REPO_ROOT=%USERPROFILE%\PycharmProjects\domain_gen_IL
cd /d %REPO_ROOT%\CMNIST
..\dgil_env\Scripts\python.exe job_scripts\gen_exps.py --exp_name domain_stress --data_dir %REPO_ROOT%\data --output_dir %REPO_ROOT%\results\cmnist_exp
```

2. Regenerate per-seed command files (`seed0` through `seed4`):
```cmd
set REPO_ROOT=%USERPROFILE%\PycharmProjects\domain_gen_IL
cd /d %REPO_ROOT%\CMNIST\job_scripts
..\..\dgil_env\Scripts\python.exe gen_reduced_seed_files.py --source domain_stress.txt --seeds 0,1,2,3,4 --heavy_size 10000 --output_prefix domain_stress_main_seed
```

3. Run one seed file at a time (example: `seed0`):
```cmd
set REPO_ROOT=%USERPROFILE%\PycharmProjects\domain_gen_IL
cd /d %REPO_ROOT%\CMNIST
for /f "usebackq delims=" %i in ("%REPO_ROOT%\CMNIST\job_scripts\domain_stress_main_seed0.txt") do @echo Running: %i & cmd /c "%i" >> "%REPO_ROOT%\domain_stress_main_seed0.log" 2>&1
```

4. Repeat step 3 for seed1-seed4 by changing file/log suffix.

Optional: CMD loop to run all seed files sequentially:
```cmd
set REPO_ROOT=%USERPROFILE%\PycharmProjects\domain_gen_IL
cd /d %REPO_ROOT%\CMNIST
for %s in (0 1 2 3 4) do @for /f "usebackq delims=" %i in ("%REPO_ROOT%\CMNIST\job_scripts\domain_stress_main_seed%s.txt") do @echo [seed%s] Running: %i & cmd /c "%i" >> "%REPO_ROOT%\domain_stress_main_seed%s.log" 2>&1
```

## Current Blockers (Tracking)
- Multi-seed longer E1 confirmation (rerun at longer schedule across seeds)
- Optional E1 and E3 extra plots (test-env curves per condition)
- E4 lambda-conditioned metrics, heatmap, robustness summary
- Table-to-CSV exact match and explicit missing-row callouts in captions

## Next Concrete Closure Steps
1. Rerun E1 at longer schedule across seeds and mark multi-seed confirmation resolved.
2. Complete E4 lambda-conditioned metrics, heatmap, and robustness summary.
3. Verify reported table values match CSV exports exactly; call out any missing rows.

## Methods Pass: Reduced vs Full Sweep

This section records the final reduced-vs-full documentation pass.

- Reduced sweep:
	- command source: CMNIST/job_scripts/domain_stress_small.txt
	- purpose: fast pipeline validation for E0-E3 with limited settings
	- typical outputs: results/cmnist_exp_small*/...

- Full sweep:
	- command source: CMNIST/job_scripts/domain_stress.txt
	- purpose: publication-grade E0-E3 grid across full algorithm and seed coverage
	- typical outputs: results/cmnist_exp/...

- E4 lambda evaluation:
	- not part of the E0-E3 training grid; run post-training from checkpoints
	- entry point: CMNIST/evaluate_lambda_grid.py

- Interpretation policy:
	- smoke and reduced results are labeled as reduced/smoke evidence
	- full-sweep claims require full grid completion and multi-seed confirmation

## Suggested Reduced Main Scope (Report-Grade, Not Smoke)

This scope is designed to keep the study publication/report oriented while reducing total runtime.

- Keep all 5 algorithms: ERM, IRM, GroupDRO, IRO, INF-TASK
- Reduce seeds from 10 to 5: use seeds 0-4
- Keep E1 domain-count conditions: all 3 (`2`, `4`, `8`)
- Keep E2 sample-size conditions: all 3
- Reduce E3 imbalance conditions from 5 to 3:
	- balanced: `2000,2000,2000,2000`
	- mild-imbalance: `2000,2000,2000,4000`
	- strong-imbalance: `2000,2000,2000,10000`

### Staged Seed Runs (Recommended)

Run seeds as separate files so you can execute `n=3` first and then add `n=2` without rerunning anything.

Generated per-seed command files (under `CMNIST/job_scripts/`):

- `domain_stress_main_seed0.txt` (45 jobs)
- `domain_stress_main_seed1.txt` (45 jobs)
- `domain_stress_main_seed2.txt` (45 jobs)
- `domain_stress_main_seed3.txt` (45 jobs)
- `domain_stress_main_seed4.txt` (45 jobs)

Generator command:

```cmd
cd /d %USERPROFILE%\PycharmProjects\domain_gen_IL\CMNIST\job_scripts
..\..\dgil_env\Scripts\python.exe gen_reduced_seed_files.py --source domain_stress.txt --seeds 0,1,2,3,4 --heavy_size 10000 --output_prefix domain_stress_main_seed
```

Run one seed file at a time (example: seed 0):

```cmd
set REPO_ROOT=%USERPROFILE%\PycharmProjects\domain_gen_IL
cd /d %REPO_ROOT%\CMNIST
for /f "usebackq delims=" %i in ("%REPO_ROOT%\CMNIST\job_scripts\domain_stress_main_seed0.txt") do @echo Running: %i & cmd /c "%i" >> "%REPO_ROOT%\domain_stress_main_seed0.log" 2>&1
```

Repeat for `seed1` through `seed4` by changing the filename/log suffix.

### Scope Comparison

- Current full scope:
	- jobs: 450
	- total steps: 306000
	- previously projected remaining runtime: about 26 days (throughput-based)

- Suggested reduced main scope:
	- jobs: 225
	- total steps: 153000
	- relative compute: about 50.0% of full scope

### Estimated Runtime With Suggested Scope

**Updated from clean seed 1 run (2026-06-19):** Seed 1 completed 45/45 jobs with 0 tracebacks in 3h56m54s, yielding ~5.26 min/job average. This supersedes prior throughput baseline.

Expected wall-clock runtime for 225 jobs (5 seeds × 45 jobs):

- About **19.7 to 23.7 hours** (about **0.82 to 0.99 days**)

Practical planning range (to account for run-to-run variance and interruptions):

- About **1 to 3 days**

Per-seed estimate:

- Each seed file (45 jobs): about **3.9 to 4.7 hours**

Staged plan estimate:

- First pass (`n=3`, seeds 0/1/2): about **11.8 to 14.1 hours**
- Additional pass (`n=2`, seeds 3/4): about **7.9 to 9.4 hours**

**Note:** A previous estimate overstated total runtime due to an arithmetic error (using total log seconds instead of per-job seconds for scaling). The seed 1 measurement above is the current best baseline.

