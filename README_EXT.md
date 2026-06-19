# README Extension: CMNIST Stress/Lambda Workflow

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

## Commands

### Full Sweep (Generate + Run)

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

### Per-Seed Reduced Main Run (Regenerate + Run)

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

