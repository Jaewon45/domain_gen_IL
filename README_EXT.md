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
- Reduced multi-seed roots: `results/cmnist_exp_small_seed0`, `results/cmnist_exp_small_seed1`, `results/cmnist_exp_small_seed2`
- Export root: `results/export`

## Commands

### Full Sweep

1. Generate full sweep command file (`domain_stress.txt`):
```bash
cd CMNIST
..\dgil_env\Scripts\python.exe job_scripts\gen_exps.py --exp_name domain_stress --data_dir c:/Users/<USER_ID>/PycharmProjects/domain_gen_IL/data --output_dir c:/Users/<USER_ID>/PycharmProjects/domain_gen_IL/results/cmnist_exp
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

### Reduced Multi-Seed Sweep (pipeline validation)

1. Run reduced multi-seed sweep (script already writes under `results/`):
```bash
cd CMNIST/job_scripts
bash run_domain_stress_small_seeds.sh
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
- Keep E1 domain-count conditions: all 4
- Keep E2 sample-size conditions: all 3
- Reduce E3 imbalance conditions from 5 to 3:
	- balanced: `2000,2000,2000,2000`
	- last-domain-heavy-strong: `2000,2000,2000,10000`
	- first-domain-heavy-strong: `10000,2000,2000,2000`

### Staged Seed Runs (Recommended)

Run seeds in two batches so you can validate intermediate outputs before committing the full reduced scope.

- Batch A: seeds 0,1,2
- Batch B: seeds 3,4

Prepared command files (generated under `CMNIST/job_scripts/`):

- `domain_stress_main_seed012.txt` (150 jobs)
- `domain_stress_main_seed34.txt` (100 jobs)

CMD run commands:

```cmd
set REPO_ROOT=%USERPROFILE%\PycharmProjects\domain_gen_IL
cd /d %REPO_ROOT%\CMNIST

REM Batch A: seeds 0,1,2
for /f "usebackq delims=" %i in ("%REPO_ROOT%\CMNIST\job_scripts\domain_stress_main_seed012.txt") do @echo Running: %i & cmd /c "%i" >> "%REPO_ROOT%\domain_stress_main_seed012.log" 2>&1

REM Batch B: seeds 3,4
for /f "usebackq delims=" %i in ("%REPO_ROOT%\CMNIST\job_scripts\domain_stress_main_seed34.txt") do @echo Running: %i & cmd /c "%i" >> "%REPO_ROOT%\domain_stress_main_seed34.log" 2>&1
```

### Scope Comparison

- Current full scope:
	- jobs: 600
	- total steps: 408000
	- previously projected remaining runtime: about 26 days (throughput-based)

- Suggested reduced main scope:
	- jobs: 250
	- total steps: 170000
	- relative compute: about 41.7% of full scope

### Estimated Runtime With Suggested Scope

Using the same observed throughput baseline used for the full-sweep estimate, expected wall-clock runtime is:

- about 10.8 to 11.0 days

Practical planning range (to account for run-to-run variance and interruptions):

- about 9 to 14 days

Per staged batch estimate (same baseline):

- Batch A (150 jobs): about 6.5 to 6.7 days
- Batch B (100 jobs): about 4.3 to 4.5 days

