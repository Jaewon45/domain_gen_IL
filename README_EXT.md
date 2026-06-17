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

1. Generate full sweep command file (`domain_stress.txt`):
```bash
cd CMNIST
..\dgil_env\Scripts\python.exe job_scripts\gen_exps.py --exp_name domain_stress --data_dir c:/Users/320257223/PycharmProjects/domain_gen_IL/data --output_dir c:/Users/320257223/PycharmProjects/domain_gen_IL/results/cmnist_exp
```

2. Run reduced multi-seed sweep (script already writes under `results/`):
```bash
cd CMNIST/job_scripts
bash run_domain_stress_small_seeds.sh
```

3. Export CSVs from a reduced run:
```bash
cd CMNIST
..\dgil_env\Scripts\python.exe export_results_csv.py ../results/cmnist_exp_small/results --output_dir ../results/export --prefix cmnist_exp_small
```

4. Plot E0-E3:
```bash
cd CMNIST
..\dgil_env\Scripts\python.exe plot_domain_stress.py ../results/cmnist_exp_small/results --output_dir ../results/cmnist_exp_small/plots
```

5. Evaluate lambda grid (E4) on saved checkpoints:
```bash
cd CMNIST
..\dgil_env\Scripts\python.exe evaluate_lambda_grid.py ../results/cmnist_exp_small/ckpts --output_dir ../results/cmnist_exp_small/lambda_results --lambda_grid 0.0:1.0:0.1
```

## Current Blockers (Tracking)
- Multi-seed longer E1 confirmation
- Failed-run status/error marker
- No silent missing grouping values
- E0 summary table (avg, worst-domain, regret)
- Optional E1 and E3 extra plots
- E4 lambda-conditioned metrics, heatmap, robustness summary
- Duplicate inflation removal
- Grouped summary consistency across phases/conditions
- README run instructions and final reduced-vs-full doc pass
- Table-to-CSV exact match and explicit missing-row callouts

## Next Concrete Closure Steps
1. Deduplicate run-level records and regenerate exports.
2. Auto-generate an E0 summary table from run-level CSV.
3. Keep run/collect/plot instructions in this file and close doc-pass checklist after review.

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
