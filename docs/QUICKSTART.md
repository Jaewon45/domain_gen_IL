# Quickstart Guide: Reproduce Report Results

This guide walks you through reproducing the eight report items from "Domain Generalization via Imprecise Learning" (ICML 2024). It covers environment setup, experiment execution, result analysis, and theory postprocessing.

**Estimated total time:** 
- Smoke test (E0, one seed, ~10 conditions): **~30 minutes**
- Full reproduction (E1–E4, five seeds, ~450 conditions): **~20–24 hours**
- Theory postprocessing: **< 1 minute**

---

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Quick Smoke Test](#quick-smoke-test-e0)
3. [Full Experiment Pipeline](#full-experiment-pipeline)
4. [Theory Postprocessing](#theory-postprocessing)
5. [Verification & Report Generation](#verification--report-generation)
6. [Troubleshooting](#troubleshooting)

---

## Environment Setup

### Prerequisites

- **Python 3.10** (required; code uses f-string syntax incompatible with Python 3.8)
- **CUDA 11.x or CPU-only PyTorch** (GPU recommended for training speed)
- **~50 GB disk space** for data and results

### Step 1: Clone and Navigate

```bash
git clone https://github.com/muandet-lab/dgil.git
cd dgil
```

### Step 2: Create Virtual Environment

```bash
# On Windows (PowerShell)
python -m venv dgil_env
.\dgil_env\Scripts\Activate.ps1

# On macOS/Linux
python -m venv dgil_env
source dgil_env/bin/activate
```

### Step 3: Install Dependencies

**CMNIST only (primary experiments):**
```bash
cd CMNIST
pip install -r requirements.txt
cd ..
```

**Full repository (includes simulations & optional ImageNet-C scaffolding):**
```bash
# Core CMNIST dependencies
cd CMNIST
pip install -r requirements.txt
cd ..

# Note: ImageNet-C requires pretrained ResNet-50 and ImageNet-C data.
# See IMAGENET_C/README_[ImgC].md for setup (optional, not required for report).
```

### Step 4: Verify Setup

```bash
# Check Python version
python --version  # Should print 3.10.x

# Quick import check
python -c "import torch; import pandas; import numpy; print('✓ All imports successful')"
```

---

## Quick Smoke Test (E0)

The smoke test validates that the entire pipeline works end-to-end on a small configuration (seed 0, 3 test environments, 13 commands). Runtime: **~30 minutes**.

### Run E0

```bash
cd CMNIST

# Generate reduced commands (if not already present)
python -m job_scripts.gen_exps \
  --exp_name domain_stress_small \
  --data_dir ../data \
  --output_dir ../results/cmnist_exp_small

# Display the commands (optional)
cat job_scripts/domain_stress_small.txt

# Run all 13 commands sequentially (WARNING: takes ~30 min)
# On Windows (PowerShell):
Get-Content job_scripts/domain_stress_small.txt | Invoke-Expression

# On macOS/Linux (bash):
source job_scripts/domain_stress_small.txt

cd ..
```

### Verify E0 Output

```bash
# Check that results were created
ls -la results/cmnist_exp_small/results/

# Expected: One subdirectory per experiment phase (e.g., domain_stress/reproduction/)
# and JSONL files with training records
```

### Generate E0 Summary Plot

```bash
cd CMNIST
python plot_domain_stress.py ../results/cmnist_exp_small/results
cd ..

# Expected: Plot files in results/cmnist_exp_small/plots/
```

---

## Full Experiment Pipeline

The full pipeline consists of four experiment phases (E1–E3b) plus one lambda-sensitivity phase (E4). Each phase takes **~5 hours per seed** on a GPU.

### Phase E1: Domain-Count Stress (2/4/6/8 source domains)

**Time per seed:** ~5 hours  
**Seeds:** 5 (0–4)  
**Total:** ~25 hours

```bash
cd CMNIST

# 1. Generate commands for E1 (creates job_scripts/domain_count_clean.txt)
python -m job_scripts.gen_exps \
  --exp_name domain_count_clean \
  --data_dir ../data \
  --output_dir ../results/cmnist_domain_count_clean_v1 \
  --phases domain_count

# 2. Run commands (sequential or via job launcher like submitit)
# Option A: Sequential (WARNING: ~25 hours)
source job_scripts/domain_count_clean.txt  # bash
# OR on Windows PowerShell:
Get-Content job_scripts/domain_count_clean.txt | Invoke-Expression

# Option B: Via submitit on HPC cluster
python -m job_scripts.submit_jobs \
  -c job_scripts/domain_count_clean.txt \
  --time 30

# 3. Analyze results
python collect_results.py ../results/cmnist_domain_count_clean_v1/results
python export_results_csv.py \
  ../results/cmnist_domain_count_clean_v1/results \
  --output_dir ../results/export \
  --prefix cmnist_e1_domain_count

# 4. Generate plots
python plot_domain_stress.py ../results/cmnist_domain_count_clean_v1/results

cd ..
```

**Expected artifacts:**
- CSV summaries: `results/export/cmnist_e1_domain_count_*.csv`
- Plots: `results/cmnist_domain_count_clean_v1/plots/`
- **Report Item 3 (Table 2 + Figure 3):** Worst-domain accuracy vs. domain count

---

### Phase E3: Imbalance Stress (balanced / first-heavy / last-heavy)

**Time per seed:** ~5 hours  
**Seeds:** 5 (0–4)  
**Total:** ~25 hours

```bash
cd CMNIST

# 1. Generate commands for E3
python -m job_scripts.gen_exps \
  --exp_name imbalance_clean \
  --data_dir ../data \
  --output_dir ../results/imbalance_clean_v1 \
  --phases imbalance

# 2. Run commands (sequential or HPC)
source job_scripts/imbalance_clean.txt  # bash

# 3. Analyze results
python collect_results.py ../results/imbalance_clean_v1/results
python analyze_imbalance_summary.py ../results/imbalance_clean_v1/results
python export_results_csv.py \
  ../results/imbalance_clean_v1/results \
  --output_dir ../results/export \
  --prefix cmnist_e3_imbalance

# 4. Generate plots
python plot_domain_stress.py ../results/imbalance_clean_v1/results

cd ..
```

**Expected artifacts:**
- CSV summaries: `results/export/cmnist_e3_imbalance_*.csv`
- Plots: `results/imbalance_clean_v1/plots/`
- **Report Item 4 (Table 3 + Figure 4):** Worst-domain accuracy vs. imbalance condition

---

### Phase E3b: Tail-Support Analysis (missing-tail stress)

**Time per seed:** ~5 hours  
**Seeds:** 5 (0–4)  
**Total:** ~25 hours

This is the principal CMNIST evidence in the paper.

```bash
cd CMNIST

# 1. Generate commands for E3b (four support conditions × 5 algorithms × 5 seeds = 100 jobs)
python -m job_scripts.gen_exps \
  --exp_name e3b_tail_support \
  --data_dir ../data \
  --output_dir ../results/E3b_tail_support \
  --phases tail_support

# 2. Run commands
source job_scripts/e3b_tail_support.txt  # bash

# 3. Analyze and generate all E3b tables/plots
python analyze_tail_support.py ../results/E3b_tail_support/results

cd ..
```

**Expected artifacts:**
- Raw results: `results/E3b_tail_support/raw_results.csv`
- Summary tables: `results/E3b_tail_support/analysis_seed0-4/summary_by_condition.csv`
- Plots: `results/E3b_tail_support/analysis_seed0-4/e3b_*.png`
- **Report Item 1 (Table 1):** Five-seed tail-support accuracy means
- **Report Item 2 (Figure 2):** Missing-tail CVaR interval bars

---

### Phase E4: Lambda Sensitivity (preference-sensitive risk)

**Time per seed:** ~2 hours  
**Seeds:** 1 (seed 0 only; not 5-seed evidence)  
**Total:** ~2 hours

⚠️ **Note:** E4 uses a fixed seed-0 checkpoint and explores λ sensitivity. This is **not** intended as 5-seed evidence.

```bash
cd CMNIST

# 1. Ensure E3b checkpoints are available
# E4 evaluation requires trained IRO and INF-TASK checkpoints from E3b
# Check:
ls -la ../results/E3b_tail_support/checkpoints/

# 2. Evaluate lambda grid (post-hoc on saved checkpoints)
python evaluate_lambda_grid.py \
  --checkpoint_dir ../results/E3b_tail_support/checkpoints \
  --output_dir ../results/cmnist_exp/lambda_results \
  --seed 0

# 3. Generate lambda plots
python plot_lambda_prediction.py ../results/cmnist_exp/lambda_results

cd ..
```

**Expected artifacts:**
- CSV results: `results/cmnist_exp/lambda_results/prediction_lambda_*.csv`
- Plots: `results/cmnist_exp/lambda_results/lambda_prediction_*.png`
- **Report Item 8 (Figure 6):** Preference sensitivity (fixed seed 0 only)

---

## Theory Postprocessing

After E3b experiments complete, run the theory postprocessing to generate Priority 7 results (ranking-reversal and identification-width analyses).

**Time:** < 1 minute  
**Inputs:** E3b checkpoint results  
**Outputs:** Theory tables and figures (Report Items 1 & 6)

### Run Theory Postprocessing

```bash
cd ..  # Back to repository root

# Run all theory tasks (Tasks 1–4):
# Task 1: Theorem-aligned 0-1 CVaR bounds (uses E3b missing-tail checkpoints)
# Task 2: Synthetic ranking-reversal specification recovery
# Task 3: Same-predictor identification-width simulation
# Task 4: Ranking-reversal probability vs. missing deployment mass
python run_posthoc_and_theory_sims.py
```

### Theory Postprocessing Outputs

All outputs are written to `results/P7_theory/`:

```
results/P7_theory/
├── theorem_aligned_01_cvar_bounds.csv          (Report Table 4: CVaR identification)
├── theorem_aligned_01_cvar_bounds.pdf
├── theorem_aligned_01_cvar_bounds.png
├── ranking_reversal_by_missing_mass.csv        (Report Figure 1: ranking-reversal probability)
├── ranking_reversal_by_missing_mass.pdf
├── ranking_reversal_by_missing_mass.png
├── same_predictor_identification_width.csv
├── same_predictor_identification_width.pdf
├── same_predictor_identification_width.png
├── synthetic_risk_profile_spec.txt
└── README.md                                    (Theory outputs documentation)
```

**Expected results:**
- **Report Item 1 (Figure 1):** Ranking-reversal probability at ε=0,0.1,0.2,0.3 should be ~0.636, 0.945, 0.998, 1.000
- **Report Item 6 (Table 4):** CVaR identification widths at α=0.9 should match: ERM 0.546, GroupDRO 0.549, INF-TASK 0.613, IRM 0.610, IRO 0.615

---

## Verification & Report Generation

### Step 1: Verify All Canonical Outputs Exist

```bash
# Check that all experiment phases produced results
ls -la results/cmnist_domain_count_clean_v1/results/
ls -la results/imbalance_clean_v1/results/
ls -la results/E3b_tail_support/results/
ls -la results/P7_theory/

# Expected: JSONL files and CSV summaries in each
```

### Step 2: Export All Results to Report-Grade CSVs

```bash
cd CMNIST

# Export E1, E3, E3b, E4, and GroupDRO control results
for prefix in e1_domain_count e3_imbalance e3b_tail_support e4_lambda_control groupdro_control; do
  python export_results_csv.py \
    ../results/cmnist_exp/results \
    --output_dir ../results/export \
    --prefix "cmnist_$prefix"
done

cd ..
```

### Step 3: Generate Summary Tables

All analysis scripts already generated summaries; verify they exist:

```bash
# E1 domain-count summary
cat results/cmnist_domain_count_clean_v1/plots/domain_count_summary.csv

# E3 imbalance summary
cat results/imbalance_clean_v1/plots/imbalance_summary.csv

# E3b tail-support summary (5 seeds × 4 conditions × 5 algorithms = 100)
cat results/E3b_tail_support/analysis_seed0-4/summary_by_condition.csv

# Theory outputs
ls -la results/P7_theory/
```

### Step 4: Map Results to Report Items

| Report Item | Source | File(s) | Regeneration Command |
|---|---|---|---|
| Figure 1: Ranking Reversal | Priority 7 | `results/P7_theory/ranking_reversal_by_missing_mass.{pdf,png}` | `python run_posthoc_and_theory_sims.py` (Task 4) |
| Table 1: Tail-Support Stress | E3b | `results/E3b_tail_support/analysis_seed0-4/summary_by_condition.csv` | `python CMNIST/analyze_tail_support.py` |
| Figure 2: Missing-Tail CVaR Intervals | E3b | `results/E3b_tail_support/analysis_seed0-4/e3b_tail_accuracy_by_condition.png` | `python CMNIST/analyze_tail_support.py` |
| Table 2 + Figure 3: Domain-Count Control | E1 | `results/cmnist_domain_count_clean_v1/plots/domain_count_summary.csv` + plot | `python CMNIST/plot_domain_stress.py` |
| Table 3 + Figure 4: Imbalance Control | E3 | `results/imbalance_clean_v1/plots/imbalance_summary.csv` + plot | `python CMNIST/plot_domain_stress.py` |
| Table 4: CVaR Identification | Priority 7 | `results/P7_theory/theorem_aligned_01_cvar_bounds.csv` | `python run_posthoc_and_theory_sims.py` (Task 1) |
| Table 5 + Figure 5: ERM/GroupDRO Diagnostic | GroupDRO Control | `results/cmnist_groupdro_control_v2/` | See [GroupDRO Control](#groupdro-matched-budget-control-optional) |
| Figure 6: Preference Sensitivity (λ) | E4 | `results/cmnist_exp/lambda_results/lambda_prediction_accuracy_curve.png` | `python CMNIST/plot_lambda_prediction.py` |

---

## Advanced: Full Scripted Reproduction

If you want to automate the entire pipeline (not recommended without HPC cluster):

```bash
#!/bin/bash
# run_all_experiments.sh
# WARNING: This takes 20+ hours on a GPU; use HPC cluster with job launcher

set -e  # Exit on any error

echo "=== DOMAIN GENERALIZATION VIA IMPRECISE LEARNING - FULL REPRODUCTION ==="

# Step 1: Environment
python --version
pip list | grep torch

# Step 2: E1 Domain Count
cd CMNIST
python -m job_scripts.gen_exps --exp_name domain_count_clean --phases domain_count
python -m job_scripts.submit_jobs -c job_scripts/domain_count_clean.txt --time 30  # HPC only
python analyze_domain_count.py ../results/cmnist_domain_count_clean_v1/results
python plot_domain_stress.py ../results/cmnist_domain_count_clean_v1/results
cd ..

# Step 3: E3 Imbalance
cd CMNIST
python -m job_scripts.gen_exps --exp_name imbalance_clean --phases imbalance
python -m job_scripts.submit_jobs -c job_scripts/imbalance_clean.txt --time 30  # HPC only
python analyze_imbalance_summary.py ../results/imbalance_clean_v1/results
python plot_domain_stress.py ../results/imbalance_clean_v1/results
cd ..

# Step 4: E3b Tail Support (Principal)
cd CMNIST
python -m job_scripts.gen_exps --exp_name e3b_tail_support --phases tail_support
python -m job_scripts.submit_jobs -c job_scripts/e3b_tail_support.txt --time 30  # HPC only
python analyze_tail_support.py ../results/E3b_tail_support/results
cd ..

# Step 5: E4 Lambda (Post-hoc)
cd CMNIST
python evaluate_lambda_grid.py --checkpoint_dir ../results/E3b_tail_support/checkpoints --seed 0
python plot_lambda_prediction.py ../results/cmnist_exp/lambda_results
cd ..

# Step 6: Theory Postprocessing
python run_posthoc_and_theory_sims.py

echo "✓ All experiments complete. Check results_submit/ and results/P7_theory/ for report artifacts."
```

---

## Troubleshooting

### Common Issues

**Q: "ImportError: No module named 'torch'"**  
A: Virtual environment not activated or PyTorch not installed.
```bash
# Activate environment
source dgil_env/bin/activate  # macOS/Linux
.\dgil_env\Scripts\Activate.ps1  # Windows

# Reinstall PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Q: "MNIST data not found"**  
A: Data is downloaded automatically on first run, but requires internet.
```bash
# Pre-download MNIST
cd CMNIST
python -c "from torchvision import datasets; datasets.MNIST('../data', download=True)"
cd ..
```

**Q: "CUDA out of memory"**  
A: Reduce batch size or switch to CPU.
```bash
# Use CPU (slow, but works)
cd CMNIST
python train_sandbox.py --algorithm erm --device cpu --steps 1000
cd ..
```

**Q: "Job script not found"**  
A: Make sure you're in the CMNIST directory when running `gen_exps.py`.
```bash
cd CMNIST
python -m job_scripts.gen_exps --help  # Should show usage
```

**Q: "Theory postprocessing fails with 'raw_results.csv not found'"**  
A: E3b experiments must have completed first.
```bash
# Check E3b output
ls -la results/E3b_tail_support/analysis_seed0-4/raw_results.csv
# If missing, re-run analyze_tail_support.py
```

---

## Reporting Issues

If experiments fail or produce unexpected results:

1. **Check logs:** Examine `CMNIST/job_scripts/logs/` for error messages
2. **Verify setup:** Run `python --version` and `python -c "import torch; print(torch.cuda.is_available())"`
3. **Test on smoke data:** Run E0 first to validate pipeline
4. **Check GitHub Issues:** See [Known Issues](../README.md#known-issues)

---

## References

- **Main paper:** [Domain Generalization via Imprecise Learning](https://arxiv.org/abs/...)
- **Detailed documentation:** See [docs/](../) for phase-by-phase experiment design
- **Code documentation:** See [CMNIST/README.md](../CMNIST/README.md) for algorithm details
- **Theory audit:** See [docs/implementation_theory_audit.md](implementation_theory_audit.md) for CVaR and theorem correctness

---

**Last updated:** 2026-09-24  
**Python version:** 3.10+  
**Estimated setup time:** 15 minutes (env + deps)
