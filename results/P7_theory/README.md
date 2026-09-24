# P7 Theory Outputs

This directory contains the canonical Priority 7 (theorem-aligned) theory simulation and validation outputs.

## Files

| File | Type | Source | Description |
|---|---|---|---|
| `theorem_aligned_01_cvar_bounds.csv` | CSV | `run_posthoc_and_theory_sims.py::run_task1()` | Per-method lower/realised deployment/upper CVaR bounds at α=0.9 for missing-tail condition |
| `theorem_aligned_01_cvar_bounds.pdf` | Figure | `run_posthoc_and_theory_sims.py::run_task1()` | Bar chart of CVaR bounds with error bars (Report Figure 2) |
| `theorem_aligned_01_cvar_bounds.png` | Figure | `run_posthoc_and_theory_sims.py::run_task1()` | PNG version of CVaR bounds chart |
| `same_predictor_identification_width.csv` | CSV | `run_posthoc_and_theory_sims.py::run_task3()` | Theoretical vs. numerical identification width at α ∈ {0.5, 0.75, 0.9} across ε ∈ {0, 0.05, 0.1, 0.2, 0.3} |
| `same_predictor_identification_width.pdf` | Figure | `run_posthoc_and_theory_sims.py::run_task3()` | Identification width vs. missing deployment mass ε (Report Figure attached in audit) |
| `same_predictor_identification_width.png` | Figure | `run_posthoc_and_theory_sims.py::run_task3()` | PNG version |
| `ranking_reversal_by_missing_mass.csv` | CSV | `run_posthoc_and_theory_sims.py::run_task4()` | Mean ranking-reversal probability across 1,000 repetitions for each ε |
| `ranking_reversal_by_missing_mass.pdf` | Figure | `run_posthoc_and_theory_sims.py::run_task4()` | Ranking-reversal probability vs. ε (Report Figure 1) |
| `ranking_reversal_by_missing_mass.png` | Figure | `run_posthoc_and_theory_sims.py::run_task4()` | PNG version |
| `synthetic_risk_profile_spec.txt` | Text | `run_posthoc_and_theory_sims.py::run_task2()` | Specification of synthetic ranking-reversal experiment: K, risk profiles, bounds, etc. |

## Generation

To regenerate all P7 theory outputs:

```bash
cd c:\Users\320257223\PycharmProjects\domain_gen_IL
python run_posthoc_and_theory_sims.py
```

**Dependencies:**
- `results/E3b_tail_support/` must exist (E3b tail-support stress experiment results)
- `results/cmnist_priority7_theory_v1/ranking_reversal_summary.csv` must exist (ranking-reversal synthetic simulation)

**Duration:** < 1 minute

## Report Integration

These outputs are used in the main paper:
- **Report Item 1 (Figure 1):** `ranking_reversal_by_missing_mass.{pdf,png}` 
- **Report Item 2 (Figure 2):** `theorem_aligned_01_cvar_bounds.{pdf,png}`
- **Report Item 6 (Table 4):** `theorem_aligned_01_cvar_bounds.csv`, `same_predictor_identification_width.csv`

## History

**2026-09-24 Consolidation:** All theory outputs were previously duplicated across:
- Repository root (now deleted)
- `results/` (now moved to `results/P7_theory/`)
- `results_submit/additional/`
- `results_submit/figures/`
- `results_submit/figures/P7_theory/`
- `results_submit/tables/P7_theory/`

Canonical location is now exclusively `results/P7_theory/`. Root-level files are added to `.gitignore` to prevent re-accumulation. Duplicates under `results_submit/` should be removed or converted to symlinks if the submission bundle requires them.
