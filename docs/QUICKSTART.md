# Active experiment quickstart

Use this document only for the active CMNIST E3b and ImageNet-100-C support-removal protocols. Historical workflow plans are in `docs/legacy/`.

## CMNIST E3b

```bash
cd CMNIST
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m job_scripts.gen_exps --exp_name e3b_tail_support --data_dir ../data --output_dir ../results/cmnist/e3b --seed_list 0,1,2,3,4
python -m job_scripts.gen_exps --exp_name e3b_tail_support_eqrm_vrex --data_dir ../data --output_dir ../results/cmnist/e3b --seed_list 0,1,2,3,4
source job_scripts/e3b_tail_support.txt
source job_scripts/e3b_tail_support_eqrm_vrex.txt
python analyze_tail_support.py ../results/cmnist/e3b/results --output_dir ../results/cmnist/e3b/analysis/eleven_grid_predictive
```

The two manifests form the seven-method sweep. They use final checkpoints and seeds `0--4`. The generated 11-grid analysis is for worst-environment prediction; the four-anchor audit is intentionally not run by this quickstart.

## ImageNet-100-C E3b

```bash
python -m pip install -r IMAGENET100C/requirements.txt
python -m unittest discover -s IMAGENET100C/tests -v
bash IMAGENET100C/run_e3b.sh --seed 0 --results-root results/imagenet100c --gpus 0,1
bash IMAGENET100C/run_evaluate_e3b.sh --seed 0 --results-root results/imagenet100c --gpus 0,1
```

Repeat the canonical runner for the predeclared ImageNet seed set. It runs all seven methods across Balanced, Long-tail, Near-missing, and Missing; each run writes to `results/imagenet100c/seed<seed>/E3b_<condition>_<algorithm>/`.

For the exact data, model, and transition-policy contract, see [IMAGENET100C/README.md](../IMAGENET100C/README.md). For submission release gates, see [REPRODUCIBILITY.md](../REPRODUCIBILITY.md).
