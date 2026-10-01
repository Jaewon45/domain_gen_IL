# CMNIST

The active CMNIST entry point is `train_sandbox.py`. It implements binary Colored MNIST with the FiLMedMLP architecture and supports ERM, IRM, VREx, EQRM, GroupDRO, INF-TASK, and IRO.

## Setup and checks

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m py_compile train_sandbox.py datasets.py algorithms.py
```

## Canonical E3b support-removal sweep

The source anchors are `{0.1, 0.2, 0.5, 0.9}`. The generator defaults to seeds `0,1,2,3,4`, uses final checkpoints, and writes all runs to the output root supplied by `--output_dir`.

```bash
python -m job_scripts.gen_exps --exp_name e3b_tail_support --data_dir ../data --output_dir ../results/cmnist/e3b --seed_list 0,1,2,3,4
python -m job_scripts.gen_exps --exp_name e3b_tail_support_eqrm_vrex --data_dir ../data --output_dir ../results/cmnist/e3b --seed_list 0,1,2,3,4
source job_scripts/e3b_tail_support.txt
source job_scripts/e3b_tail_support_eqrm_vrex.txt
python analyze_tail_support.py ../results/cmnist/e3b/results --output_dir ../results/cmnist/e3b/analysis/eleven_grid_predictive
```

The first manifest supplies ERM, IRM, GroupDRO, IRO, and INF-TASK. The second supplies VREx and EQRM under the same source counts, seeds, checkpoint policy, 400-step ERM warm-up, and cosine schedule. The analyser’s 11-grid output is a predictive metric only; do not use it as the four-anchor identification audit. Historical upstream-QRM reproduction commands, including negative EQRM quantile settings, are retained only in `QRM/` and are not part of this protocol.
