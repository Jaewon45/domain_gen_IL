# Domain Generalisation via Imprecise Learning

This is the active submission workspace. The primary evidence is CMNIST support removal; ImageNet-100-C is an external constructed-domain replication. The upstream repository README is preserved unchanged in [README_REFERENCE.md](README_REFERENCE.md).

## Active entry points

- [REPRODUCIBILITY.md](REPRODUCIBILITY.md): submission release gates and sanity checks.
- [CMNIST/README.md](CMNIST/README.md): CMNIST training and seven-method E3b commands.
- [IMAGENET100C/README.md](IMAGENET100C/README.md): four-anchor ImageNet-100-C protocol and canonical runner.
- [docs/RESULTS.md](docs/RESULTS.md): interpretation of tracked artifacts.

The retired non-100 ImageNet-C planning material is under `docs/legacy/`; do not run it as part of the active protocol.

## Installation

CMNIST:

```bash
cd CMNIST
python -m pip install -r requirements.txt
```

ImageNet-100-C:

```bash
cd IMAGENET100C
python -m pip install -r requirements.txt
```

Run the relevant unit tests before launching a sweep. The active commands use fixed final checkpoints and write into versioned output roots; do not reuse an existing run directory.
