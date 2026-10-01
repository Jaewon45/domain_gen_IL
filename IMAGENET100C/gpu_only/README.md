# GPU-only E3b benchmark

This directory is an isolated performance benchmark. It does not modify the
ImageNet100C training or evaluation pipeline and does not produce paper
checkpoints.

The benchmark uses the active E3b manifest to stage fixed-size uint8 images on
CPU, then measures a 1,000-step four-domain training loop where sampling,
corruption, resize/crop/normalize, model forward/backward, and optimization run
on CUDA. JPEG decoding and the one-time fixed-size staging pass remain CPU-side
and are reported separately.

Example after the current paper training jobs finish:

```bash
cd /home/ra95tig/lrz_mount/domgen
CUDA_VISIBLE_DEVICES=0 /home/ra95tig/anaconda3/envs/domgen/bin/python \
  -m IMAGENET100C.gpu_only.benchmark_e3b \
  --manifest /home/ra95tig/imagenet100c_results/seed3/E3b_balanced_erm/manifest.json \
  --steps 1000 --batch-size 64 --output-dir /home/ra95tig/imagenet100c_gpu_benchmark
```

The implementation is intentionally an efficiency benchmark, not an exact
replacement for the CPU ImageNet-C backend. Its GPU corruption operators are
Torch approximations and must not be used for paper results without a separate
numerical validation study.

## GPU-native training test

To run an approximate GPU-native E3b training job and save a checkpoint plus
JSONL history:

```bash
CUDA_VISIBLE_DEVICES=0 python -m IMAGENET100C.gpu_only.train_e3b \
  --manifest /path/to/E3b_balanced_erm/manifest.json \
  --algorithm erm --steps 600 --batch-size 64 \
  --output-dir /tmp/imagenet100c_gpu_erm
```

Use `--images-per-domain 1000` for a smaller staging pool. The default `0`
stages every assigned source image. Outputs are deliberately separate from
paper results and are marked approximate.
