# ImageNet100C Pipeline Optimization

This document describes the current ImageNet100C pipeline and the isolated GPU-native optimization benchmark. The optimization path is intended for performance experiments; it is not an automatic replacement for the exact paper protocol.

## Scope

The pipeline has three distinct stages:

1. Dataset loading and image staging.
2. Corruption and image transformation.
3. Model training or evaluation.

The current paper pipeline preserves the existing CPU ImageNet-C implementation. The GPU-native path moves tensor operations to CUDA after a one-time CPU staging pass.

## Before: current exact pipeline

| Stage | CPU tasks | GPU tasks |
| --- | --- | --- |
| Dataset loading | Hugging Face row access, JPEG/image decoding, RGB conversion | None |
| Image corruption | PIL and `imagecorruptions` corruption generation for Gaussian noise, defocus blur, snow, and contrast | None |
| Image transformation | Resize, crop, tensor conversion, normalization | None |
| Batch preparation | Sampling, DataLoader iteration, batch assembly, CPU-to-GPU transfer | None until the batch arrives |
| Training | Most input preparation work; thread and DataLoader overhead | ResNet-50 forward pass, loss, backward pass, optimizer update |
| Evaluation | Image loading, corruption, transformation, DataLoader iteration | ResNet-50 inference, loss, top-1/top-5 metrics |
| Caching | All cache generation work, tensor serialization | None |

The exact path is reproducible because it reuses the existing PIL/NumPy corruption backend, manifest ordering, and seed-dependent sampling.

## After: GPU-native optimization path

| Stage | CPU tasks | GPU tasks |
| --- | --- | --- |
| Dataset loading | One-time JPEG/image decoding and fixed-size staging to uint8 tensors | None |
| Image corruption | No repeated corruption work after staging | Torch-native Gaussian noise, contrast, blur, and snow approximations |
| Image transformation | One-time fixed-size staging only | Resize, crop, tensor conversion, normalization |
| Batch preparation | Minimal scheduling and host-to-device setup | Sampling from staged pools and batch assembly |
| Training | Initial staging and lightweight orchestration | Corruption, preprocessing, ResNet-50 forward pass, loss, backward pass, optimizer update |
| Evaluation | One-time exact cache generation if exact inputs are required | Cached tensor transfer, ResNet-50 inference, loss, and metrics |
| Caching | Optional one-time staging/cache creation | Optional GPU-generated cache for approximate experiments |

## Implemented isolated benchmark

The isolated benchmark is located at:

- `IMAGENET100C/gpu_only/benchmark_e3b.py`
- `IMAGENET100C/gpu_only/README.md`

It uses an active E3b manifest, stages 1,000 images per domain on the CPU, and then times a 1,000-step four-domain GPU loop. The benchmark reports CPU staging and GPU training times separately.

The reference run recorded:

| Measurement | Runtime |
| --- | ---: |
| CPU staging for 1,000 images per domain | 1,277.2 seconds |
| GPU-native approximate 1,000-step ERM loop | 120.7 seconds |
| GPU loop throughput | 8.28 steps/second |

The benchmark's corruption operators are approximate. Its outputs must not be mixed with exact paper results without numerical validation.

## Exact evaluation cache

The exact evaluation cache is implemented by:

- `IMAGENET100C/build_eval_cache.py`
- `IMAGENET100C/evaluate_cached.py`

`build_eval_cache.py` deliberately uses the existing CPU corruption path so that cached tensors match the original evaluation protocol. One cache can be reused by all methods for the same seed, dataset revision, transform, image count, corruption set, and severity set.

After the cache is complete, `evaluate_cached.py` performs the repeated model inference on GPU and avoids regenerating the corrupted images for every checkpoint.

## Reproducibility rules

The GPU-native path changes the numerical protocol unless all of the following are demonstrated:

- Identical image ordering and sample indices.
- Identical corruption parameters and random-number mapping.
- Matching resize, crop, interpolation, and normalization semantics.
- Deterministic CUDA behavior where applicable.
- Agreement between CPU and GPU outputs within a documented tolerance.

Therefore:

- Use the exact CPU path for paper checkpoints and final reported results.
- Use the exact CPU-generated cache for reproducible repeated evaluations.
- Use `gpu_only` for throughput experiments and future optimization work.
- Do not compare GPU-native approximate checkpoints directly with CPU-protocol paper checkpoints without a validation study.

## Recommended optimization sequence

1. Keep the exact CPU pipeline as the reference implementation.
2. Stage or cache decoded images once per seed and protocol.
3. Replace one corruption at a time with a Torch implementation.
4. Compare CPU and GPU corruption outputs on fixed images and seeds.
5. Benchmark end-to-end throughput with the same batch size and number of steps.
6. Only promote a GPU implementation to the paper path after numerical and distributional checks pass.
