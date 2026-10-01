#!/usr/bin/env python3
"""Benchmark an approximate GPU-native E3b training input pipeline."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import resize, to_tensor

from ..algorithms import DomainAlgorithm
from ..data import load_imagenet100
from ..manifests import load_manifest
from ..models import build_model


MEAN = torch.tensor([0.485, 0.456, 0.406])
STD = torch.tensor([0.229, 0.224, 0.225])


def stage_domain(dataset, indices, images_per_domain):
    selected = list(indices[:images_per_domain])
    images = []
    labels = []
    for index in selected:
        row = dataset[int(index)]
        image = row["image"].convert("RGB")
        image = resize(image, [256, 256])
        images.append((to_tensor(image) * 255.0).to(torch.uint8))
        labels.append(int(row["label"]))
    return torch.stack(images), torch.tensor(labels, dtype=torch.long)


def gpu_corrupt_and_transform(raw, corruption, device, generator):
    images = raw.to(device=device, dtype=torch.float32) / 255.0
    batch_size = images.shape[0]
    severity = torch.randint(1, 6, (batch_size,), device=device, generator=generator)
    if corruption == "gaussian_noise":
        scale = severity.to(images.dtype).view(-1, 1, 1, 1) * 0.025
        images = images + torch.randn(images.shape, device=device, generator=generator) * scale
    elif corruption == "contrast":
        factor = (1.25 - severity.to(images.dtype) * 0.05).view(-1, 1, 1, 1)
        images = (images - images.mean(dim=(2, 3), keepdim=True)) * factor + 0.5
    elif corruption == "defocus_blur":
        images = F.avg_pool2d(images, kernel_size=5, stride=1, padding=2)
    elif corruption == "snow":
        density = severity.to(images.dtype).view(-1, 1, 1, 1) * 0.025
        mask = torch.rand(images.shape, device=device, generator=generator)
        images = images + (mask < density).to(images.dtype) * 0.8
    else:
        raise ValueError(f"Unsupported corruption domain: {corruption}")
    images = images.clamp(0.0, 1.0)
    images = F.interpolate(images, size=(232, 232), mode="bilinear", align_corners=False)
    images = images[:, :, 4:228, 4:228]
    mean = MEAN.to(device=device, dtype=images.dtype).view(1, 3, 1, 1)
    std = STD.to(device=device, dtype=images.dtype).view(1, 3, 1, 1)
    return (images - mean) / std


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--images-per-domain", type=int, default=1000)
    parser.add_argument("--algorithm", default="erm", choices=["erm", "irm", "vrex", "eqrm", "groupdro", "inftask", "iro"])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA device is required for the GPU-only benchmark")
    if args.batch_size % 4 != 0:
        raise ValueError("--batch-size must be divisible by four active E3b domains")

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device("cuda")
    manifest = load_manifest(args.manifest)
    dataset = load_imagenet100(
        manifest["dataset"]["name"], manifest["dataset"]["revision"], split="train"
    )
    domains = list(manifest["source_assignments"])
    if len(domains) != 4:
        raise ValueError(f"Expected four E3b domains, got {domains}")

    stage_start = time.perf_counter()
    staged = {}
    labels = {}
    for domain in domains:
        staged[domain], labels[domain] = stage_domain(
            dataset, manifest["source_assignments"][domain], args.images_per_domain
        )
    stage_seconds = time.perf_counter() - stage_start

    gpu_pools = {domain: images.to(device, non_blocking=True) for domain, images in staged.items()}
    gpu_labels = {domain: values.to(device, non_blocking=True) for domain, values in labels.items()}
    model = build_model("finetune_last_stage").to(device)
    algorithm = DomainAlgorithm(
        args.algorithm,
        model,
        learning_rate=3e-4,
        weight_decay=1e-4,
        groupdro_eta=0.1,
        num_lambda_samples=4,
        eqrm_alpha=0.75,
        seed=int(manifest["global_seed"]),
        device=device,
    )
    generator = torch.Generator(device=device).manual_seed(int(manifest["global_seed"]))
    per_domain = args.batch_size // len(domains)

    for _ in range(10):
        minibatches = []
        for domain in domains:
            indices = torch.randint(gpu_pools[domain].shape[0], (per_domain,), device=device, generator=generator)
            images = gpu_corrupt_and_transform(gpu_pools[domain][indices], domain, device, generator)
            minibatches.append((domain, images, gpu_labels[domain][indices]))
        algorithm.update(minibatches)
    torch.cuda.synchronize(device)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(args.steps):
        minibatches = []
        for domain in domains:
            indices = torch.randint(gpu_pools[domain].shape[0], (per_domain,), device=device, generator=generator)
            images = gpu_corrupt_and_transform(gpu_pools[domain][indices], domain, device, generator)
            minibatches.append((domain, images, gpu_labels[domain][indices]))
        algorithm.update(minibatches)
    end_event.record()
    torch.cuda.synchronize(device)
    gpu_seconds = start_event.elapsed_time(end_event) / 1000.0

    result = {
        "benchmark": "approximate_gpu_native_e3b",
        "manifest": str(Path(args.manifest).resolve()),
        "algorithm": args.algorithm,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "images_per_domain": args.images_per_domain,
        "domains": domains,
        "device": torch.cuda.get_device_name(device),
        "cpu_staging_seconds": stage_seconds,
        "gpu_training_seconds": gpu_seconds,
        "gpu_seconds_per_step": gpu_seconds / args.steps,
        "gpu_steps_per_second": args.steps / gpu_seconds,
        "note": "GPU corruption/transforms are approximate and are not paper-protocol results.",
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
