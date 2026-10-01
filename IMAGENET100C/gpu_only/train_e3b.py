#!/usr/bin/env python3
"""Run an isolated approximate GPU-native E3b training experiment."""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from ..algorithms import DomainAlgorithm
from ..data import load_imagenet100
from ..manifests import load_manifest
from ..models import build_model
from .benchmark_e3b import gpu_corrupt_and_transform, stage_domain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--images-per-domain", type=int, default=0, help="0 stages every assigned source image")
    parser.add_argument("--algorithm", default="erm", choices=["erm", "irm", "vrex", "eqrm", "groupdro", "inftask", "iro"])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA device is required for the GPU-native trainer")
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
        limit = len(manifest["source_assignments"][domain]) if args.images_per_domain == 0 else args.images_per_domain
        staged[domain], labels[domain] = stage_domain(
            dataset, manifest["source_assignments"][domain], limit
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    history_path = output_dir / "history.jsonl"
    start = time.perf_counter()
    with history_path.open("w", encoding="utf-8") as history:
        for step in range(1, args.steps + 1):
            minibatches = []
            for domain in domains:
                indices = torch.randint(
                    gpu_pools[domain].shape[0], (per_domain,), device=device, generator=generator
                )
                images = gpu_corrupt_and_transform(
                    gpu_pools[domain][indices], domain, device, generator
                )
                minibatches.append((domain, images, gpu_labels[domain][indices]))
            update = algorithm.update(minibatches)
            update["step"] = step
            history.write(json.dumps(update, sort_keys=True) + "\n")
            history.flush()
            if step % 50 == 0:
                print(json.dumps({"step": step, "objective": update["objective"]}, sort_keys=True), flush=True)

    torch.cuda.synchronize(device)
    checkpoint = {
        "schema_version": 1,
        "benchmark": "approximate_gpu_native_e3b",
        "manifest_path": str(Path(args.manifest).resolve()),
        "algorithm": args.algorithm,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "images_per_domain": args.images_per_domain,
        "domains": domains,
        "device": torch.cuda.get_device_name(device),
        "cpu_staging_seconds": stage_seconds,
        "gpu_training_seconds": time.perf_counter() - start,
        "model_state": copy.deepcopy(model.state_dict()),
        "algorithm_state": algorithm.checkpoint_state(),
        "note": "Approximate GPU-native corruption/transforms; not paper-protocol output.",
    }
    torch.save(checkpoint, output_dir / "final.pt")
    print(json.dumps({key: checkpoint[key] for key in ["algorithm", "steps", "cpu_staging_seconds", "gpu_training_seconds", "device"]}, indent=2))


if __name__ == "__main__":
    main()
