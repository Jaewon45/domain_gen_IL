#!/usr/bin/env python3
"""Evaluate a checkpoint using an exact reusable validation tensor cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import TensorDataset

try:
    from .data import datasets_from_manifest, load_imagenet100
    from .evaluate import evaluate_dataset, parse_lambda_grid, summarize
    from .manifests import load_manifest
    from .models import build_model, build_transform_and_spec
except ImportError:
    from data import datasets_from_manifest, load_imagenet100
    from evaluate import evaluate_dataset, parse_lambda_grid, summarize
    from manifests import load_manifest
    from models import build_model, build_transform_and_spec


CORRUPTION_TYPES = ("gaussian_noise", "defocus_blur", "snow", "contrast")
SEVERITIES = (1, 2, 3, 4, 5)


def cached_dataset(cache_dir: Path, name: str):
    payload = torch.load(cache_dir / name, map_location="cpu", weights_only=False)
    return TensorDataset(payload["images"], payload["labels"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--lambda_grid", default="0.0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    manifest_path = args.manifest or checkpoint["manifest_path"]
    manifest = load_manifest(manifest_path)
    transform, transform_spec = build_transform_and_spec(checkpoint["weight_version"])
    if transform_spec != manifest["transform"]:
        raise ValueError("Runtime transform differs from the run manifest")

    cache_dir = Path(args.cache_dir)
    with (cache_dir / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if int(metadata["global_seed"]) != int(manifest["global_seed"]):
        raise ValueError("Cache seed and checkpoint seed differ")
    if metadata["transform"] != transform_spec:
        raise ValueError("Cache transform differs from the checkpoint transform")
    if metadata["corruption_types"] != list(CORRUPTION_TYPES):
        raise ValueError("Cache corruption types do not match the focused protocol")

    validation = load_imagenet100(
        manifest["dataset"]["name"], manifest["dataset"]["revision"], split="validation"
    )
    source_train = load_imagenet100(
        manifest["dataset"]["name"], manifest["dataset"]["revision"], split="train"
    )
    source_validation_datasets = datasets_from_manifest(
        source_train, manifest, transform, validation=True
    )
    clean = cached_dataset(cache_dir, "clean.pt")
    condition_paths = {
        (corruption, severity): cache_dir / f"{corruption}__severity{severity}.pt"
        for corruption in CORRUPTION_TYPES
        for severity in SEVERITIES
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(checkpoint["model_mode"], weight_version=checkpoint["weight_version"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    lambda_grid = parse_lambda_grid(args.lambda_grid)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    for preference in lambda_grid:
        clean_metrics = evaluate_dataset(model, clean, device, args.batch_size, args.workers, preference)
        source_rows = []
        for domain, dataset in source_validation_datasets.items():
            source_rows.append({
                "domain": domain,
                **evaluate_dataset(model, dataset, device, args.batch_size, args.workers, preference),
            })
        condition_rows = []
        for (corruption, severity), path in condition_paths.items():
            dataset = cached_dataset(cache_dir, path.name)
            condition_rows.append({
                "corruption": corruption,
                "severity": severity,
                **evaluate_dataset(model, dataset, device, args.batch_size, args.workers, preference),
            })
        record = {
            "schema_version": 1,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "manifest_id": manifest["manifest_id"],
            "algorithm": checkpoint["algorithm"],
            "experiment": manifest["experiment"],
            "condition": manifest["condition"],
            "protocol": manifest["protocol"],
            "seed": manifest["global_seed"],
            "lambda": preference,
            "clean": clean_metrics,
            "conditions": condition_rows,
            "source_validation_domain_rows": source_rows,
            "environment_counts": manifest["active_domain_counts"],
            "per_class_counts": manifest["source_class_counts"],
            "selection_uses_target_validation": False,
            "evaluation_scope": {
                "kind": "pilot",
                "max_eval_images": int(metadata["max_eval_images"]),
                "validation_sampling": "class_stratified",
                "corruption_types": list(CORRUPTION_TYPES),
                "severities": list(SEVERITIES),
            },
        }
        record.update(summarize(condition_rows, source_rows, manifest))
        records.append(record)
        print(json.dumps({key: record[key] for key in ["algorithm", "lambda", "mean_domain_error", "worst_domain_error", "domain_cvar_error"]}, sort_keys=True))

    with (output_dir / "evaluation.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
