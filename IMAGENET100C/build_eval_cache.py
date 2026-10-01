#!/usr/bin/env python3
"""Build exact, reusable ImageNet-100-C validation tensor caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

try:
    from .data import final_evaluation_datasets, load_imagenet100
    from .manifests import load_manifest
    from .models import build_transform_and_spec
except ImportError:
    from data import final_evaluation_datasets, load_imagenet100
    from manifests import load_manifest
    from models import build_transform_and_spec


CORRUPTION_TYPES = ("gaussian_noise", "defocus_blur", "snow", "contrast")


def materialize(dataset):
    images = []
    labels = []
    for index in range(len(dataset)):
        image, label = dataset[index]
        images.append(image.contiguous())
        labels.append(int(label))
    return torch.stack(images), torch.tensor(labels, dtype=torch.long)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--max_eval_images", type=int, default=1000)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    manifest = load_manifest(checkpoint["manifest_path"])
    transform, transform_spec = build_transform_and_spec(checkpoint["weight_version"])
    if transform_spec != manifest["transform"]:
        raise ValueError("Runtime transform differs from the run manifest")
    validation = load_imagenet100(
        manifest["dataset"]["name"], manifest["dataset"]["revision"], split="validation"
    )
    clean, conditions = final_evaluation_datasets(
        validation,
        transform,
        int(manifest["global_seed"]),
        args.max_eval_images,
        corruption_types=CORRUPTION_TYPES,
    )

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "schema_version": 1,
        "manifest_id": manifest["manifest_id"],
        "global_seed": manifest["global_seed"],
        "transform": transform_spec,
        "max_eval_images": args.max_eval_images,
        "corruption_types": list(CORRUPTION_TYPES),
        "severities": [1, 2, 3, 4, 5],
        "ordering": "ImageNet100CDataset position order",
    }

    print("Caching clean validation data")
    images, labels = materialize(clean)
    torch.save({"images": images, "labels": labels}, cache_dir / "clean.pt")
    for (corruption, severity), dataset in conditions.items():
        print(f"Caching {corruption} severity {severity}")
        images, labels = materialize(dataset)
        torch.save(
            {"images": images, "labels": labels},
            cache_dir / f"{corruption}__severity{severity}.pt",
        )
    with (cache_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, sort_keys=True, indent=2)
    print(f"Saved exact evaluation cache to {cache_dir}")


if __name__ == "__main__":
    main()
