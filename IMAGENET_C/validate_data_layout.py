#!/usr/bin/env python3
"""Validate clean ImageNet and ImageNet-C layout/mapping compatibility."""

import argparse
import os
from pathlib import Path
from typing import Dict, List

from torchvision.datasets import ImageFolder, ImageNet  # type: ignore

from datasets import CORRUPTIONS, SEVERITIES


EXPECTED_IMAGES = 50000
EXPECTED_CLASSES = 1000


def list_subdirs(path: Path) -> List[str]:
    if not path.is_dir():
        raise FileNotFoundError(f"Directory not found: {path}")
    return sorted([entry.name for entry in path.iterdir() if entry.is_dir()])


def validate_clean_root(imagenet_root: Path) -> Dict[str, object]:
    clean = ImageNet(root=str(imagenet_root), split="val")
    val_root = imagenet_root / "val"
    val_dirs = list_subdirs(val_root)
    val_images = list(val_root.glob("*/*.JPEG"))

    if len(clean) != EXPECTED_IMAGES:
        raise ValueError(f"Clean ImageNet image count mismatch: {len(clean)} != {EXPECTED_IMAGES}")
    if len(clean.wnids) != EXPECTED_CLASSES:
        raise ValueError(f"Clean ImageNet wnid list length mismatch: {len(clean.wnids)} != {EXPECTED_CLASSES}")
    if len(clean.wnid_to_idx) != EXPECTED_CLASSES:
        raise ValueError(f"Clean ImageNet wnid_to_idx length mismatch: {len(clean.wnid_to_idx)} != {EXPECTED_CLASSES}")
    if len(val_dirs) != EXPECTED_CLASSES:
        raise ValueError(f"Clean val directory count mismatch: {len(val_dirs)} != {EXPECTED_CLASSES}")
    if len(val_images) != EXPECTED_IMAGES:
        raise ValueError(f"Clean val image count mismatch: {len(val_images)} != {EXPECTED_IMAGES}")

    return {
        "clean": clean,
        "wnid_set": set(clean.wnids),
        "val_dir_set": set(val_dirs),
        "val_image_count": len(val_images),
    }


def validate_corruption_structure(imagenet_c_root: Path, clean_wnid_set: set) -> None:
    missing_corruptions = [corr for corr in CORRUPTIONS if not (imagenet_c_root / corr).is_dir()]
    if missing_corruptions:
        raise FileNotFoundError(f"Missing corruption directories: {missing_corruptions}")

    for corruption in CORRUPTIONS:
        corruption_root = imagenet_c_root / corruption
        severities = list_subdirs(corruption_root)
        expected = {str(level) for level in SEVERITIES}
        if set(severities) != expected:
            raise ValueError(
                f"Severity mismatch for {corruption}: found {sorted(severities)} expected {sorted(expected)}"
            )

        for severity in SEVERITIES:
            condition_root = corruption_root / str(severity)
            wnids = set(list_subdirs(condition_root))
            if wnids != clean_wnid_set:
                missing = sorted(clean_wnid_set - wnids)
                extra = sorted(wnids - clean_wnid_set)
                detail = []
                if missing:
                    detail.append(f"missing first={missing[0]}")
                if extra:
                    detail.append(f"extra first={extra[0]}")
                detail_txt = "; ".join(detail) if detail else "mismatch"
                raise ValueError(f"WNID directory set mismatch at {condition_root}: {detail_txt}")


def validate_mapping(clean, imagenet_c_root: Path) -> Dict[str, int]:
    sample_condition = imagenet_c_root / "gaussian_noise" / "1"
    if not sample_condition.is_dir():
        raise FileNotFoundError(f"Sample condition missing: {sample_condition}")

    corrupted = ImageFolder(root=str(sample_condition))
    if len(corrupted) != EXPECTED_IMAGES:
        raise ValueError(f"Sample condition image count mismatch: {len(corrupted)} != {EXPECTED_IMAGES}")
    if len(corrupted.classes) != EXPECTED_CLASSES:
        raise ValueError(f"Sample condition classes mismatch: {len(corrupted.classes)} != {EXPECTED_CLASSES}")

    if clean.wnid_to_idx != corrupted.class_to_idx:
        raise ValueError("WNID index mapping mismatch: clean.wnid_to_idx != corrupted.class_to_idx")
    if clean.wnids != corrupted.classes:
        raise ValueError("WNID order mismatch: clean.wnids != corrupted.classes")

    return {
        "sample_images": len(corrupted),
        "sample_classes": len(corrupted.classes),
    }


def full_verify_all_conditions(imagenet_c_root: Path) -> None:
    for corruption in CORRUPTIONS:
        for severity in SEVERITIES:
            condition_root = imagenet_c_root / corruption / str(severity)
            dataset = ImageFolder(root=str(condition_root))
            if len(dataset) != EXPECTED_IMAGES:
                raise ValueError(
                    f"Condition count mismatch at {condition_root}: {len(dataset)} != {EXPECTED_IMAGES}"
                )
            if len(dataset.classes) != EXPECTED_CLASSES:
                raise ValueError(
                    f"Condition class count mismatch at {condition_root}: {len(dataset.classes)} != {EXPECTED_CLASSES}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ImageNet + ImageNet-C layout compatibility")
    parser.add_argument("--imagenet_root", required=True)
    parser.add_argument("--imagenet_c_root", required=True)
    parser.add_argument("--full_verify", action="store_true")
    args = parser.parse_args()

    imagenet_root = Path(args.imagenet_root).resolve()
    imagenet_c_root = Path(args.imagenet_c_root).resolve()

    clean_result = validate_clean_root(imagenet_root)
    clean = clean_result["clean"]
    clean_wnid_set = clean_result["wnid_set"]

    validate_corruption_structure(imagenet_c_root, clean_wnid_set)
    mapping = validate_mapping(clean, imagenet_c_root)

    if args.full_verify:
        full_verify_all_conditions(imagenet_c_root)

    print(f"Clean images: {len(clean)}")
    print(f"Clean classes: {len(clean.wnids)}")
    print(f"ImageNet-C corruptions: {len(CORRUPTIONS)}")
    print("Severity levels per corruption: 5")
    print("WNID mapping match: PASS")
    print(f"Sample condition images: {mapping['sample_images']}")
    print(f"Sample condition classes: {mapping['sample_classes']}")
    if args.full_verify:
        print("Full condition verify: PASS")
    print("Overall validation: PASS")


if __name__ == "__main__":
    main()
