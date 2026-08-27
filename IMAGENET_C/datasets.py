#!/usr/bin/env python3
"""Shared dataset helpers for the ImageNet-C extension."""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


CORRUPTIONS = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
]
SEVERITIES = [1, 2, 3, 4, 5]


def load_folds(folds_path: str) -> Dict[str, List[str]]:
    with open(folds_path, "r", encoding="utf-8") as handle:
        folds = json.load(handle)
    return {str(key): [str(value) for value in values] for key, values in folds.items()}


def ensure_output_subdirs(output_dir: Path, subdirs: Sequence[str], allow_existing_files: bool = False) -> None:
    if output_dir.exists() and not allow_existing_files:
        for _, _, files in os.walk(output_dir):
            if files:
                raise FileExistsError(
                    f"Output directory already contains files: {output_dir}. "
                    "Use a fresh results/imagenet_c_* root instead of overwriting artifacts."
                )
    for subdir in subdirs:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)


def write_manifest(output_dir: Path, manifest_name: str, argv: Sequence[str], header: str) -> None:
    manifest_path = output_dir / "manifests" / manifest_name
    with manifest_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# {header}\n")
        handle.write(" ".join(argv) + "\n")


def split_visible_domains(folds: Dict[str, List[str]], fold_name: str) -> Tuple[List[str], List[str]]:
    if fold_name not in folds:
        raise KeyError(f"Unknown fold: {fold_name}. Available folds: {sorted(folds.keys())}")
    held_out = list(folds[fold_name])
    visible = [corruption for corruption in CORRUPTIONS if corruption not in held_out]
    return visible, held_out


def smoke_feature_splits(
    folds_path: str,
    fold_name: str,
    seed: int,
    feature_dim: int = 64,
    num_classes: int = 20,
    train_samples_per_class: int = 4,
    val_samples_per_class: int = 2,
    test_samples_per_class: int = 3,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    folds = load_folds(folds_path)
    visible_domains, held_out_domains = split_visible_domains(folds, fold_name)

    class_prototypes = rng.normal(loc=0.0, scale=1.0, size=(num_classes, feature_dim)).astype(np.float32)
    corruption_shifts = {
        corruption: rng.normal(loc=0.0, scale=0.3, size=(feature_dim,)).astype(np.float32)
        for corruption in CORRUPTIONS
    }
    severity_shifts = {
        severity: (severity - 3.0) * rng.normal(loc=0.0, scale=0.05, size=(feature_dim,)).astype(np.float32)
        for severity in SEVERITIES
    }

    def make_condition(corruption: str, severity: int, samples_per_class: int) -> Tuple[torch.Tensor, torch.Tensor]:
        features = []
        labels = []
        for class_id in range(num_classes):
            base = class_prototypes[class_id] + corruption_shifts[corruption] + severity_shifts[severity]
            noise = rng.normal(loc=0.0, scale=0.35 + 0.05 * severity, size=(samples_per_class, feature_dim)).astype(np.float32)
            class_features = base[None, :] + noise
            features.append(class_features)
            labels.append(np.full((samples_per_class,), class_id, dtype=np.int64))
        x = torch.tensor(np.concatenate(features, axis=0), dtype=torch.float32)
        y = torch.tensor(np.concatenate(labels, axis=0), dtype=torch.long)
        return x, y

    train_domains: Dict[str, Dict[str, torch.Tensor]] = {}
    val_conditions: List[Dict[str, object]] = []
    eval_conditions: List[Dict[str, object]] = []

    for corruption in visible_domains:
        train_x_parts = []
        train_y_parts = []
        for severity in [1, 2, 3]:
            x, y = make_condition(corruption, severity, train_samples_per_class)
            train_x_parts.append(x)
            train_y_parts.append(y)
        train_domains[corruption] = {
            "x": torch.cat(train_x_parts, dim=0),
            "y": torch.cat(train_y_parts, dim=0),
        }

        val_x, val_y = make_condition(corruption, 4, val_samples_per_class)
        val_conditions.append(
            {
                "corruption": corruption,
                "severity": 4,
                "x": val_x,
                "y": val_y,
                "split": "visible_val",
                "domain_seen": True,
            }
        )

    for corruption in CORRUPTIONS:
        for severity in SEVERITIES:
            samples = test_samples_per_class
            x, y = make_condition(corruption, severity, samples)
            eval_conditions.append(
                {
                    "corruption": corruption,
                    "severity": severity,
                    "x": x,
                    "y": y,
                    "split": "held_out" if corruption in held_out_domains else "all",
                    "domain_seen": corruption in visible_domains,
                }
            )

    clean_features = []
    clean_labels = []
    for class_id in range(num_classes):
        base = class_prototypes[class_id]
        noise = rng.normal(loc=0.0, scale=0.25, size=(test_samples_per_class, feature_dim)).astype(np.float32)
        clean_features.append(base[None, :] + noise)
        clean_labels.append(np.full((test_samples_per_class,), class_id, dtype=np.int64))
    clean_condition = {
        "corruption": "clean",
        "severity": 0,
        "x": torch.tensor(np.concatenate(clean_features, axis=0), dtype=torch.float32),
        "y": torch.tensor(np.concatenate(clean_labels, axis=0), dtype=torch.long),
        "split": "clean",
        "domain_seen": True,
    }

    return {
        "fold": fold_name,
        "visible_domains": visible_domains,
        "held_out_domains": held_out_domains,
        "train_domains": train_domains,
        "val_conditions": val_conditions,
        "eval_conditions": eval_conditions,
        "clean_condition": clean_condition,
        "feature_dim": feature_dim,
        "num_classes": num_classes,
    }


def maybe_import_torchvision():
    try:
        from torchvision import datasets, models  # type: ignore
        return datasets, models
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "torchvision is required for real ImageNet/ImageNet-C evaluation. "
            "Use the smoke mode if real assets are unavailable."
        ) from exc


def default_imagenet_transforms(models_module):
    weights = models_module.ResNet50_Weights.DEFAULT
    return weights.transforms(), weights


def _has_any_subdirectory(root: str) -> bool:
    try:
        with os.scandir(root) as entries:
            return any(entry.is_dir() for entry in entries)
    except FileNotFoundError:
        return False


def validate_imagefolder_root(root: str, dataset_label: str) -> None:
    if not os.path.isdir(root):
        raise FileNotFoundError(f"{dataset_label} directory not found: {root}")
    if not _has_any_subdirectory(root):
        raise FileNotFoundError(
            f"{dataset_label} is missing class subfolders under: {root}. "
            "Expected ImageFolder layout like <root>/n01440764/*.JPEG."
        )


def imagefolder_dataset(root: str, transform, max_images: Optional[int] = None):
    datasets_module, _ = maybe_import_torchvision()
    validate_imagefolder_root(root, "ImageFolder dataset")
    dataset = datasets_module.ImageFolder(root=root, transform=transform)
    if max_images is None or max_images >= len(dataset):
        return dataset
    indices = list(range(max_images))
    return torch.utils.data.Subset(dataset, indices)


def imagenet_dataset(root: str, split: str, transform, max_images: Optional[int] = None):
    datasets_module, _ = maybe_import_torchvision()
    dataset = datasets_module.ImageNet(root=root, split=split, transform=transform)
    if max_images is None or max_images >= len(dataset):
        return dataset
    indices = list(range(max_images))
    return torch.utils.data.Subset(dataset, indices)


def clean_imagenet_dataset(
    imagenet_root: str,
    transform,
    max_images: Optional[int],
    backend: str = "imagenet",
):
    backend = backend.lower()
    if backend not in {"imagenet", "imagefolder", "auto"}:
        raise ValueError("backend must be one of: imagenet, imagefolder, auto")

    if backend == "imagefolder":
        val_root = os.path.join(imagenet_root, "val")
        validate_imagefolder_root(val_root, "Clean ImageNet validation (ImageFolder backend)")
        return imagefolder_dataset(val_root, transform, max_images)

    if backend == "imagenet":
        try:
            return imagenet_dataset(imagenet_root, split="val", transform=transform, max_images=max_images)
        except RuntimeError as exc:
            if "ILSVRC2012_devkit_t12.tar.gz" in str(exc):
                raise RuntimeError(
                    "torchvision.datasets.ImageNet requires ILSVRC2012_devkit_t12.tar.gz under the ImageNet root. "
                    "If you only have class-folder val data, rerun with --clean_loader_backend imagefolder (or auto)."
                ) from exc
            raise

    # auto: try torchvision.datasets.ImageNet first, then fall back to ImageFolder on val directory.
    try:
        return imagenet_dataset(imagenet_root, split="val", transform=transform, max_images=max_images)
    except Exception as imagenet_exc:
        val_root = os.path.join(imagenet_root, "val")
        try:
            validate_imagefolder_root(val_root, "Clean ImageNet validation (auto backend fallback)")
            return imagefolder_dataset(val_root, transform, max_images)
        except Exception as imagefolder_exc:
            raise RuntimeError(
                "Unable to load clean ImageNet validation split using either backend. "
                f"ImageNet backend error: {imagenet_exc}. "
                f"ImageFolder fallback error: {imagefolder_exc}"
            ) from imagefolder_exc


def infer_num_classes(dataset) -> int:
    if hasattr(dataset, "classes"):
        return len(dataset.classes)
    if hasattr(dataset, "dataset") and hasattr(dataset.dataset, "classes"):
        return len(dataset.dataset.classes)
    raise AttributeError("Could not infer class count from dataset")


def build_real_feature_splits(
    folds_path: str,
    fold_name: str,
    imagenet_train_corrupted_root: str,
    imagenet_root: str,
    imagenet_c_root: str,
    batch_size: int,
    max_images_per_condition: Optional[int],
    device: torch.device,
    clean_loader_backend: str = "imagenet",
) -> Dict[str, object]:
    from features import extract_dataset_features, load_frozen_resnet50

    folds = load_folds(folds_path)
    visible_domains, held_out_domains = split_visible_domains(folds, fold_name)
    feature_extractor, transform, feature_dim = load_frozen_resnet50(device)

    clean_dataset = clean_imagenet_dataset(
        imagenet_root=imagenet_root,
        transform=transform,
        max_images=max_images_per_condition,
        backend=clean_loader_backend,
    )
    num_classes = infer_num_classes(clean_dataset)

    train_domains: Dict[str, Dict[str, torch.Tensor]] = {}
    val_conditions: List[Dict[str, object]] = []
    eval_conditions: List[Dict[str, object]] = []

    for corruption in visible_domains:
        train_x_parts = []
        train_y_parts = []
        for severity in [1, 2, 3]:
            train_root = os.path.join(imagenet_train_corrupted_root, corruption, str(severity))
            if not os.path.isdir(train_root):
                raise FileNotFoundError(f"Missing corrupted train directory: {train_root}")
            dataset = imagefolder_dataset(train_root, transform, max_images_per_condition)
            x, y = extract_dataset_features(dataset, feature_extractor, device, batch_size)
            train_x_parts.append(x)
            train_y_parts.append(y)

        train_domains[corruption] = {
            "x": torch.cat(train_x_parts, dim=0),
            "y": torch.cat(train_y_parts, dim=0),
        }

        val_root = os.path.join(imagenet_train_corrupted_root, corruption, "4")
        if not os.path.isdir(val_root):
            raise FileNotFoundError(f"Missing corrupted validation directory: {val_root}")
        val_dataset = imagefolder_dataset(val_root, transform, max_images_per_condition)
        val_x, val_y = extract_dataset_features(val_dataset, feature_extractor, device, batch_size)
        val_conditions.append(
            {
                "corruption": corruption,
                "severity": 4,
                "x": val_x,
                "y": val_y,
                "split": "visible_val",
                "domain_seen": True,
            }
        )

    for corruption in CORRUPTIONS:
        for severity in SEVERITIES:
            eval_root = os.path.join(imagenet_c_root, corruption, str(severity))
            if not os.path.isdir(eval_root):
                raise FileNotFoundError(f"Missing ImageNet-C evaluation directory: {eval_root}")
            dataset = imagefolder_dataset(eval_root, transform, max_images_per_condition)
            x, y = extract_dataset_features(dataset, feature_extractor, device, batch_size)
            eval_conditions.append(
                {
                    "corruption": corruption,
                    "severity": severity,
                    "x": x,
                    "y": y,
                    "split": "held_out" if corruption in held_out_domains else "all",
                    "domain_seen": corruption in visible_domains,
                }
            )

    clean_x, clean_y = extract_dataset_features(clean_dataset, feature_extractor, device, batch_size)
    clean_condition = {
        "corruption": "clean",
        "severity": 0,
        "x": clean_x,
        "y": clean_y,
        "split": "clean",
        "domain_seen": True,
    }

    return {
        "fold": fold_name,
        "visible_domains": visible_domains,
        "held_out_domains": held_out_domains,
        "train_domains": train_domains,
        "val_conditions": val_conditions,
        "eval_conditions": eval_conditions,
        "clean_condition": clean_condition,
        "feature_dim": feature_dim,
        "num_classes": num_classes,
    }
