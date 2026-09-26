"""Hugging Face ImageNet-100 loading and corruption-domain datasets."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

try:
    from .corruptions import CORRUPTION_TYPES, SEVERITIES, corrupt_image, deterministic_choice
except ImportError:  # Support ``python IMAGENET100C/train.py``.
    from corruptions import CORRUPTION_TYPES, SEVERITIES, corrupt_image, deterministic_choice


def load_imagenet100(dataset_name: str, revision: str, split: Optional[str] = None):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Hugging Face datasets is required. Install `datasets` before loading ImageNet-100."
        ) from exc
    return load_dataset(dataset_name, revision=revision, split=split)


def validate_label_space(labels: Sequence[int], num_classes: int = 100) -> None:
    if not labels:
        raise ValueError("Dataset has no labels")
    if any(not isinstance(value, int) for value in labels):
        raise TypeError("ImageNet-100 labels must be Python integers")
    minimum, maximum = min(labels), max(labels)
    if minimum < 0 or maximum >= num_classes:
        raise ValueError(f"Expected labels in [0,{num_classes - 1}], got [{minimum},{maximum}]")


class ImageNet100CDataset(Dataset):
    """A view over selected HF rows with deterministic corruption sampling."""

    def __init__(
        self,
        base_dataset,
        indices: Sequence[int],
        transform,
        *,
        split: str,
        global_seed: int,
        protocol: str,
        domain: Optional[str] = None,
        corruption_name: Optional[str] = None,
        severity: Optional[int] = None,
        clean: bool = False,
        corruption_backend=None,
    ):
        self.base_dataset = base_dataset
        self.indices = [int(value) for value in indices]
        self.transform = transform
        self.split = str(split)
        self.global_seed = int(global_seed)
        self.protocol = str(protocol)
        self.domain = None if domain is None else str(domain)
        self.corruption_name = corruption_name
        self.severity = severity
        self.clean = bool(clean)
        self.corruption_backend = corruption_backend
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.indices)

    def _condition(self, image_index: int):
        if self.clean:
            return None, 0
        if self.corruption_name is not None and self.severity is not None:
            return self.corruption_name, int(self.severity)
        if self.protocol == "mechanism":
            if self.domain not in CORRUPTION_TYPES:
                raise ValueError(f"Mechanism domain must be a corruption type, got {self.domain}")
            severity = deterministic_choice(
                SEVERITIES, self.global_seed, self.split, image_index, self.domain, self.epoch
            )
            return self.domain, int(severity)
        if self.protocol == "severity":
            severity = int(self.domain)
            corruption = deterministic_choice(
                CORRUPTION_TYPES, self.global_seed, self.split, image_index, "corruption", severity, self.epoch
            )
            return str(corruption), severity
        raise ValueError(f"Unknown protocol: {self.protocol}")

    def __getitem__(self, position: int):
        image_index = self.indices[position]
        row = self.base_dataset[image_index]
        image = row["image"].convert("RGB")
        label = int(row["label"])
        if not 0 <= label < 100:
            raise ValueError(f"Label outside [0,99] at index {image_index}: {label}")
        corruption_name, severity = self._condition(image_index)
        if corruption_name is not None:
            image = corrupt_image(
                image,
                corruption_name,
                severity,
                global_seed=self.global_seed,
                split=self.split,
                image_index=image_index,
                epoch=self.epoch,
                backend=self.corruption_backend,
            )
        tensor = self.transform(image)
        return tensor, label


def datasets_from_manifest(base_train, manifest: Mapping[str, object], transform, *, validation: bool = False):
    key = "source_validation_assignments" if validation else "source_assignments"
    split = "source_validation" if validation else "source_train"
    assignments = manifest[key]
    return {
        domain: ImageNet100CDataset(
            base_train,
            indices,
            transform,
            split=split,
            global_seed=int(manifest["global_seed"]),
            protocol=str(manifest["protocol"]),
            domain=domain,
        )
        for domain, indices in assignments.items()
        if len(indices) > 0
    }


def proportional_batch_sizes(counts: Mapping[str, int], total_batch_size: int) -> Dict[str, int]:
    active = {str(name): int(count) for name, count in counts.items() if int(count) > 0}
    if not active:
        raise ValueError("At least one active domain is required")
    if total_batch_size < len(active):
        raise ValueError("total_batch_size must be at least the number of active domains")
    total = sum(active.values())
    raw = {name: total_batch_size * count / total for name, count in active.items()}
    sizes = {name: max(1, int(value)) for name, value in raw.items()}
    while sum(sizes.values()) < total_batch_size:
        name = max(active, key=lambda item: (raw[item] - sizes[item], active[item], item))
        sizes[name] += 1
    while sum(sizes.values()) > total_batch_size:
        candidates = [name for name in active if sizes[name] > 1]
        if not candidates:
            break
        name = max(candidates, key=lambda item: (sizes[item] - raw[item], sizes[item], item))
        sizes[name] -= 1
    return sizes


class InfiniteDomainBatches:
    def __init__(self, datasets: Mapping[str, Dataset], batch_sizes: Mapping[str, int], seed: int, workers: int):
        self.datasets = dict(datasets)
        self.loaders = {}
        self.iterators = {}
        for offset, (domain, dataset) in enumerate(self.datasets.items()):
            generator = torch.Generator().manual_seed(int(seed) + offset)
            loader = DataLoader(
                dataset,
                batch_size=int(batch_sizes[domain]),
                shuffle=True,
                drop_last=False,
                num_workers=int(workers),
                generator=generator,
            )
            self.loaders[domain] = loader
            self.iterators[domain] = iter(loader)

    def next(self, epoch: int):
        batches = []
        for domain, loader in self.loaders.items():
            dataset = self.datasets[domain]
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch)
            try:
                batch = next(self.iterators[domain])
            except StopIteration:
                self.iterators[domain] = iter(loader)
                batch = next(self.iterators[domain])
            batches.append((domain, batch[0], batch[1]))
        return batches


def _class_stratified_prefix_indices(base_dataset, max_images: int) -> Sequence[int]:
    """Select a deterministic, approximately class-balanced evaluation prefix."""
    limit = min(len(base_dataset), int(max_images))
    if limit <= 0:
        raise ValueError("max_images must be positive")
    labels = [int(value) for value in base_dataset["label"]]
    classes = sorted(set(labels))
    quotient, remainder = divmod(limit, len(classes))
    quotas = {label: quotient + (position < remainder) for position, label in enumerate(classes)}
    selected = []
    used = {label: 0 for label in classes}
    for index, label in enumerate(labels):
        if used[label] < quotas[label]:
            selected.append(index)
            used[label] += 1
        if len(selected) == limit:
            break
    if len(selected) != limit:
        raise ValueError(f"Could select only {len(selected)} of {limit} stratified validation images")
    return selected


def final_evaluation_datasets(
    base_validation,
    transform,
    seed: int,
    max_images: Optional[int] = None,
    corruption_types: Optional[Sequence[str]] = None,
):
    indices = (
        list(range(len(base_validation)))
        if max_images is None
        else list(_class_stratified_prefix_indices(base_validation, int(max_images)))
    )
    selected_corruptions = list(CORRUPTION_TYPES if corruption_types is None else corruption_types)
    unknown = sorted(set(selected_corruptions) - set(CORRUPTION_TYPES))
    if unknown:
        raise ValueError(f"Unknown corruption types: {unknown}")
    if not selected_corruptions:
        raise ValueError("At least one corruption type is required")
    clean = ImageNet100CDataset(
        base_validation, indices, transform, split="validation", global_seed=seed,
        protocol="mechanism", clean=True,
    )
    conditions = {}
    for corruption_name in selected_corruptions:
        for severity in SEVERITIES:
            conditions[(corruption_name, severity)] = ImageNet100CDataset(
                base_validation,
                indices,
                transform,
                split="validation",
                global_seed=seed,
                protocol="mechanism",
                corruption_name=corruption_name,
                severity=severity,
            )
    return clean, conditions

