"""Run manifests and deterministic class-stratified source assignments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

try:
    from .corruptions import CORRUPTION_TYPES
except ImportError:
    from corruptions import CORRUPTION_TYPES


SCHEMA_VERSION = 1


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def manifest_id(manifest: Mapping[str, object]) -> str:
    payload = dict(manifest)
    payload.pop("manifest_id", None)
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:16]


def save_manifest(manifest: Mapping[str, object], path: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(manifest)
    payload["manifest_id"] = manifest_id(payload)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_manifest(path: str) -> Dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    expected = manifest_id(manifest)
    if manifest.get("manifest_id") != expected:
        raise ValueError(f"Manifest checksum mismatch for {path}")
    return manifest


def _balanced_quotas(total: int, classes: Sequence[int]) -> Dict[int, int]:
    if total < 0:
        raise ValueError("Counts must be non-negative")
    ordered = sorted(int(label) for label in classes)
    quotient, remainder = divmod(total, len(ordered))
    return {label: quotient + (position < remainder) for position, label in enumerate(ordered)}


def class_stratified_assignments(
    labels: Sequence[int],
    domain_counts: Mapping[str, int],
    *,
    seed: int,
    validation_per_domain: int = 0,
) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
    """Assign non-overlapping indices while balancing every domain by class."""
    labels_array = np.asarray(labels, dtype=np.int64)
    classes = sorted(int(value) for value in np.unique(labels_array))
    if not classes:
        raise ValueError("No labels supplied")
    rng = np.random.default_rng(seed)
    pools = {}
    cursors = {label: 0 for label in classes}
    for label in classes:
        indices = np.flatnonzero(labels_array == label).astype(np.int64)
        rng.shuffle(indices)
        pools[label] = indices.tolist()

    def allocate(counts: Mapping[str, int]) -> Dict[str, List[int]]:
        result: Dict[str, List[int]] = {}
        for domain, count_value in counts.items():
            count = int(count_value)
            quotas = _balanced_quotas(count, classes)
            chosen: List[int] = []
            for label in classes:
                start = cursors[label]
                stop = start + quotas[label]
                if stop > len(pools[label]):
                    raise ValueError(
                        f"Insufficient class {label} images for domain {domain}: "
                        f"need {stop}, have {len(pools[label])}"
                    )
                chosen.extend(pools[label][start:stop])
                cursors[label] = stop
            rng.shuffle(chosen)
            result[str(domain)] = chosen
        return result

    train = allocate({str(k): int(v) for k, v in domain_counts.items() if int(v) > 0})
    validation_counts = {
        str(domain): int(validation_per_domain)
        for domain, count in domain_counts.items()
        if int(count) > 0 and validation_per_domain > 0
    }
    validation = allocate(validation_counts)
    return train, validation


def class_counts(indices: Sequence[int], labels: Sequence[int]) -> Dict[str, int]:
    values = np.asarray(labels, dtype=np.int64)[list(indices)]
    unique, counts = np.unique(values, return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(unique, counts)}


def build_manifest(
    *,
    dataset_name: str,
    dataset_revision: str,
    class_names: Sequence[str],
    labels: Sequence[int],
    protocol: str,
    experiment: str,
    condition: str,
    domain_counts: Mapping[str, int],
    seed: int,
    transform_spec: Mapping[str, object],
    validation_per_domain: int,
    weight_version: str,
) -> Dict[str, object]:
    assignments, source_validation = class_stratified_assignments(
        labels, domain_counts, seed=seed, validation_per_domain=validation_per_domain
    )
    active_counts = {name: len(indices) for name, indices in assignments.items()}
    manifest: Dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "dataset": {"name": dataset_name, "revision": dataset_revision},
        "splits": {"source": "train", "final_evaluation": "validation"},
        "protocol": protocol,
        "experiment": experiment,
        "condition": condition,
        "global_seed": int(seed),
        "weight_version": weight_version,
        "transform": dict(transform_spec),
        "class_mapping": {str(index): name for index, name in enumerate(class_names)},
        "requested_domain_counts": {str(k): int(v) for k, v in domain_counts.items()},
        "active_domain_counts": active_counts,
        "source_assignments": assignments,
        "source_validation_assignments": source_validation,
        "source_class_counts": {
            domain: class_counts(indices, labels) for domain, indices in assignments.items()
        },
        "source_validation_class_counts": {
            domain: class_counts(indices, labels) for domain, indices in source_validation.items()
        },
        "corruptions": [],
        "severity_values": [1, 2, 3, 4, 5],
        "severity_sampling_key": [
            "global_seed", "split", "image_index", "corruption_name", "epoch"
        ],
        "corruption_rng_seed_key": [
            "global_seed", "split", "image_index", "corruption_name", "severity", "epoch"
        ],
        "selection_data": "source-only held-out subset of ImageNet-100 train",
        "target_validation_used_for_selection": False,
    }
    if protocol == "mechanism":
        manifest["corruptions"] = list(domain_counts.keys())
        manifest["domain_definition"] = "corruption_type"
        manifest["within_domain_sampling"] = {"severity": "uniform", "values": [1, 2, 3, 4, 5]}
        manifest["external_deployment_law"] = {
            "kind": "uniform_15_corruption_types",
            "weights": {name: 1.0 / 15.0 for name in CORRUPTION_TYPES},
        }
        if experiment == "E3b":
            anchors = list(domain_counts.keys())
            if len(anchors) != 4:
                raise ValueError("E3b requires exactly four pre-registered anchor types")
            manifest["identification_deployment_law"] = {
                "kind": "uniform_four_type_anchor",
                "anchor_types": anchors,
                "weights": {name: 0.25 for name in anchors},
            }
        else:
            manifest["identification_deployment_law"] = {
                "kind": "uniform_15_corruption_types",
                "anchor_types": list(CORRUPTION_TYPES),
                "weights": {name: 1.0 / 15.0 for name in CORRUPTION_TYPES},
            }
    elif protocol == "severity":
        manifest["corruptions"] = "all_15_uniform"
        manifest["domain_definition"] = "severity"
        manifest["within_domain_sampling"] = {"corruption_type": "uniform_15"}
        manifest["external_deployment_law"] = {
            "kind": "uniform_15_corruption_types",
            "weights": {name: 1.0 / 15.0 for name in CORRUPTION_TYPES},
        }
        manifest["identification_deployment_law"] = None
    else:
        raise ValueError(f"Unknown protocol: {protocol}")
    manifest["manifest_id"] = manifest_id(manifest)
    return manifest

