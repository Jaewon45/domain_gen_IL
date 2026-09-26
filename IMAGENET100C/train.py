#!/usr/bin/env python3
"""Configuration-driven ImageNet-100-C training entry point."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Dict, List, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from .algorithms import DomainAlgorithm
    from .data import (
        InfiniteDomainBatches, datasets_from_manifest, load_imagenet100,
        proportional_batch_sizes, validate_label_space,
    )
    from .manifests import build_manifest, load_manifest, save_manifest
    from .models import WEIGHT_VERSION, build_model, build_transform_and_spec
except ImportError:
    from algorithms import DomainAlgorithm
    from data import InfiniteDomainBatches, datasets_from_manifest, load_imagenet100, proportional_batch_sizes, validate_label_space
    from manifests import build_manifest, load_manifest, save_manifest
    from models import WEIGHT_VERSION, build_model, build_transform_and_spec


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> Dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def uniform_counts(total: int, domains: List[str]) -> Dict[str, int]:
    quotient, remainder = divmod(int(total), len(domains))
    return {domain: quotient + (position < remainder) for position, domain in enumerate(domains)}


def scale_counts(counts: Mapping[str, int], maximum: int):
    if maximum <= 0 or sum(counts.values()) <= maximum:
        return dict(counts)
    active = [name for name, count in counts.items() if count > 0]
    raw = {name: maximum * counts[name] / sum(counts.values()) for name in active}
    scaled = {name: max(1, int(value)) for name, value in raw.items()}
    while sum(scaled.values()) < maximum:
        name = max(active, key=lambda key: raw[key] - scaled[key])
        scaled[name] += 1
    while sum(scaled.values()) > maximum:
        candidates = [name for name in active if scaled[name] > 1]
        name = max(candidates, key=lambda key: scaled[key] - raw[key])
        scaled[name] -= 1
    return {name: scaled.get(name, 0) for name in counts}


def resolve_experiment(config, experiment: str, condition: str, source_count: int, samples_per_type: int):
    definitions = config["experiments"]
    if experiment == "severity_support":
        definition = definitions[experiment]
        domains = [str(value) for value in definition["source_severities"]]
        return "severity", uniform_counts(int(definition["total_budget"]), domains), "severity_1_3"

    source_order = list(config["primary_source_order"])
    definition = definitions[experiment]
    if experiment == "E0":
        domains = source_order[: int(definition["source_count"])]
        return "mechanism", dict(zip(domains, definition["counts"])), condition or "fixed_four"
    if experiment == "E1":
        count = int(source_count)
        if count not in definition["source_counts"]:
            raise ValueError(f"E1 source_count must be one of {definition['source_counts']}")
        domains = source_order[:count]
        return "mechanism", uniform_counts(int(definition["total_budget"]), domains), f"{count}_types"
    if experiment == "E2":
        amount = int(samples_per_type)
        if amount not in definition["samples_per_type"]:
            raise ValueError(f"E2 samples_per_type must be one of {definition['samples_per_type']}")
        domains = source_order[: int(definition["source_count"])]
        return "mechanism", {domain: amount for domain in domains}, f"{amount}_per_type"
    if experiment in {"E3", "E3b"}:
        if condition not in definition["conditions"]:
            raise ValueError(f"{experiment} condition must be one of {list(definition['conditions'])}")
        domains = list(config["e3b_anchor_types"]) if experiment == "E3b" else source_order[: int(definition["source_count"])]
        return "mechanism", dict(zip(domains, definition["conditions"][condition])), condition
    raise ValueError("Training experiment must be E0, E1, E2, E3, E3b, or severity_support")


def _uniform_upper_cvar(values, alpha: float) -> float:
    ordered = sorted((float(value) for value in values), reverse=True)
    if alpha >= 1.0:
        return ordered[0]
    tail_mass = (1.0 - alpha) * len(ordered)
    remaining, total = tail_mass, 0.0
    for value in ordered:
        used = min(1.0, remaining)
        total += used * value
        remaining -= used
        if remaining <= 1e-12:
            break
    return total / tail_mass


def source_validation_objective(model, datasets, device, batch_size: int, lambda_grid) -> float:
    """Uniform-lambda mean of source-domain CVaR cross-entropies."""
    if not datasets:
        return float("nan")
    model.eval()
    cvars = []
    for lambda_value in lambda_grid:
        losses = []
        with torch.no_grad():
            for dataset in datasets.values():
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
                domain_total, count = 0.0, 0
                for images, targets in loader:
                    images, targets = images.to(device), targets.to(device)
                    preference = torch.full((images.shape[0], 1), lambda_value, device=device)
                    domain_total += float(F.cross_entropy(model(images, preference), targets, reduction="sum"))
                    count += targets.numel()
                losses.append(domain_total / count)
        cvars.append(_uniform_upper_cvar(losses, lambda_value))
    return float(np.mean(cvars))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ImageNet-100-C domain-generalization models")
    parser.add_argument("--config", default=str(Path(__file__).parent / "configs" / "experiments.json"))
    parser.add_argument("--experiment", choices=["E0", "E1", "E2", "E3", "E3b", "severity_support"], default="E0")
    parser.add_argument("--condition", default="balanced")
    parser.add_argument("--source_count", type=int, default=4)
    parser.add_argument("--samples_per_type", type=int, default=1000)
    parser.add_argument("--algorithm", choices=["erm", "groupdro", "inftask", "iro"], default="erm")
    parser.add_argument("--backbone_mode", choices=["frozen_feature_pilot", "finetune_last_stage"], default="frozen_feature_pilot")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_lambda_samples", type=int, choices=[2, 4, 8], default=4)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--validation_per_domain", type=int, default=100)
    parser.add_argument("--max_source_images", type=int, default=0, help="Scale source counts down; use 100 for loader smoke")
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--groupdro_eta", type=float, default=0.1)
    parser.add_argument(
        "--checkpoint_selection",
        choices=["final", "source_val_uniform_lambda_cvar"],
        default="final",
        help="Default is the fixed-budget final checkpoint; no target validation is used.",
    )
    parser.add_argument("--selection_lambda_grid", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--manifest", default=None, help="Reuse an existing immutable assignment manifest")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "checkpoints").mkdir()
    config = load_config(args.config)
    protocol, domain_counts, condition = resolve_experiment(
        config, args.experiment, args.condition, args.source_count, args.samples_per_type
    )
    if args.max_source_images:
        domain_counts = scale_counts(domain_counts, args.max_source_images)

    transform, transform_spec = build_transform_and_spec(config["weight_version"])
    # Request only the source split in the training process. The target
    # validation split is not even instantiated here.
    train_split = load_imagenet100(config["dataset"], config["dataset_revision"], split="train")
    labels = [int(value) for value in train_split["label"]]
    validate_label_space(labels)
    label_feature = train_split.features["label"]
    class_names = list(label_feature.names)
    if len(class_names) != 100:
        raise ValueError(f"Expected 100 class names, got {len(class_names)}")

    manifest_path = output_dir / "manifest.json"
    if args.manifest:
        manifest = load_manifest(args.manifest)
        if int(manifest["global_seed"]) != args.seed:
            raise ValueError("Manifest seed and --seed differ")
        expected_counts = {str(name): int(count) for name, count in domain_counts.items()}
        if manifest["protocol"] != protocol or manifest["requested_domain_counts"] != expected_counts:
            raise ValueError("Reused manifest does not match the requested protocol/domain counts")
    else:
        validation_per_domain = min(args.validation_per_domain, max(1, args.max_source_images // max(1, len(domain_counts)))) if args.max_source_images else args.validation_per_domain
        manifest = build_manifest(
            dataset_name=config["dataset"], dataset_revision=config["dataset_revision"],
            class_names=class_names, labels=labels, protocol=protocol,
            experiment=args.experiment, condition=condition, domain_counts=domain_counts,
            seed=args.seed, transform_spec=transform_spec,
            validation_per_domain=validation_per_domain,
            weight_version=config["weight_version"],
        )
    selection_lambda_grid = [float(value) for value in args.selection_lambda_grid.split(",") if value.strip()]
    if not selection_lambda_grid or any(value < 0 or value > 1 for value in selection_lambda_grid):
        raise ValueError("selection_lambda_grid values must lie in [0,1]")
    manifest["run_configuration"] = {
        "algorithm": args.algorithm,
        "backbone_mode": args.backbone_mode,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "checkpoint_selection": args.checkpoint_selection,
        "selection_lambda_grid": selection_lambda_grid if args.checkpoint_selection != "final" else None,
        "num_lambda_samples": args.num_lambda_samples,
    }
    save_manifest(manifest, str(manifest_path))
    manifest = load_manifest(str(manifest_path))

    train_datasets = datasets_from_manifest(train_split, manifest, transform, validation=False)
    validation_datasets = datasets_from_manifest(train_split, manifest, transform, validation=True)
    batch_sizes = proportional_batch_sizes(manifest["active_domain_counts"], args.batch_size)
    batches = InfiniteDomainBatches(train_datasets, batch_sizes, args.seed, args.workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.backbone_mode, weight_version=config["weight_version"]).to(device)
    algorithm = DomainAlgorithm(
        args.algorithm, model, learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        groupdro_eta=args.groupdro_eta, num_lambda_samples=args.num_lambda_samples,
        seed=args.seed, device=device,
    )

    best_loss = float("inf")
    best_model = None
    history = []
    for step in range(1, args.steps + 1):
        update = algorithm.update(batches.next(step))
        record = {"step": step, **update}
        if args.checkpoint_selection == "source_val_uniform_lambda_cvar" and (step % args.eval_every == 0 or step == args.steps):
            selection_loss = source_validation_objective(
                model, validation_datasets, device, args.batch_size, selection_lambda_grid
            )
            record["source_validation_uniform_lambda_cvar"] = selection_loss
            if selection_loss < best_loss:
                best_loss = selection_loss
                best_model = copy.deepcopy(model.state_dict())
        history.append(record)
        print(json.dumps(record, sort_keys=True))

    checkpoint = {
        "schema_version": 1,
        "args": vars(args),
        "dataset": manifest["dataset"],
        "manifest_path": str(manifest_path.resolve()),
        "manifest_id": manifest["manifest_id"],
        "model_mode": args.backbone_mode,
        "weight_version": config["weight_version"],
        "algorithm": args.algorithm,
        "num_classes": 100,
        "selection": {
            "policy": "fixed_training_budget_final_checkpoint",
            "metric": None,
            "lambda_grid": None,
            "best_value": None,
            "uses_target_validation": False,
        },
        "model_state": copy.deepcopy(model.state_dict()),
        "final_algorithm_state": algorithm.checkpoint_state(),
    }
    torch.save(checkpoint, output_dir / "checkpoints" / "final.pt")
    if args.checkpoint_selection == "source_val_uniform_lambda_cvar":
        if best_model is None:
            raise RuntimeError("No source-validation checkpoint was selected")
        best_checkpoint = dict(checkpoint)
        best_checkpoint["selection"] = {
            "policy": "best_source_only_validation_checkpoint",
            "metric": "uniform_lambda_source_validation_cvar_cross_entropy",
            "lambda_grid": selection_lambda_grid,
            "best_value": best_loss,
            "uses_target_validation": False,
        }
        best_checkpoint["model_state"] = best_model
        torch.save(best_checkpoint, output_dir / "checkpoints" / "best_source_val.pt")
    with (output_dir / "history.jsonl").open("w", encoding="utf-8") as handle:
        for record in history:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(f"Saved run to {output_dir}")


if __name__ == "__main__":
    main()

