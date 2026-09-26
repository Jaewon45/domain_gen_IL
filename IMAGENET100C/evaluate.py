#!/usr/bin/env python3
"""Final clean, 15x5 corruption, E4, and identification evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from .corruptions import CORRUPTION_TYPES, SEVERITIES
    from .data import datasets_from_manifest, final_evaluation_datasets, load_imagenet100
    from .manifests import load_manifest
    from .models import build_model, build_transform_and_spec
except ImportError:
    from corruptions import CORRUPTION_TYPES, SEVERITIES
    from data import datasets_from_manifest, final_evaluation_datasets, load_imagenet100
    from manifests import load_manifest
    from models import build_model, build_transform_and_spec


def weighted_upper_cvar(values: Sequence[float], weights: Sequence[float], alpha: float) -> float:
    """Fractional discrete upper-tail CVaR, including alpha=1 as max loss."""
    if len(values) == 0 or len(values) != len(weights):
        raise ValueError("values and weights must be non-empty and equally sized")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0,1]")
    pairs = sorted(zip(values, weights), key=lambda pair: pair[0], reverse=True)
    total_weight = float(sum(weights))
    if total_weight <= 0:
        raise ValueError("weights must have positive mass")
    normalized = [(float(value), float(weight) / total_weight) for value, weight in pairs]
    tail_mass = 1.0 - float(alpha)
    if tail_mass <= 1e-12:
        return normalized[0][0]
    remaining = tail_mass
    total = 0.0
    for value, weight in normalized:
        used = min(weight, remaining)
        total += used * value
        remaining -= used
        if remaining <= 1e-12:
            break
    return total / tail_mass


def identification_interval(
    observed_losses: Sequence[float],
    epsilon: float,
    alpha: float,
    observed_weights: Sequence[float] = None,
) -> Tuple[float, float]:
    if not observed_losses:
        raise ValueError("At least one observed loss is required")
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0,1]")
    if observed_weights is None:
        observed_weights = [1.0 / len(observed_losses)] * len(observed_losses)
    if len(observed_weights) != len(observed_losses) or sum(observed_weights) <= 0:
        raise ValueError("observed_weights must match losses and have positive mass")
    conditional_weights = [float(weight) / sum(observed_weights) for weight in observed_weights]
    weights = [(1.0 - epsilon) * weight for weight in conditional_weights] + [epsilon]
    lower = weighted_upper_cvar(list(observed_losses) + [0.0], weights, alpha)
    upper = weighted_upper_cvar(list(observed_losses) + [1.0], weights, alpha)
    return lower, upper


def intervals_separated(first: Sequence[float], second: Sequence[float]) -> bool:
    return float(first[1]) < float(second[0]) or float(second[1]) < float(first[0])


def parse_lambda_grid(value: str) -> List[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item < 0 or item > 1 for item in values):
        raise ValueError("lambda values must lie in [0,1]")
    return values


def evaluate_dataset(model, dataset, device, batch_size: int, workers: int, preference: float):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    loss_sum = 0.0
    top1_sum = 0
    top5_sum = 0
    count = 0
    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            lambdas = torch.full((images.shape[0], 1), preference, device=device)
            logits = model(images, lambdas)
            loss_sum += float(F.cross_entropy(logits, targets, reduction="sum"))
            predictions = logits.topk(5, dim=1).indices
            top1_sum += int((predictions[:, 0] == targets).sum())
            top5_sum += int((predictions == targets[:, None]).any(dim=1).sum())
            count += targets.numel()
    return {"count": count, "cross_entropy": loss_sum / count, "top1": top1_sum / count, "top5": top5_sum / count}


def summarize(
    condition_rows: List[Dict[str, object]],
    source_validation_rows: List[Dict[str, object]],
    manifest: Mapping[str, object],
    alphas=(0.5, 0.75, 0.9),
):
    evaluated_corruptions = [
        corruption for corruption in CORRUPTION_TYPES
        if any(row["corruption"] == corruption for row in condition_rows)
    ]
    if not evaluated_corruptions:
        raise ValueError("No corruption conditions were evaluated")
    per_type = []
    for corruption in evaluated_corruptions:
        rows = [row for row in condition_rows if row["corruption"] == corruption]
        per_type.append({
            "corruption": corruption,
            "severity_averaged_top1": float(np.mean([row["top1"] for row in rows])),
            "severity_averaged_top5": float(np.mean([row["top5"] for row in rows])),
            "severity_averaged_cross_entropy": float(np.mean([row["cross_entropy"] for row in rows])),
            "severity_averaged_error": float(np.mean([1.0 - row["top1"] for row in rows])),
        })
    domain_errors = [row["severity_averaged_error"] for row in per_type]
    uniform_weights = [1.0 / len(per_type)] * len(per_type)
    summary = {
        "per_type": per_type,
        "mean_domain_error": float(np.mean(domain_errors)),
        "worst_domain_error": float(np.max(domain_errors)),
        "supplementary_worst_type_severity": max(condition_rows, key=lambda row: 1.0 - row["top1"]),
        "per_severity_average_top1": {
            str(severity): float(np.mean([row["top1"] for row in condition_rows if row["severity"] == severity]))
            for severity in SEVERITIES
        },
        "primary_deployment_law": (
            manifest["external_deployment_law"]
            if evaluated_corruptions == list(CORRUPTION_TYPES)
            else {
                "kind": "pilot_uniform_evaluated_corruption_types",
                "weights": {name: 1.0 / len(evaluated_corruptions) for name in evaluated_corruptions},
            }
        ),
    }
    summary["domain_cvar_error"] = {
        str(alpha): weighted_upper_cvar(domain_errors, uniform_weights, alpha) for alpha in alphas
    }

    identification_law = manifest.get("identification_deployment_law")
    if identification_law is None:
        summary["partial_identification"] = {}
        summary["identification_deployment_law"] = None
        summary["identification_p_obs_source"] = "not_applicable_to_severity_protocol"
        return summary
    law_weights = {str(name): float(weight) for name, weight in identification_law["weights"].items()}
    source_by_domain = {row["domain"]: row for row in source_validation_rows}
    observed_names = [name for name in law_weights if name in source_by_domain]
    observed_losses = [1.0 - float(source_by_domain[name]["top1"]) for name in observed_names]
    observed_weights = [law_weights[name] for name in observed_names]
    epsilon = 1.0 - sum(observed_weights)
    final_by_type = {row["corruption"]: row["severity_averaged_error"] for row in per_type}
    realized_names = list(law_weights)
    missing_realized_types = [name for name in realized_names if name not in final_by_type]
    if missing_realized_types:
        summary["partial_identification"] = {}
        summary["identification_deployment_law"] = identification_law
        summary["identification_p_obs_source"] = "held_out_source_validation_subset"
        summary["identification_not_computed_reason"] = (
            "Pilot corruption subset does not cover the complete identification deployment law"
        )
        summary["identification_missing_evaluated_types"] = missing_realized_types
        return summary
    realized_losses = [final_by_type[name] for name in realized_names]
    realized_weights = [law_weights[name] for name in realized_names]
    intervals = {}
    for alpha in alphas:
        lower, upper = identification_interval(observed_losses, epsilon, alpha, observed_weights)
        realized = weighted_upper_cvar(realized_losses, realized_weights, alpha)
        intervals[str(alpha)] = {
            "lower": lower,
            "upper": upper,
            "width": upper - lower,
            "epsilon": epsilon,
            "upper_endpoint_expected_vacuous": bool(epsilon >= 1.0 - alpha - 1e-12),
            "realized_held_out_deployment_cvar_descriptive_plugin": realized,
        }
    summary["partial_identification"] = intervals
    summary["identification_deployment_law"] = identification_law
    summary["identification_p_obs_source"] = "held_out_source_validation_subset"
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an ImageNet-100-C checkpoint")
    parser.add_argument("checkpoint")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--lambda_grid", default="0.0", help="Use 0,0.1,...,1 for E4")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max_eval_images", type=int, default=None)
    parser.add_argument(
        "--corruption_types",
        default=None,
        help="Optional comma-separated corruption subset for explicitly labeled pilot evaluation.",
    )
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    manifest_path = args.manifest or checkpoint["manifest_path"]
    manifest = load_manifest(manifest_path)
    if manifest["manifest_id"] != checkpoint["manifest_id"]:
        raise ValueError("Checkpoint and manifest IDs differ")
    transform, transform_spec = build_transform_and_spec(checkpoint["weight_version"])
    if transform_spec != manifest["transform"]:
        raise ValueError("Runtime transform differs from the run manifest")
    validation = load_imagenet100(
        manifest["dataset"]["name"], manifest["dataset"]["revision"], split="validation"
    )
    source_train = load_imagenet100(
        manifest["dataset"]["name"], manifest["dataset"]["revision"], split="train"
    )
    source_validation_datasets = datasets_from_manifest(source_train, manifest, transform, validation=True)
    corruption_types = None
    if args.corruption_types:
        corruption_types = [value.strip() for value in args.corruption_types.split(",") if value.strip()]
    clean_dataset, conditions = final_evaluation_datasets(
        validation, transform, int(manifest["global_seed"]), args.max_eval_images,
        corruption_types=corruption_types,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(checkpoint["model_mode"], weight_version=checkpoint["weight_version"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    lambda_grid = parse_lambda_grid(args.lambda_grid)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    records = []
    for preference in lambda_grid:
        clean = evaluate_dataset(model, clean_dataset, device, args.batch_size, args.workers, preference)
        source_validation_rows = []
        for domain, dataset in source_validation_datasets.items():
            source_validation_rows.append({
                "domain": domain,
                **evaluate_dataset(model, dataset, device, args.batch_size, args.workers, preference),
            })
        condition_rows = []
        for (corruption, severity), dataset in conditions.items():
            metrics = evaluate_dataset(model, dataset, device, args.batch_size, args.workers, preference)
            condition_rows.append({"corruption": corruption, "severity": severity, **metrics})
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
            "clean": clean,
            "conditions": condition_rows,
            "source_validation_domain_rows": source_validation_rows,
            "environment_counts": manifest["active_domain_counts"],
            "per_class_counts": manifest["source_class_counts"],
            "selection_uses_target_validation": False,
            "evaluation_scope": {
                "kind": "full" if corruption_types is None and args.max_eval_images is None else "pilot",
                "max_eval_images": args.max_eval_images,
                "validation_sampling": "all" if args.max_eval_images is None else "class_stratified",
                "corruption_types": list(CORRUPTION_TYPES if corruption_types is None else corruption_types),
                "severities": list(SEVERITIES),
            },
        }
        record.update(summarize(condition_rows, source_validation_rows, manifest))
        records.append(record)
        print(json.dumps({key: record[key] for key in ["algorithm", "lambda", "mean_domain_error", "worst_domain_error", "domain_cvar_error"]}, sort_keys=True))

    with (output_dir / "evaluation.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

