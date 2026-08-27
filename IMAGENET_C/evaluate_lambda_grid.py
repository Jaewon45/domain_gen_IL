#!/usr/bin/env python3
"""Evaluate ImageNet-C checkpoints across a lambda grid."""

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from datasets import build_real_feature_splits, smoke_feature_splits
from models import LinearClassifier


def build_lambda_grid(grid_spec: str) -> List[float]:
    if "," in grid_spec:
        return [float(value.strip()) for value in grid_spec.split(",") if value.strip()]
    if ":" in grid_spec:
        start, stop, step = [float(value) for value in grid_spec.split(":")]
        grid = []
        current = start
        while current <= stop + 1e-9:
            grid.append(round(current, 10))
            current += step
        return grid
    raise ValueError("lambda_grid must be comma-separated or start:stop:step")


def weighted_cvar(losses: List[float], alpha: float) -> float:
    loss_array = np.asarray(losses, dtype=float)
    quantile = float(np.quantile(loss_array, alpha, method="linear"))
    tail_losses = loss_array[loss_array >= quantile]
    return float(np.mean(tail_losses))


def resolve_checkpoint_paths(checkpoint_arg: str) -> List[str]:
    if os.path.isdir(checkpoint_arg):
        return sorted(glob.glob(os.path.join(checkpoint_arg, "*.pt")))
    return sorted(glob.glob(checkpoint_arg))


def load_split_bundle(checkpoint_args: Dict[str, object]):
    if checkpoint_args["data_mode"] == "smoke":
        return smoke_feature_splits(
            folds_path=checkpoint_args["folds_path"],
            fold_name=checkpoint_args["fold"],
            seed=int(checkpoint_args["seed"]),
            feature_dim=int(checkpoint_args["smoke_feature_dim"]),
            num_classes=int(checkpoint_args["smoke_num_classes"]),
            train_samples_per_class=int(checkpoint_args["smoke_train_samples_per_class"]),
            val_samples_per_class=int(checkpoint_args["smoke_val_samples_per_class"]),
            test_samples_per_class=int(checkpoint_args["smoke_test_samples_per_class"]),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return build_real_feature_splits(
        folds_path=checkpoint_args["folds_path"],
        fold_name=checkpoint_args["fold"],
        imagenet_train_corrupted_root=checkpoint_args["imagenet_train_corrupted_root"],
        imagenet_root=checkpoint_args.get("imagenet_root") or checkpoint_args.get("imagenet_val_root"),
        imagenet_c_root=checkpoint_args["imagenet_c_root"],
        batch_size=int(checkpoint_args["batch_size"]),
        max_images_per_condition=checkpoint_args.get("max_images_per_condition"),
        device=device,
        clean_loader_backend=checkpoint_args.get("clean_loader_backend", "imagenet"),
    )


def evaluate_rows(model: LinearClassifier, split_bundle: Dict[str, object]) -> List[Dict[str, object]]:
    rows = []
    model.eval()
    with torch.no_grad():
        for condition in list(split_bundle["eval_conditions"]) + [split_bundle["clean_condition"]]:
            logits = model(condition["x"])
            loss = F.cross_entropy(logits, condition["y"]).item()
            accuracy = (logits.argmax(dim=1) == condition["y"]).float().mean().item()
            rows.append(
                {
                    "corruption": condition["corruption"],
                    "severity": int(condition["severity"]),
                    "split": condition["split"],
                    "domain_seen": bool(condition["domain_seen"]),
                    "accuracy": float(accuracy),
                    "loss": float(loss),
                }
            )
    return rows


def summarize_rows(rows: List[Dict[str, object]], lambda_value: float) -> Dict[str, float]:
    held_out_rows = [row for row in rows if row["split"] == "held_out"]
    all_corruption_rows = [row for row in rows if row["split"] in {"held_out", "all"}]
    clean_row = next((row for row in rows if row["split"] == "clean"), None)

    def aggregate(target_rows: List[Dict[str, object]], prefix: str) -> Dict[str, float]:
        if not target_rows:
            return {
                f"{prefix}_mean_accuracy": float("nan"),
                f"{prefix}_worst_accuracy": float("nan"),
                f"{prefix}_cvar_loss": float("nan"),
            }
        accuracies = [float(row["accuracy"]) for row in target_rows]
        losses = [float(row["loss"]) for row in target_rows]
        return {
            f"{prefix}_mean_accuracy": float(np.mean(accuracies)),
            f"{prefix}_worst_accuracy": float(np.min(accuracies)),
            f"{prefix}_cvar_loss": weighted_cvar(losses, lambda_value),
        }

    summary = {}
    summary.update(aggregate(held_out_rows, "held_out"))
    summary.update(aggregate(all_corruption_rows, "all_corruptions"))
    summary["clean_accuracy"] = float(clean_row["accuracy"]) if clean_row else float("nan")
    summary["clean_loss"] = float(clean_row["loss"]) if clean_row else float("nan")
    return summary


def evaluate_checkpoint(checkpoint_path: str, lambda_grid: List[float]) -> List[Dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_args = checkpoint["args"]
    split_bundle = load_split_bundle(checkpoint_args)
    model = LinearClassifier(int(checkpoint_args["feature_dim"]), int(checkpoint_args["num_classes"]))
    model.load_state_dict(checkpoint["model_dict"])

    eval_rows = evaluate_rows(model, split_bundle)
    records = []
    for lambda_value in lambda_grid:
        record = {
            "checkpoint_path": checkpoint_path,
            "run_id": Path(checkpoint_path).stem.replace("_best", ""),
            "algorithm": checkpoint_args["algorithm"],
            "fold": checkpoint_args["fold"],
            "seed": checkpoint_args["seed"],
            "data_mode": checkpoint_args["data_mode"],
            "lambda_eval": float(lambda_value),
            "eval_rows": eval_rows,
        }
        record.update(summarize_rows(eval_rows, lambda_value))
        records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ImageNet-C checkpoints across a lambda grid.")
    parser.add_argument("checkpoint_path")
    parser.add_argument("--lambda_grid", default="0.0:1.0:0.1")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    checkpoint_paths = resolve_checkpoint_paths(args.checkpoint_path)
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints found for: {args.checkpoint_path}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lambda_grid = build_lambda_grid(args.lambda_grid)

    for checkpoint_path in checkpoint_paths:
        records = evaluate_checkpoint(checkpoint_path, lambda_grid)
        out_path = output_dir / f"{Path(checkpoint_path).stem}.jsonl"
        with out_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(f"Saved {len(records)} lambda rows to {out_path}")


if __name__ == "__main__":
    main()