#!/usr/bin/env python3
"""Smoke-capable ImageNet-C evaluation repeatability pipeline."""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from datasets import (
    CORRUPTIONS,
    SEVERITIES,
    clean_imagenet_dataset,
    default_imagenet_transforms,
    ensure_output_subdirs,
    imagefolder_dataset,
    maybe_import_torchvision,
    write_manifest,
)


def apply_plot_style() -> None:
    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 18,
    })


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


def weighted_cvar(losses: Sequence[float], alpha: float) -> float:
    loss_array = np.asarray(losses, dtype=float)
    if loss_array.size == 0:
        return float("nan")
    alpha = float(min(max(alpha, 0.0), 1.0))
    quantile = float(np.quantile(loss_array, alpha, method="linear"))
    tail_losses = loss_array[loss_array >= quantile]
    return float(np.mean(tail_losses))


def smoke_condition_metrics(corruption_index: int, severity: int) -> Dict[str, float]:
    severity_gap = 0.028 * (severity - 1)
    corruption_gap = 0.008 * corruption_index
    base_accuracy = 0.78 - severity_gap - corruption_gap
    oscillation = 0.01 * math.sin((corruption_index + 1) * severity)
    accuracy = float(np.clip(base_accuracy + oscillation, 0.05, 0.99))

    base_loss = 0.62 + 0.06 * (severity - 1) + 0.015 * corruption_index
    loss = float(base_loss + 0.02 * abs(math.cos((corruption_index + 1) * severity)))
    return {
        "accuracy": accuracy,
        "loss": loss,
    }


def build_smoke_frame(repetition: int) -> pd.DataFrame:
    rows = []
    for corruption_index, corruption in enumerate(CORRUPTIONS):
        for severity in SEVERITIES:
            metrics = smoke_condition_metrics(corruption_index, severity)
            rows.append(
                {
                    "repetition": repetition,
                    "dataset": "imagenet_c_smoke",
                    "model_name": "resnet50_smoke",
                    "corruption": corruption,
                    "severity": severity,
                    "domain_id": corruption,
                    "condition_id": f"{corruption}_s{severity}",
                    "accuracy": metrics["accuracy"],
                    "loss": metrics["loss"],
                }
            )
    return pd.DataFrame(rows)


def write_raw_repetitions(frames: List[pd.DataFrame], output_dir: Path) -> None:
    for repetition, frame in enumerate(frames):
        raw_path = output_dir / "raw" / f"repetition_{repetition}.jsonl"
        with raw_path.open("w", encoding="utf-8") as handle:
            for row in frame.to_dict(orient="records"):
                handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_repeatability_summary(frames: List[pd.DataFrame], output_dir: Path) -> pd.DataFrame:
    stacked = pd.concat(frames, ignore_index=True)
    grouped = (
        stacked.groupby(["corruption", "severity"])[["accuracy", "loss"]]
        .agg(["mean", "min", "max", "std"])
        .reset_index()
    )
    grouped.columns = [
        column if isinstance(column, str) else "_".join(token for token in column if token)
        for column in grouped.columns
    ]
    grouped["accuracy_range"] = grouped["accuracy_max"] - grouped["accuracy_min"]
    grouped["loss_range"] = grouped["loss_max"] - grouped["loss_min"]
    grouped.to_csv(output_dir / "summary" / "repeatability_summary.csv", index=False)
    return stacked


def write_corruption_severity_matrix(stacked: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    matrix = (
        stacked.groupby(["corruption", "severity"])["accuracy"]
        .mean()
        .reset_index()
        .pivot(index="corruption", columns="severity", values="accuracy")
        .reindex(CORRUPTIONS)
    )
    matrix.to_csv(output_dir / "summary" / "corruption_severity_matrix.csv")
    return matrix


def build_lambda_curve(stacked: pd.DataFrame, lambda_grid: Sequence[float], output_dir: Path) -> pd.DataFrame:
    mean_losses = (
        stacked.groupby(["repetition", "corruption"])["loss"]
        .mean()
        .reset_index()
    )
    rows = []
    for lambda_value in lambda_grid:
        per_rep_scores = []
        for repetition, group in mean_losses.groupby("repetition"):
            score = weighted_cvar(group["loss"].tolist(), lambda_value)
            per_rep_scores.append(score)
        rows.append(
            {
                "lambda": float(lambda_value),
                "aggregated_risk_mean": float(np.mean(per_rep_scores)),
                "aggregated_risk_std": float(np.std(per_rep_scores)),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "summary" / "lambda_sensitivity_curve.csv", index=False)
    return frame


def plot_heatmap(matrix: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(10, 7))
    plt.imshow(matrix.to_numpy(), aspect="auto", cmap="viridis")
    plt.colorbar(label="Accuracy")
    plt.xticks(range(len(SEVERITIES)), [str(severity) for severity in SEVERITIES])
    plt.yticks(range(len(CORRUPTIONS)), [corruption.replace("_", " ") for corruption in CORRUPTIONS])
    plt.xlabel("Severity")
    plt.ylabel("Corruption")
    plt.title("ImageNet-C Smoke: Corruption by Severity Accuracy")
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "corruption_severity_heatmap.png", dpi=200)
    plt.close()


def plot_corruption_bar(stacked: pd.DataFrame, output_dir: Path) -> None:
    corruption_means = (
        stacked.groupby("corruption")["accuracy"]
        .mean()
        .reindex(CORRUPTIONS)
    )
    plt.figure(figsize=(11, 5))
    plt.bar(range(len(corruption_means)), corruption_means.to_numpy(), color="#4C78A8")
    plt.xticks(
        range(len(corruption_means)),
        [corruption.replace("_", " ") for corruption in corruption_means.index],
        rotation=45,
        ha="right",
    )
    plt.ylabel("Mean accuracy")
    plt.title("ImageNet-C Smoke: Mean Accuracy by Corruption")
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "corruption_accuracy_bar.png", dpi=200)
    plt.close()


def plot_lambda_curve(lambda_frame: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(lambda_frame["lambda"], lambda_frame["aggregated_risk_mean"], marker="o", color="#E45756")
    plt.xlabel("Lambda")
    plt.ylabel("Aggregated risk")
    plt.title("ImageNet-C Smoke: Lambda Sensitivity")
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "lambda_sensitivity_curve.png", dpi=200)
    plt.close()


def run_smoke(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    ensure_output_subdirs(output_dir, ["raw", "summary", "plots", "manifests"], allow_existing_files=False)
    apply_plot_style()

    repetitions = [build_smoke_frame(repetition) for repetition in range(args.repetitions)]
    write_raw_repetitions(repetitions, output_dir)
    stacked = write_repeatability_summary(repetitions, output_dir)
    matrix = write_corruption_severity_matrix(stacked, output_dir)
    lambda_frame = build_lambda_curve(stacked, build_lambda_grid(args.lambda_grid), output_dir)

    plot_heatmap(matrix, output_dir)
    plot_corruption_bar(stacked, output_dir)
    plot_lambda_curve(lambda_frame, output_dir)
    write_manifest(output_dir, "eval_repeatability_repetitions.txt", os.sys.argv, "Smoke repeatability manifest")

    print(f"Smoke repeatability outputs written to {output_dir}")
    print(f"Rows per repetition: {len(repetitions[0])}")
    print(f"Lambda grid points: {len(lambda_frame)}")


def build_real_frame(
    model,
    clean_loader,
    corruption_loaders: List[Dict[str, object]],
    repetition: int,
    device: torch.device,
) -> pd.DataFrame:
    rows = []
    loss_fn = torch.nn.CrossEntropyLoss()
    model.eval()
    with torch.no_grad():
        if clean_loader is not None:
            clean_accuracy, clean_loss = evaluate_loader(model, clean_loader, loss_fn, device)
            rows.append(
                {
                    "repetition": repetition,
                    "dataset": "imagenet_c",
                    "model_name": type(model).__name__.lower(),
                    "corruption": "clean",
                    "severity": 0,
                    "domain_id": "clean",
                    "condition_id": "clean",
                    "accuracy": clean_accuracy,
                    "loss": clean_loss,
                }
            )
        for loader_info in corruption_loaders:
            accuracy, loss = evaluate_loader(model, loader_info["loader"], loss_fn, device)
            rows.append(
                {
                    "repetition": repetition,
                    "dataset": "imagenet_c",
                    "model_name": type(model).__name__.lower(),
                    "corruption": loader_info["corruption"],
                    "severity": loader_info["severity"],
                    "domain_id": loader_info["corruption"],
                    "condition_id": f"{loader_info['corruption']}_s{loader_info['severity']}",
                    "accuracy": accuracy,
                    "loss": loss,
                }
            )
    return pd.DataFrame(rows)


def evaluate_loader(model, loader, loss_fn, device: torch.device) -> tuple:
    total_correct = 0
    total_examples = 0
    total_loss = 0.0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        batch_loss = loss_fn(logits, targets)
        total_loss += batch_loss.item() * targets.shape[0]
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_examples += targets.shape[0]
    return float(total_correct / total_examples), float(total_loss / total_examples)


def maybe_subset_loader(dataset, batch_size: int):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def load_real_model(checkpoint_path: Optional[str]):
    _, models_module = maybe_import_torchvision()
    _, weights = default_imagenet_transforms(models_module)
    model = models_module.resnet50(weights=weights if checkpoint_path is None else None)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
    return model, weights


def run_real(args: argparse.Namespace) -> None:
    imagenet_root = args.imagenet_root or args.imagenet_val_root
    if imagenet_root is None or args.imagenet_c_root is None:
        raise ValueError("Real mode requires --imagenet_root and --imagenet_c_root")

    output_dir = Path(args.output_dir)
    ensure_output_subdirs(output_dir, ["raw", "summary", "plots", "manifests"], allow_existing_files=False)
    apply_plot_style()
    _, models_module = maybe_import_torchvision()
    transform, _ = default_imagenet_transforms(models_module)

    clean_dataset = clean_imagenet_dataset(
        imagenet_root=imagenet_root,
        transform=transform,
        max_images=args.max_images_per_condition,
        backend=args.clean_loader_backend,
    )
    clean_loader = maybe_subset_loader(clean_dataset, args.batch_size)
    corruption_loaders = []
    for corruption in CORRUPTIONS:
        for severity in SEVERITIES:
            corruption_root = os.path.join(args.imagenet_c_root, corruption, str(severity))
            if not os.path.isdir(corruption_root):
                raise FileNotFoundError(f"Missing ImageNet-C condition directory: {corruption_root}")
            dataset = imagefolder_dataset(corruption_root, transform, args.max_images_per_condition)
            corruption_loaders.append(
                {
                    "corruption": corruption,
                    "severity": severity,
                    "loader": maybe_subset_loader(dataset, args.batch_size),
                }
            )

    model, _ = load_real_model(args.checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    frames = [build_real_frame(model, clean_loader, corruption_loaders, repetition, device) for repetition in range(args.repetitions)]
    write_raw_repetitions(frames, output_dir)
    stacked = write_repeatability_summary(frames, output_dir)
    non_clean = stacked[stacked["corruption"] != "clean"].copy()
    matrix = write_corruption_severity_matrix(non_clean, output_dir)
    lambda_frame = build_lambda_curve(non_clean, build_lambda_grid(args.lambda_grid), output_dir)
    plot_heatmap(matrix, output_dir)
    plot_corruption_bar(non_clean, output_dir)
    plot_lambda_curve(lambda_frame, output_dir)
    write_manifest(output_dir, "eval_repeatability_repetitions.txt", os.sys.argv, "Real repeatability manifest")
    print(f"Real ImageNet-C repeatability outputs written to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ImageNet-C evaluation repeatability pipeline.")
    parser.add_argument("--output_dir", default="results/imagenet_c_eval_repeatability_smoke_v1")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--lambda_grid", default="0.0:1.0:0.1")
    parser.add_argument("--smoke", action="store_true", help="Run deterministic synthetic smoke evaluation.")
    parser.add_argument("--imagenet_root", default=None)
    # Backward-compat alias; prefer --imagenet_root.
    parser.add_argument("--imagenet_val_root", default=None)
    parser.add_argument("--imagenet_c_root", default=None)
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_images_per_condition", type=int, default=None)
    parser.add_argument("--clean_loader_backend", choices=["imagenet", "imagefolder", "auto"], default="imagenet")
    args = parser.parse_args()

    if args.smoke:
        run_smoke(args)
        return
    run_real(args)


if __name__ == "__main__":
    main()