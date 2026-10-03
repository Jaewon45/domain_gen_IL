#!/usr/bin/env python3
"""Build E3b tables and plots for CMNIST, ImageNet100C, and simulations."""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import pandas as pd


def write_summary(raw: pd.DataFrame, output: Path, dataset: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output / "raw_results.csv", index=False)
    if raw.empty:
        return
    group_columns = [column for column in ("algorithm", "condition") if column in raw]
    metric_columns = [column for column in ("mean_accuracy", "worst_accuracy", "clean_accuracy") if column in raw]
    summary = raw.groupby(group_columns, dropna=False)[metric_columns].agg(["mean", "std", "count"]).reset_index()
    summary.columns = ["_".join(str(part) for part in column if part).rstrip("_") for column in summary.columns]
    summary.to_csv(output / "summary_by_condition.csv", index=False)
    with (output / "summary_by_condition.tex").open("w", encoding="utf-8") as handle:
        handle.write(summary.to_latex(index=False, float_format=lambda value: f"{value:.4f}"))
    for metric in metric_columns:
        import matplotlib.pyplot as plt
        pivot = raw.groupby(["condition", "algorithm"], dropna=False)[metric].mean().unstack()
        axis = pivot.plot(kind="bar", figsize=(9, 5), ylabel=metric.replace("_", " ").title())
        axis.set_title(f"{dataset} E3b {metric.replace('_', ' ')}")
        axis.figure.tight_layout()
        axis.figure.savefig(output / f"{metric}_by_condition.png", dpi=200)
        plt.close(axis.figure)


def cmnist_records(root: Path) -> pd.DataFrame:
    rows = []
    for path in root.rglob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            args = record.get("args", {})
            condition = record.get("tail_support_condition") or args.get("tail_support_condition") or "unknown"
            accuracies = [value for key, value in record.items() if key.endswith("_acc_final") and isinstance(value, (int, float))]
            if not accuracies:
                continue
            rows.append({
                "dataset": "CMNIST",
                "algorithm": record.get("algorithm", "unknown"),
                "condition": condition,
                "seed": record.get("seed"),
                "mean_accuracy": sum(accuracies) / len(accuracies),
                "worst_accuracy": min(accuracies),
                "source_file": str(path),
            })
    return pd.DataFrame(rows)


def imagenet_records(root: Path) -> pd.DataFrame:
    rows = []
    for path in root.rglob("evaluation.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            rows.append({
                "dataset": "ImageNet100C",
                "algorithm": record.get("algorithm"),
                "condition": record.get("condition"),
                "seed": record.get("seed"),
                "mean_accuracy": 1.0 - float(record["mean_domain_error"]),
                "worst_accuracy": 1.0 - float(record["worst_domain_error"]),
                "clean_accuracy": record.get("clean", {}).get("top1"),
                "lambda": record.get("lambda"),
                "source_file": str(path),
            })
    return pd.DataFrame(rows)


def simulation_report(root: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    inventory = []
    for path in sorted(root.rglob("*.csv")):
        try:
            frame = pd.read_csv(path)
            inventory.append({"source_file": str(path), "rows": len(frame), "columns": ",".join(frame.columns)})
            relative = path.relative_to(root)
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(destination, index=False)
        except (OSError, pd.errors.ParserError):
            continue
    pd.DataFrame(inventory).to_csv(output / "simulation_file_inventory.csv", index=False)
    width_files = list(root.rglob("identification_width_by_alpha.csv"))
    if width_files:
        import matplotlib.pyplot as plt
        frame = pd.read_csv(width_files[0])
        alpha = next((column for column in frame.columns if column in {"alpha", "risk_level"}), None)
        width = next((column for column in frame.columns if "width" in column), None)
        if alpha and width:
            axis = frame.plot(x=alpha, y=width, marker="o", figsize=(8, 5), legend=False)
            axis.set_title("Deterministic synthetic identification width")
            axis.set_ylabel(width.replace("_", " ").title())
            axis.figure.tight_layout()
            axis.figure.savefig(output / "identification_width_by_alpha.png", dpi=220)
            plt.close(axis.figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmnist-root", type=Path, default=Path("results/CMNIST"))
    parser.add_argument("--imagenet-root", type=Path, default=Path("results/ImgNet/e3b"))
    parser.add_argument("--simulation-root", type=Path, default=Path("sim"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/reports/E3b"))
    args = parser.parse_args()
    write_summary(cmnist_records(args.cmnist_root), args.output_dir / "CMNIST", "CMNIST")
    write_summary(imagenet_records(args.imagenet_root), args.output_dir / "ImgNet", "ImageNet100C")
    simulation_report(args.simulation_root, args.output_dir / "simulation")
    print(f"Wrote E3b reports to {args.output_dir}")


if __name__ == "__main__":
    main()
