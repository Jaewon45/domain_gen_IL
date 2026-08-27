#!/usr/bin/env python3
"""Analyze ImageNet-C experiment records and export smoke-friendly summaries."""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


ALGORITHM_DISPLAY_NAMES = {
    "erm": "ERM",
    "groupdro": "GroupDRO",
    "iro": "IRO",
}
BAR_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]


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


def load_records(results_dir: str) -> List[Dict[str, object]]:
    path = Path(results_dir) / "raw" / "train_runs.jsonl"
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def normalize_frames(records: List[Dict[str, object]]):
    run_rows = []
    eval_rows = []
    for record in records:
        run_rows.append({
            key: value for key, value in record.items() if key != "eval_rows"
        })
        for eval_row in record.get("eval_rows", []):
            row = dict(eval_row)
            row["algorithm"] = record["algorithm"]
            row["seed"] = record["seed"]
            row["fold"] = record["fold"]
            row["run_id"] = record["run_id"]
            eval_rows.append(row)
    return pd.DataFrame(run_rows), pd.DataFrame(eval_rows)


def write_csv_outputs(run_df: pd.DataFrame, eval_df: pd.DataFrame, output_dir: Path) -> None:
    run_df.to_csv(output_dir / "summary" / "by_fold_seed.csv", index=False)

    metric_cols = [
        "clean_accuracy",
        "clean_loss",
        "held_out_mean_accuracy",
        "held_out_worst_accuracy",
        "held_out_mean_loss",
        "held_out_worst_loss",
        "all_corruptions_mean_accuracy",
        "all_corruptions_worst_accuracy",
    ]
    existing_metrics = [column for column in metric_cols if column in run_df.columns]
    by_algorithm = (
        run_df.groupby(["fold", "algorithm"])[existing_metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    by_algorithm.columns = [
        column if isinstance(column, str) else "_".join(token for token in column if token)
        for column in by_algorithm.columns
    ]
    by_algorithm.to_csv(output_dir / "summary" / "by_fold_algorithm.csv", index=False)

    held_out_only = eval_df[eval_df["split"] == "held_out"].copy()
    held_out_only.to_csv(output_dir / "summary" / "held_out_only.csv", index=False)
    eval_df.to_csv(output_dir / "summary" / "all_corruptions.csv", index=False)


def plot_worst_domain(run_df: pd.DataFrame, output_dir: Path) -> None:
    plot_df = run_df[["algorithm", "held_out_worst_accuracy"]].copy()
    plot_df["algorithm"] = plot_df["algorithm"].map(lambda value: ALGORITHM_DISPLAY_NAMES.get(value, value.upper()))
    plt.figure(figsize=(8, 5))
    plt.bar(plot_df["algorithm"], plot_df["held_out_worst_accuracy"], color=BAR_COLORS[: len(plot_df)])
    plt.ylabel("Worst held-out accuracy")
    plt.title("ImageNet-C Smoke: Worst Held-Out Domain Accuracy")
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "held_out_worst_domain.png", dpi=200)
    plt.close()


def plot_mean_accuracy(run_df: pd.DataFrame, output_dir: Path) -> None:
    plot_df = run_df[["algorithm", "held_out_mean_accuracy"]].copy()
    plot_df["algorithm"] = plot_df["algorithm"].map(lambda value: ALGORITHM_DISPLAY_NAMES.get(value, value.upper()))
    plt.figure(figsize=(8, 5))
    plt.bar(plot_df["algorithm"], plot_df["held_out_mean_accuracy"], color=BAR_COLORS[: len(plot_df)])
    plt.ylabel("Mean held-out accuracy")
    plt.title("ImageNet-C Smoke: Mean Held-Out Accuracy")
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "held_out_mean_accuracy.png", dpi=200)
    plt.close()


def plot_severity_profile(eval_df: pd.DataFrame, output_dir: Path) -> None:
    plot_df = eval_df[eval_df["split"] == "held_out"].copy()
    if plot_df.empty:
        return
    plt.figure(figsize=(8, 5))
    for algorithm, group in plot_df.groupby("algorithm"):
        severity_curve = group.groupby("severity")["accuracy"].mean().reset_index()
        plt.plot(
            severity_curve["severity"],
            severity_curve["accuracy"],
            marker="o",
            label=ALGORITHM_DISPLAY_NAMES.get(algorithm, algorithm.upper()),
        )
    plt.xlabel("Severity")
    plt.ylabel("Mean held-out accuracy")
    plt.title("ImageNet-C Smoke: Severity Profile by Algorithm")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "severity_profile_by_algorithm.png", dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze ImageNet-C run artifacts.")
    parser.add_argument("results_dir")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir or args.results_dir)
    for subdir in ["summary", "plots"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    apply_plot_style()
    records = load_records(args.results_dir)
    run_df, eval_df = normalize_frames(records)
    if run_df.empty:
        raise ValueError("No training records found to analyze.")

    write_csv_outputs(run_df, eval_df, output_dir)
    plot_worst_domain(run_df, output_dir)
    plot_mean_accuracy(run_df, output_dir)
    plot_severity_profile(eval_df, output_dir)
    print(f"Wrote ImageNet-C analysis outputs to {output_dir}")


if __name__ == "__main__":
    main()