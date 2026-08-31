#!/usr/bin/env python3
"""Create cross-seed summaries and per-environment curves for clean E1."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from collect_results import load_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(args.results_dir)
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("No E1 records found")
    frame = frame[frame["phase"] == "domain_count"].copy()
    if frame.empty:
        raise ValueError("No domain_count records found")
    frame["n_train_domains"] = frame["n_train_domains"].astype(int)

    metric_columns = [
        column for column in ["worst_domain_acc_final", "avg_domain_acc_final", "best_domain_acc_final"]
        if column in frame.columns
    ]
    summary = (
        frame.groupby(["n_train_domains", "algorithm"])[metric_columns]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = [
        column if isinstance(column, str) else "_".join(part for part in column if part)
        for column in summary.columns
    ]
    summary.to_csv(output_dir / "domain_count_by_algorithm.csv", index=False)

    env_columns = sorted(
        [column for column in frame.columns if column.endswith("_acc_final") and column.split("_")[0].replace(".", "", 1).isdigit()],
        key=lambda column: float(column.split("_")[0]),
    )
    long_rows = []
    for _, row in frame.iterrows():
        for column in env_columns:
            long_rows.append({
                "n_train_domains": int(row["n_train_domains"]),
                "algorithm": row["algorithm"],
                "seed": int(row["seed"]),
                "test_env": float(column.split("_")[0]),
                "accuracy": float(row[column]),
            })
    long_frame = pd.DataFrame(long_rows)
    long_frame.to_csv(output_dir / "domain_count_accuracy_by_test_env.csv", index=False)

    algo_display = {
        "erm": "ERM",
        "groupdro": "GroupDRO",
        "inftask": "INF-TASK",
        "irm": "IRM",
        "iro": "IRO",
    }
    algo_colors = {
        "erm": "#0072B2",
        "groupdro": "#D55E00",
        "inftask": "#009E73",
        "irm": "#CC79A7",
        "iro": "#E69F00",
    }
    algo_markers = {
        "erm": "o",
        "groupdro": "s",
        "inftask": "^",
        "irm": "d",
        "iro": "v",
    }
    algo_linestyles = {
        "erm": "-",
        "groupdro": "--",
        "inftask": "-.",
        "irm": ":",
        "iro": "-",
    }

    figure, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=True, sharex=True)
    axes_flat = axes.flatten()
    domain_counts = sorted(long_frame["n_train_domains"].unique())
    algo_order = ["erm", "groupdro", "inftask", "irm", "iro"]

    for axis, domain_count in zip(axes_flat, domain_counts):
        subset = long_frame[long_frame["n_train_domains"] == domain_count]
        for algorithm in algo_order:
            if algorithm not in subset["algorithm"].unique():
                continue
            group = subset[subset["algorithm"] == algorithm]
            curve = group.groupby("test_env")["accuracy"].agg(["mean", "std"]).reset_index()
            disp_name = algo_display.get(algorithm, algorithm.upper())
            axis.plot(
                curve["test_env"],
                curve["mean"],
                marker=algo_markers.get(algorithm, "o"),
                linestyle=algo_linestyles.get(algorithm, "-"),
                color=algo_colors.get(algorithm, "#333333"),
                linewidth=2.2,
                markersize=7,
                label=disp_name,
            )
            if curve["std"].notna().any():
                axis.fill_between(
                    curve["test_env"],
                    curve["mean"] - curve["std"].fillna(0),
                    curve["mean"] + curve["std"].fillna(0),
                    color=algo_colors.get(algorithm, "#333333"),
                    alpha=0.1,
                )
        axis.set_title(f"{domain_count} Source Domains", fontsize=16, pad=8, fontweight="bold")
        axis.set_xlabel("Test Environment e", fontsize=14, labelpad=6)
        axis.set_ylabel("Accuracy", fontsize=14, labelpad=6)
        axis.tick_params(labelsize=12)
        axis.grid(True, linestyle="--", alpha=0.5)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=5, fontsize=14, framealpha=0.9)
    figure.tight_layout(rect=[0, 0.05, 1, 0.98])
    figure.savefig(output_dir / "domain_count_accuracy_by_test_env.png", dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Records: {len(frame)}; seeds: {sorted(frame['seed'].unique())}; conditions: {sorted(frame['n_train_domains'].unique())}")
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
