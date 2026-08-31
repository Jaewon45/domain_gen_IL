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
        column for column in ["worst_domain_acc_best", "avg_domain_acc_best", "best_domain_acc_best"]
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
        [column for column in frame.columns if column.endswith("_acc_best") and column.split("_")[0].replace(".", "", 1).isdigit()],
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

    figure, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
    for axis, domain_count in zip(axes, sorted(long_frame["n_train_domains"].unique())):
        subset = long_frame[long_frame["n_train_domains"] == domain_count]
        for algorithm, group in subset.groupby("algorithm"):
            curve = group.groupby("test_env")["accuracy"].agg(["mean", "std"]).reset_index()
            axis.plot(curve["test_env"], curve["mean"], marker="o", label=algorithm)
            if curve["std"].notna().any():
                axis.fill_between(curve["test_env"], curve["mean"] - curve["std"].fillna(0), curve["mean"] + curve["std"].fillna(0), alpha=0.08)
        axis.set_title(f"{domain_count} source domains")
        axis.set_xlabel("Test environment e")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Accuracy")
    axes[-1].legend(loc="best", fontsize=9)
    figure.suptitle("E1 clean domain-count accuracy across test environments")
    figure.tight_layout()
    figure.savefig(output_dir / "domain_count_accuracy_by_test_env.png", dpi=200)
    plt.close(figure)
    print(f"Records: {len(frame)}; seeds: {sorted(frame['seed'].unique())}; conditions: {sorted(frame['n_train_domains'].unique())}")
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
