#!/usr/bin/env python3
"""Create cross-seed summary tables for the corrected CMNIST E3 study."""

import argparse
from pathlib import Path

import pandas as pd

from collect_results import load_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    records = pd.DataFrame(load_records(args.results_dir))
    if records.empty:
        raise ValueError("No records found")
    frame = records[records["phase"] == "imbalance"].copy()
    if frame.empty:
        raise ValueError("No imbalance records found")

    metrics = [
        "worst_domain_acc_final",
        "avg_domain_acc_final",
        "best_domain_acc_final",
        "worst_domain_acc_best",
        "avg_domain_acc_best",
        "best_domain_acc_best",
    ]
    metrics = [metric for metric in metrics if metric in frame.columns]
    summary = (
        frame.groupby(["imbalance_type", "algorithm"])[metrics]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = [
        column if isinstance(column, str) else "_".join(part for part in column if part)
        for column in summary.columns
    ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "imbalance_by_algorithm_cross_seed.csv"
    summary.to_csv(summary_path, index=False)

    env_columns = sorted(
        [
            column for column in frame.columns
            if column.endswith("_acc_final")
            and column.split("_")[0].replace(".", "", 1).isdigit()
        ],
        key=lambda column: float(column.split("_")[0]),
    )
    long_rows = []
    for _, row in frame.iterrows():
        for column in env_columns:
            long_rows.append({
                "imbalance_type": row["imbalance_type"],
                "algorithm": row["algorithm"],
                "seed": int(row["seed"]),
                "test_env": float(column.split("_")[0]),
                "accuracy": float(row[column]),
            })
    long_frame = pd.DataFrame(long_rows)
    long_frame.to_csv(output_dir / "imbalance_accuracy_by_test_env_cross_seed.csv", index=False)
    print(f"Records: {len(frame)}; summary rows: {len(summary)}; output: {output_dir}")


if __name__ == "__main__":
    main()
