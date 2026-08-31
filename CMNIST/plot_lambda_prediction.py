#!/usr/bin/env python3
"""Plot clean lambda sensitivity from prediction-level CMNIST metrics."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DISPLAY_NAMES = {"iro": "IRO", "inftask": "INF-TASK"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_csv")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.metrics_csv)
    required = {"algorithm", "lambda_eval", "test_env", "accuracy"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    grouped = frame.groupby(["algorithm", "lambda_eval"])["accuracy"].agg(["mean", "std"]).reset_index()
    figure, axis = plt.subplots(figsize=(8, 5))
    for algorithm, group in grouped.groupby("algorithm"):
        group = group.sort_values("lambda_eval")
        axis.plot(group["lambda_eval"], group["mean"], marker="o", label=DISPLAY_NAMES.get(algorithm, algorithm.upper()))
        axis.fill_between(group["lambda_eval"], group["mean"] - group["std"].fillna(0), group["mean"] + group["std"].fillna(0), alpha=0.12)
    axis.set_xlabel("Lambda")
    axis.set_ylabel("Mean accuracy across deployment environments")
    axis.set_title("CMNIST lambda sensitivity from fixed-input predictions")
    axis.set_ylim(0.0, 1.0)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path = output_dir / "lambda_prediction_accuracy_curve.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    grouped.to_csv(output_dir / "lambda_prediction_accuracy_summary.csv", index=False)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
