#!/usr/bin/env python3
"""Plot clean lambda sensitivity from prediction-level CMNIST metrics."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DISPLAY_NAMES = {"iro": "IRO", "inftask": "INF-TASK"}
COLORMAP = {"iro": "#0072B2", "inftask": "#D55E00"}
MARKERS = {"iro": "o", "inftask": "s"}
LINESTYLES = {"iro": "-", "inftask": "--"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_csv")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.metrics_csv)
    required = {"algorithm", "lambda_eval", "test_env", "accuracy"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    grouped = frame.groupby(["algorithm", "lambda_eval"])["accuracy"].agg(["mean", "std"]).reset_index()
    figure, axis = plt.subplots(figsize=(9, 6))
    for algorithm, group in grouped.groupby("algorithm"):
        group = group.sort_values("lambda_eval")
        disp_name = DISPLAY_NAMES.get(algorithm, algorithm.upper())
        axis.plot(
            group["lambda_eval"],
            group["mean"],
            marker=MARKERS.get(algorithm, "o"),
            linestyle=LINESTYLES.get(algorithm, "-"),
            color=COLORMAP.get(algorithm, "#333333"),
            linewidth=2.2,
            markersize=7,
            label=disp_name,
        )
        axis.fill_between(
            group["lambda_eval"],
            group["mean"] - group["std"].fillna(0),
            group["mean"] + group["std"].fillna(0),
            color=COLORMAP.get(algorithm, "#333333"),
            alpha=0.12,
        )
    axis.set_xlabel("Lambda (λ)", fontsize=16, labelpad=8)
    axis.set_ylabel("Mean Accuracy Across Deployment Envs", fontsize=16, labelpad=8)
    
    ymin = max(0.0, grouped["mean"].min() - 0.02)
    ymax = min(1.0, grouped["mean"].max() + 0.02)
    axis.set_ylim(ymin, ymax)
    axis.tick_params(labelsize=14)
    axis.grid(True, linestyle="--", alpha=0.5)
    axis.legend(fontsize=14, loc="best", framealpha=0.9)
    figure.tight_layout()
    path = output_dir / "lambda_prediction_accuracy_curve.png"
    figure.savefig(path, dpi=300)
    plt.close(figure)
    grouped.to_csv(output_dir / "lambda_prediction_accuracy_summary.csv", index=False)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
