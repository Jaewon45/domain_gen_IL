#!/usr/bin/env python3
"""Plot GroupDRO control comparison: Observed-source worst loss vs Held-out target worst accuracy."""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_source_target(csv_path, output_path):
    df = pd.read_csv(csv_path)

    # Clean column names (strip quotes if any)
    df.columns = [c.strip('"') for c in df.columns]

    algo_display = {"erm": "ERM", "groupdro": "GroupDRO"}
    algo_colors = {"erm": "#0072B2", "groupdro": "#D55E00"}
    algo_hatches = {"erm": "", "groupdro": "//"}

    df["seed"] = df["seed"].astype(int)
    seeds = sorted(df["seed"].unique())
    algos = ["erm", "groupdro"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    x = np.arange(len(seeds))
    width = 0.35

    # Left subplot: Observed-Source Worst-Domain Loss
    ax1 = axes[0]
    for i, algo in enumerate(algos):
        sub = df[df["algorithm"].str.strip('"').str.lower() == algo].sort_values("seed")
        y_col = "source_worst_loss_best" if "source_worst_loss_best" in sub.columns else "source_worst_loss_final"
        vals = [sub[sub["seed"] == s][y_col].values[0] if len(sub[sub["seed"] == s]) > 0 else 0.0 for s in seeds]
        disp = algo_display.get(algo, algo.upper())
        offset = (i - 0.5) * width
        ax1.bar(
            x + offset,
            vals,
            width,
            label=disp,
            color=algo_colors[algo],
            edgecolor="black",
            linewidth=0.8,
            hatch=algo_hatches[algo],
        )

    ax1.set_xlabel("Seed", fontsize=14, labelpad=6)
    ax1.set_ylabel("Observed-Source Worst Loss", fontsize=14, labelpad=6)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Seed {s}" for s in seeds], fontsize=12)
    ax1.tick_params(labelsize=12)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.legend(fontsize=13, loc="upper right", framealpha=0.9)

    # Right subplot: Held-Out Target Worst-Domain Accuracy
    ax2 = axes[1]
    for i, algo in enumerate(algos):
        sub = df[df["algorithm"].str.strip('"').str.lower() == algo].sort_values("seed")
        y_col = "target_worst_accuracy_best" if "target_worst_accuracy_best" in sub.columns else "target_worst_accuracy_final"
        vals = [sub[sub["seed"] == s][y_col].values[0] if len(sub[sub["seed"] == s]) > 0 else 0.0 for s in seeds]
        disp = algo_display.get(algo, algo.upper())
        offset = (i - 0.5) * width
        ax2.bar(
            x + offset,
            vals,
            width,
            label=disp,
            color=algo_colors[algo],
            edgecolor="black",
            linewidth=0.8,
            hatch=algo_hatches[algo],
        )

    ax2.set_xlabel("Seed", fontsize=14, labelpad=6)
    ax2.set_ylabel("Held-Out Target Worst Accuracy", fontsize=14, labelpad=6)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Seed {s}" for s in seeds], fontsize=12)
    ax2.set_ylim(0.0, 1.0)
    ax2.tick_params(labelsize=12)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.legend(fontsize=13, loc="upper right", framealpha=0.9)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv_path", default="results_submit/tables/GroupDRO_control/source_target_by_seed.csv")
    parser.add_argument("--output_path", default="results_submit/figures/GroupDRO_control/source_target_by_seed.png")
    args = parser.parse_args()
    plot_source_target(args.csv_path, args.output_path)


if __name__ == "__main__":
    main()
