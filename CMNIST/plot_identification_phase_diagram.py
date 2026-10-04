#!/usr/bin/env python3
"""Plot CMNIST post-training identification sensitivity under missing support.

For each final-checkpoint run in the ``missing_tail`` condition, the script
uses only the observed anchor risks at 0.1, 0.2, and 0.5.  It computes the
sharp unrestricted plug-in CVaR interval obtained by completing the missing
0.9 anchor with risk zero or one.  The left panel averages interval width over
all retained methods and seeds; the right panel gives the fraction of
same-seed unordered method pairs with strictly separated intervals.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OBSERVED_ANCHORS = (0.1, 0.2, 0.5)
DEFAULT_ALPHAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95)
DEFAULT_EPSILONS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)


def parse_grid(text: str) -> tuple[float, ...]:
    values = tuple(float(token.strip()) for token in text.split(",") if token.strip())
    if not values or any(value < 0.0 or value >= 1.0 for value in values):
        raise ValueError("Grid values must be non-empty and lie in [0, 1).")
    return values


def discrete_cvar(values: Iterable[float], weights: Iterable[float], alpha: float) -> float:
    """Exact upper-tail CVaR for a finite distribution, including boundary mass."""
    pairs = sorted(zip(values, weights), key=lambda item: item[0], reverse=True)
    tail_mass = 1.0 - alpha
    used_mass = 0.0
    total = 0.0
    for value, weight in pairs:
        take = min(float(weight), tail_mass - used_mass)
        if take > 0.0:
            total += float(value) * take
            used_mass += take
        if used_mass >= tail_mass - 1e-12:
            break
    if not np.isclose(used_mass, tail_mass, rtol=0.0, atol=1e-10):
        raise ValueError("CVaR weights did not cover the requested tail mass.")
    return total / tail_mass


def load_missing_tail_runs(root: Path) -> pd.DataFrame:
    """Load one final-risk profile per method and seed from CMNIST JSONL output."""
    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            args = record.get("args", {})
            condition = record.get("tail_support_condition") or args.get("tail_support_condition")
            if condition != "missing_tail":
                continue
            try:
                risks = {
                    anchor: 1.0 - float(record[f"{anchor}_acc_final"])
                    for anchor in OBSERVED_ANCHORS
                }
                row = {
                    "algorithm": str(record["algorithm"]),
                    "seed": int(record["seed"]),
                    "source_file": str(path),
                    **{f"risk_{anchor}": risk for anchor, risk in risks.items()},
                }
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid final-accuracy record in {path}") from error
            if any(risk < 0.0 or risk > 1.0 for risk in risks.values()):
                raise ValueError(f"Observed 0--1 risks outside [0,1] in {path}")
            rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise FileNotFoundError(f"No missing_tail final-checkpoint records found under {root}")
    duplicates = frame.duplicated(["algorithm", "seed"], keep=False)
    if duplicates.any():
        duplicate_rows = frame.loc[duplicates, ["algorithm", "seed", "source_file"]]
        raise ValueError("Expected one missing_tail result per method and seed; found duplicates:\n"
                         + duplicate_rows.to_string(index=False))
    methods = sorted(frame["algorithm"].unique())
    seeds = sorted(frame["seed"].unique())
    expected = len(methods) * len(seeds)
    if len(frame) != expected:
        raise ValueError("CMNIST missing-tail matrix is incomplete.")
    return frame.sort_values(["algorithm", "seed"]).reset_index(drop=True)


def summarize_grid(runs: pd.DataFrame, alphas: tuple[float, ...], epsilons: tuple[float, ...]) -> pd.DataFrame:
    methods = sorted(runs["algorithm"].unique())
    seeds = sorted(runs["seed"].unique())
    rows: list[dict[str, float | int]] = []
    for alpha, epsilon in itertools.product(alphas, epsilons):
        endpoints: dict[tuple[str, int], tuple[float, float]] = {}
        for _, run in runs.iterrows():
            observed_risks = [float(run[f"risk_{anchor}"]) for anchor in OBSERVED_ANCHORS]
            observed_weights = [(1.0 - epsilon) / len(OBSERVED_ANCHORS)] * len(OBSERVED_ANCHORS)
            lower = discrete_cvar(observed_risks + [0.0], observed_weights + [epsilon], alpha)
            upper = discrete_cvar(observed_risks + [1.0], observed_weights + [epsilon], alpha)
            endpoints[(str(run["algorithm"]), int(run["seed"]))] = (lower, upper)
        widths = [upper - lower for lower, upper in endpoints.values()]
        certified = 0
        total_pairs = 0
        for seed in seeds:
            for first, second in itertools.combinations(methods, 2):
                first_lower, first_upper = endpoints[(first, seed)]
                second_lower, second_upper = endpoints[(second, seed)]
                certified += int(first_upper < second_lower or second_upper < first_lower)
                total_pairs += 1
        rows.append({
            "alpha": alpha,
            "epsilon": epsilon,
            "mean_identification_width": float(np.mean(widths)),
            "certified_pairs": certified,
            "total_pairs": total_pairs,
            "certified_pair_fraction": certified / total_pairs,
            "methods": len(methods),
            "seeds": len(seeds),
        })
    return pd.DataFrame(rows)


def grid_edges(values: tuple[float, ...]) -> np.ndarray:
    """Create visually stable heatmap edges for an irregularly spaced grid."""
    points = np.asarray(values, dtype=float)
    if len(points) == 1:
        return np.array([points[0] - 0.025, points[0] + 0.025])
    midpoints = (points[:-1] + points[1:]) / 2.0
    return np.concatenate(([points[0] - (midpoints[0] - points[0])], midpoints,
                           [points[-1] + (points[-1] - midpoints[-1])]))


def draw_phase_diagram(summary: pd.DataFrame, alphas: tuple[float, ...], epsilons: tuple[float, ...], output: Path) -> None:
    width = summary.pivot(index="alpha", columns="epsilon", values="mean_identification_width").loc[list(alphas), list(epsilons)]
    fraction = summary.pivot(index="alpha", columns="epsilon", values="certified_pair_fraction").loc[list(alphas), list(epsilons)]
    figure, axes = plt.subplots(1, 2, figsize=(13.4, 5.3), constrained_layout=True, sharey=True)
    panels = (
        (axes[0], width.to_numpy(), "Mean identification width", "viridis", 0.0, 1.0),
        (axes[1], fraction.to_numpy(), "Strictly certified pair fraction", "magma", 0.0, 1.0),
    )
    epsilon_edges = grid_edges(epsilons)
    alpha_edges = grid_edges(alphas)
    for index, (axis, matrix, label, colormap, lower, upper) in enumerate(panels):
        image = axis.pcolormesh(epsilon_edges, alpha_edges, matrix, shading="flat", cmap=colormap,
                                vmin=lower, vmax=upper)
        colorbar = figure.colorbar(image, ax=axis, pad=0.02)
        colorbar.set_label(label, fontsize=12)
        axis.plot(epsilons, [1.0 - epsilon for epsilon in epsilons], "w--", linewidth=1.8,
                  label=r"$\alpha=1-\epsilon$")
        axis.axvline(0.25, color="white", linestyle=":", linewidth=2.0,
                    label=r"declared $\epsilon=1/4$")
        axis.set_xlim(min(epsilon_edges), max(epsilon_edges))
        axis.set_ylim(min(alpha_edges), max(alpha_edges))
        axis.set_xlabel(r"Missing deployment mass $\epsilon$", fontsize=13)
        axis.set_title("(a) Mean interval width" if index == 0 else "(b) Strict ranking certificate",
                       loc="left", fontsize=14, fontweight="bold")
        axis.set_xticks(epsilons)
        axis.tick_params(axis="x", labelrotation=45)
        axis.legend(loc="lower left", fontsize=10, framealpha=0.92)
    axes[0].set_ylabel(r"Deployment-CVaR level $\alpha$", fontsize=13)
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("results/CMNIST"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/reports/E3b/CMNIST"))
    parser.add_argument("--alphas", default=",".join(map(str, DEFAULT_ALPHAS)))
    parser.add_argument("--epsilons", default=",".join(map(str, DEFAULT_EPSILONS)))
    args = parser.parse_args()
    alphas = parse_grid(args.alphas)
    epsilons = parse_grid(args.epsilons)
    runs = load_missing_tail_runs(args.input_dir)
    summary = summarize_grid(runs, alphas, epsilons)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "cmnist_identification_phase_diagram_summary.csv"
    runs.to_csv(args.output_dir / "cmnist_identification_phase_diagram_runs.csv", index=False)
    summary.to_csv(summary_path, index=False)
    output = args.output_dir / "cmnist_identification_phase_diagram"
    draw_phase_diagram(summary, alphas, epsilons, output)
    print(f"Runs: {len(runs)} ({runs.algorithm.nunique()} methods x {runs.seed.nunique()} seeds)")
    print(f"Wrote {summary_path}")
    print(f"Wrote {output.with_suffix('.png')}")
    print(f"Wrote {output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
