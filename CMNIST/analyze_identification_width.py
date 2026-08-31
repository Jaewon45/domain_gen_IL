#!/usr/bin/env python3
"""Compute a bounded-risk identification-width diagnostic for Priority 7.

This is a transparent diagnostic, not a substitute for the final theorem. It
assumes missing deployment mass epsilon and missing-domain risks in [0, 1].
"""

import argparse
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from priority7_theory import cvar, risk_profiles, source_prior


def parse_grid(text, cast=float):
    return [cast(token.strip()) for token in text.split(",") if token.strip()]


def summarize_condition(n_domains, sample_size, exponent, missing_fraction, alpha, tradeoff, repetitions, seed):
    ranks = np.arange(n_domains, dtype=float)
    source, missing_count = source_prior(n_domains, exponent, missing_fraction)
    missing_indices = np.arange(n_domains - missing_count, n_domains) if missing_count else np.array([], dtype=int)
    observed_indices = np.arange(0, n_domains - missing_count) if missing_count else np.arange(n_domains)
    deployment = np.full(n_domains, 1.0 / n_domains)
    effective_epsilon = float(missing_count) / float(n_domains)
    profiles = risk_profiles(n_domains, tradeoff)

    possible_scores = []
    for losses in profiles.values():
        lower_losses = losses.copy()
        upper_losses = losses.copy()
        lower_losses[missing_indices] = 0.0
        upper_losses[missing_indices] = 1.0
        possible_scores.extend([
            cvar(lower_losses, deployment, alpha),
            cvar(upper_losses, deployment, alpha),
        ])
    simulated_lower = float(min(possible_scores))
    simulated_upper = float(max(possible_scores))
    simulated_width = simulated_upper - simulated_lower
    analytical_width = min(1.0, effective_epsilon / max(1.0 - float(alpha), 1e-12))

    rng = np.random.default_rng(seed)
    empirical_widths = []
    empirical_scores = []
    for _ in range(int(repetitions)):
        counts = rng.multinomial(int(sample_size), source)
        empirical_weights = counts / counts.sum()
        scores = [cvar(losses, empirical_weights, alpha) for losses in profiles.values()]
        empirical_scores.extend(scores)
        empirical_widths.append(max(scores) - min(scores))

    return {
        "n_domains": n_domains,
        "sample_size": sample_size,
        "exponent": exponent,
        "missing_mass_epsilon": effective_epsilon,
        "requested_missing_tail_fraction": missing_fraction,
        "alpha": alpha,
        "tradeoff": tradeoff,
        "repetitions": repetitions,
        "analytical_delta_id": analytical_width,
        "simulated_lower_cvar": simulated_lower,
        "simulated_upper_cvar": simulated_upper,
        "simulated_interval_width": simulated_width,
        "empirical_interval_width_mean": float(np.mean(empirical_widths)),
        "empirical_interval_width_p05": float(np.quantile(empirical_widths, 0.05)),
        "empirical_interval_width_p95": float(np.quantile(empirical_widths, 0.95)),
        "empirical_cvar_lower_mean": float(np.mean(empirical_scores[0::2])),
        "empirical_cvar_upper_mean": float(np.mean(empirical_scores[1::2])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="../results/cmnist_priority7_theory_v1")
    parser.add_argument("--n_domains", type=int, default=10)
    parser.add_argument("--sample_sizes", default="32,128,512,2048,8192")
    parser.add_argument("--alphas", default="0.5,0.75,0.9")
    parser.add_argument("--missing_tail_fractions", default="0.0,0.1,0.2,0.3")
    parser.add_argument("--exponents", default="0.0,0.5,1.0,2.0")
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tradeoff", type=float, default=0.5)
    parser.add_argument("--plot_alpha", type=float, default=0.9)
    parser.add_argument("--plot_exponent", type=float, default=1.0)
    args = parser.parse_args()

    args.sample_sizes = parse_grid(args.sample_sizes, int)
    args.alphas = parse_grid(args.alphas)
    args.missing_tail_fractions = parse_grid(args.missing_tail_fractions)
    args.exponents = parse_grid(args.exponents)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    conditions = itertools.product(args.sample_sizes, args.alphas, args.missing_tail_fractions, args.exponents)
    rows = [
        summarize_condition(args.n_domains, sample_size, exponent, missing_fraction, alpha, args.tradeoff, args.repetitions, args.seed + index)
        for index, (sample_size, alpha, missing_fraction, exponent) in enumerate(conditions)
    ]
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "identification_width_by_alpha.csv", index=False)

    selected = frame[(frame["alpha"] == args.plot_alpha) & (frame["exponent"] == args.plot_exponent)]
    figure, axis = plt.subplots(figsize=(9, 6))
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    markers = ["o", "s", "^", "d"]
    linestyles = ["-", "--", "-.", ":"]

    for idx, (missing_fraction, group) in enumerate(selected.groupby("requested_missing_tail_fraction")):
        group = group.sort_values("sample_size")
        axis.plot(
            group["sample_size"],
            group["analytical_delta_id"],
            marker=markers[idx % len(markers)],
            linestyle=linestyles[idx % len(linestyles)],
            color=colors[idx % len(colors)],
            linewidth=2.2,
            markersize=7,
            label=f"epsilon={missing_fraction:g}",
        )
        axis.scatter(
            group["sample_size"],
            group["simulated_interval_width"],
            marker="x",
            color=colors[idx % len(colors)],
            s=60,
            zorder=5,
        )
    axis.set_xscale("log")
    axis.set_xlabel("Source Sample Size", fontsize=16, labelpad=8)
    axis.set_ylabel("Identification Interval Width", fontsize=16, labelpad=8)
    axis.tick_params(labelsize=14)
    axis.grid(True, linestyle="--", alpha=0.5)
    axis.legend(fontsize=13, loc="best", framealpha=0.9)
    figure.tight_layout()
    figure.savefig(output_dir / "identification_width_by_alpha.png", dpi=300)
    plt.close(figure)
    print(f"Conditions: {len(frame)}")
    print(f"Wrote {output_dir / 'identification_width_by_alpha.csv'}")
    print(f"Wrote {output_dir / 'identification_width_by_alpha.png'}")


if __name__ == "__main__":
    main()
