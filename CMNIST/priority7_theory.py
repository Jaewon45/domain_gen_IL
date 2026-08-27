#!/usr/bin/env python3
"""Theory-aligned CMNIST support simulation.

This experiment is deliberately independent of neural-network training. It compares
risk-profile rankings under a uniform deployment prior and sampled long-tailed
source support, including forced missing tail domains.
"""

import argparse
import itertools
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROFILE_NAMES = ("head_favored", "tail_favored")


def parse_grid(text, cast=float):
    return [cast(token.strip()) for token in text.split(",") if token.strip()]


def cvar(losses, weights, alpha):
    losses = np.asarray(losses, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if weights.sum() <= 0:
        return float("nan")
    weights = weights / weights.sum()
    order = np.argsort(losses)
    cumulative = np.cumsum(weights[order])
    index = min(int(np.searchsorted(cumulative, alpha, side="left")), len(losses) - 1)
    threshold = losses[order][index]
    tail = losses >= threshold
    return float(np.sum(losses[tail] * weights[tail]) / np.sum(weights[tail]))


def deployment_prior(n_domains):
    return np.full(n_domains, 1.0 / n_domains)


def source_prior(n_domains, exponent, missing_tail_fraction):
    ranks = np.arange(1, n_domains + 1, dtype=float)
    probabilities = ranks ** (-float(exponent))
    missing_count = int(np.floor(n_domains * float(missing_tail_fraction)))
    if missing_count >= n_domains:
        missing_count = n_domains - 1
    if missing_count:
        probabilities[-missing_count:] = 0.0
    probabilities /= probabilities.sum()
    return probabilities, missing_count


def risk_profiles(n_domains, tradeoff):
    position = np.linspace(0.0, 1.0, n_domains)
    head_favored = 0.20 + float(tradeoff) * position
    tail_favored = 0.15 + float(tradeoff) * (1.0 - position)
    return {"head_favored": head_favored, "tail_favored": tail_favored}


def one_trial(n_domains, sample_size, exponent, missing_tail_fraction, alpha, tradeoff, rng):
    deployment = deployment_prior(n_domains)
    source, missing_count = source_prior(n_domains, exponent, missing_tail_fraction)
    counts = rng.multinomial(int(sample_size), source)
    empirical = counts / counts.sum()
    profiles = risk_profiles(n_domains, tradeoff)

    deployment_scores = {
        name: cvar(losses, deployment, alpha)
        for name, losses in profiles.items()
    }
    empirical_scores = {
        name: cvar(losses, empirical, alpha)
        for name, losses in profiles.items()
    }
    deployment_winner = min(deployment_scores, key=deployment_scores.get)
    empirical_winner = min(empirical_scores, key=empirical_scores.get)
    return {
        "deployment_winner": deployment_winner,
        "empirical_winner": empirical_winner,
        "reversal": int(empirical_winner != deployment_winner),
        "deployment_gap": abs(deployment_scores[PROFILE_NAMES[0]] - deployment_scores[PROFILE_NAMES[1]]),
        "empirical_gap": abs(empirical_scores[PROFILE_NAMES[0]] - empirical_scores[PROFILE_NAMES[1]]),
        "missing_domains": missing_count,
        "observed_tail": int(counts[-1] > 0),
        "empirical_tail_weight": float(empirical[-1]),
    }


def run_condition(condition, repetitions, seed):
    rng = np.random.default_rng(seed)
    trials = [
        one_trial(rng=rng, **condition)
        for _ in range(int(repetitions))
    ]
    frame = pd.DataFrame(trials)
    return {
        **condition,
        "repetitions": int(repetitions),
        "reversal_probability": float(frame["reversal"].mean()),
        "observed_tail_probability": float(frame["observed_tail"].mean()),
        "mean_empirical_tail_weight": float(frame["empirical_tail_weight"].mean()),
        "mean_deployment_gap": float(frame["deployment_gap"].mean()),
        "mean_empirical_gap": float(frame["empirical_gap"].mean()),
        "head_favored_empirical_wins": int((frame["empirical_winner"] == "head_favored").sum()),
        "tail_favored_empirical_wins": int((frame["empirical_winner"] == "tail_favored").sum()),
    }


def build_conditions(args):
    conditions = []
    for alpha, sample_size, missing_fraction, exponent in itertools.product(
        args.alphas, args.sample_sizes, args.missing_tail_fractions, args.exponents
    ):
        conditions.append({
            "n_domains": args.n_domains,
            "sample_size": int(sample_size),
            "exponent": float(exponent),
            "missing_tail_fraction": float(missing_fraction),
            "alpha": float(alpha),
            "tradeoff": float(args.tradeoff),
        })
    return conditions


def plot_heatmap(summary, args, output_dir):
    target = summary[
        (summary["alpha"] == args.heatmap_alpha)
        & (summary["exponent"] == args.heatmap_exponent)
    ]
    if target.empty:
        return None
    pivot = target.pivot(index="missing_tail_fraction", columns="sample_size", values="reversal_probability")
    if pivot.empty:
        return None
    figure, axis = plt.subplots(figsize=(8, 5))
    image = axis.imshow(pivot.to_numpy(), aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
    axis.set_xticks(range(len(pivot.columns)), [str(value) for value in pivot.columns])
    axis.set_yticks(range(len(pivot.index)), [str(value) for value in pivot.index])
    axis.set_xlabel("Sample size")
    axis.set_ylabel("Missing-tail fraction")
    axis.set_title("Priority 7 ranking-reversal probability")
    figure.colorbar(image, ax=axis, label="Reversal probability")
    figure.tight_layout()
    path = output_dir / "ranking_reversal_heatmap.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def plot_sample_size(summary, args, output_dir):
    target = summary[
        (summary["alpha"] == args.heatmap_alpha)
        & (summary["exponent"] == args.heatmap_exponent)
        & (summary["missing_tail_fraction"] == 0.0)
    ]
    if target.empty:
        return None
    figure, axis = plt.subplots(figsize=(8, 5))
    for missing_fraction, group in summary[
        (summary["alpha"] == args.heatmap_alpha)
        & (summary["exponent"] == args.heatmap_exponent)
    ].groupby("missing_tail_fraction"):
        curve = group.groupby("sample_size")["reversal_probability"].mean().reset_index()
        axis.plot(curve["sample_size"], curve["reversal_probability"], marker="o", label=f"missing={missing_fraction:g}")
    axis.set_xscale("log")
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Source sample size")
    axis.set_ylabel("Ranking-reversal probability")
    axis.set_title("Priority 7 evidence versus source information")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    path = output_dir / "ranking_reversal_by_sample_size.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


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
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--heatmap_alpha", type=float, default=0.9)
    parser.add_argument("--heatmap_exponent", type=float, default=1.0)
    parser.add_argument("--heatmap_sample_size", type=int, default=128)
    args = parser.parse_args()

    if args.smoke:
        args.sample_sizes = "32,128"
        args.alphas = "0.5,0.9"
        args.missing_tail_fractions = "0.0,0.2"
        args.exponents = "0.0,1.0"
        args.repetitions = min(args.repetitions, 50)

    args.sample_sizes = parse_grid(args.sample_sizes, int)
    args.alphas = parse_grid(args.alphas, float)
    args.missing_tail_fractions = parse_grid(args.missing_tail_fractions, float)
    args.exponents = parse_grid(args.exponents, float)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    conditions = build_conditions(args)
    rows = [run_condition(condition, args.repetitions, args.seed + index) for index, condition in enumerate(conditions)]
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "ranking_reversal_summary.csv", index=False)

    trial = dict(conditions[0])
    trial_rows = []
    rng = np.random.default_rng(args.seed)
    for repetition in range(min(args.repetitions, 1000)):
        trial_rows.append({"repetition": repetition, **one_trial(rng=rng, **trial)})
    pd.DataFrame(trial_rows).to_csv(output_dir / "trial_diagnostics.csv", index=False)

    generated = [plot_heatmap(summary, args, output_dir), plot_sample_size(summary, args, output_dir)]
    print(f"Conditions: {len(conditions)}")
    print(f"Repetitions per condition: {args.repetitions}")
    print(f"Wrote {output_dir / 'ranking_reversal_summary.csv'}")
    for path in generated:
        if path:
            print(f"Wrote {path}")


if __name__ == "__main__":
    main()
