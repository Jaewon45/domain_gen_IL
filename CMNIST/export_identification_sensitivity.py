#!/usr/bin/env python3
"""Export CMNIST missing-tail identification tables and sensitivity plots."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plot_identification_phase_diagram import discrete_cvar, load_missing_tail_runs


OBSERVED = (0.1, 0.2, 0.5)
ALPHAS = (0.5, 0.75, 0.9)
MULTIPLIERS = (1.0, 1.5, 2.0, 4.0)


def interval(risks: list[float], alpha: float, missing_low: float, missing_high: float) -> tuple[float, float]:
    weights = [0.25] * 4
    return (
        discrete_cvar(risks + [missing_low], weights, alpha),
        discrete_cvar(risks + [missing_high], weights, alpha),
    )


def export_declared_intervals(runs: pd.DataFrame, output: Path) -> pd.DataFrame:
    rows = []
    for _, run in runs.iterrows():
        risks = [float(run[f"risk_{anchor}"]) for anchor in OBSERVED]
        for alpha in ALPHAS:
            lower, upper = interval(risks, alpha, 0.0, 1.0)
            rows.append({"algorithm": run["algorithm"], "seed": run["seed"], "alpha": alpha,
                         "lower": lower, "upper": upper, "width": upper - lower})
    details = pd.DataFrame(rows).sort_values(["alpha", "algorithm", "seed"])
    details.to_csv(output / "cmnist_missing_tail_intervals_run_level.csv", index=False)
    summary = details.groupby(["algorithm", "alpha"], as_index=False).agg(
        lower_mean=("lower", "mean"), upper_mean=("upper", "mean"),
        width_mean=("width", "mean"), width_std=("width", "std"), runs=("seed", "count"))
    summary.to_csv(output / "cmnist_missing_tail_intervals_summary.csv", index=False)
    lines = ["\\begin{table}[t]", "\\centering", "\\small",
             "\\caption{CMNIST missing-tail plug-in deployment-CVaR identification intervals. Values are means over five final-checkpoint seeds; the declared four-anchor deployment law has missing mass $\\epsilon=1/4$.}",
             "\\label{tab:cmnist-missing-tail-intervals}",
             "\\begin{tabular}{llccc}", "\\toprule", "Method & $\\alpha$ & Lower & Upper & Width \\\\", "\\midrule"]
    for row in summary.itertuples(index=False):
        lines.append(f"{row.algorithm.upper()} & {row.alpha:.2f} & {row.lower_mean:.4f} & {row.upper_mean:.4f} & {row.width_mean:.4f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    (output / "cmnist_missing_tail_intervals.tex").write_text("\n".join(lines), encoding="utf-8")
    return details


def structured_rows(runs: pd.DataFrame) -> pd.DataFrame:
    observed_by_run = []
    for _, run in runs.iterrows():
        risks = [float(run[f"risk_{anchor}"]) for anchor in OBSERVED]
        slopes = [abs(first - second) / abs(first_anchor - second_anchor)
                  for first_anchor, first in zip(OBSERVED, risks)
                  for second_anchor, second in zip(OBSERVED, risks)
                  if first_anchor < second_anchor]
        observed_by_run.append((run, risks, max(slopes)))
    l0 = max(value for _, _, value in observed_by_run)
    rows = []
    for run, risks, _ in observed_by_run:
        for alpha in ALPHAS:
            for multiplier in MULTIPLIERS:
                bound = multiplier * l0
                low = max(0.0, max(risk - bound * abs(0.9 - anchor) for anchor, risk in zip(OBSERVED, risks)))
                high = min(1.0, min(risk + bound * abs(0.9 - anchor) for anchor, risk in zip(OBSERVED, risks)))
                lower, upper = interval(risks, alpha, low, high)
                rows.append({"algorithm": run["algorithm"], "seed": run["seed"], "alpha": alpha,
                             "restriction": f"{multiplier:g}$L_0$", "L": bound,
                             "missing_risk_lower": low, "missing_risk_upper": high,
                             "lower": lower, "upper": upper, "width": upper - lower})
            lower, upper = interval(risks, alpha, 0.0, 1.0)
            rows.append({"algorithm": run["algorithm"], "seed": run["seed"], "alpha": alpha,
                         "restriction": "$\\infty$", "L": float("inf"),
                         "missing_risk_lower": 0.0, "missing_risk_upper": 1.0,
                         "lower": lower, "upper": upper, "width": upper - lower})
    frame = pd.DataFrame(rows)
    frame.attrs["L0"] = l0
    return frame


def export_structured_sensitivity(runs: pd.DataFrame, output: Path) -> None:
    details = structured_rows(runs)
    details.to_csv(output / "cmnist_structured_completion_run_level.csv", index=False)
    summary = details.groupby(["alpha", "restriction"], sort=False, as_index=False).agg(
        mean_width=("width", "mean"), std_width=("width", "std"), runs=("seed", "count"))
    summary.to_csv(output / "cmnist_structured_completion_summary.csv", index=False)
    restriction_order = ["1$L_0$", "1.5$L_0$", "2$L_0$", "4$L_0$", "$\\infty$"]
    figure, axis = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    for alpha in ALPHAS:
        subset = summary[summary.alpha == alpha].set_index("restriction").reindex(restriction_order)
        axis.plot(restriction_order, subset.mean_width, marker="o", linewidth=2, label=rf"$\alpha={alpha:g}$")
    axis.set_xlabel("Missing-anchor Lipschitz restriction")
    axis.set_ylabel("Mean identification-interval width")
    axis.set_ylim(bottom=0)
    axis.legend(title="Deployment CVaR")
    axis.set_title("CMNIST structured-completion sensitivity")
    figure.savefig(output / "cmnist_structured_completion_sensitivity.png", dpi=300, bbox_inches="tight")
    figure.savefig(output / "cmnist_structured_completion_sensitivity.pdf", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("results/CMNIST"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/reports/E3b/CMNIST"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = load_missing_tail_runs(args.input_dir)
    export_declared_intervals(runs, args.output_dir)
    export_structured_sensitivity(runs, args.output_dir)
    print(f"Exported identification sensitivity for {len(runs)} runs to {args.output_dir}")


if __name__ == "__main__":
    main()
