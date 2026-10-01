#!/usr/bin/env python3
"""Export run-level plug-in identification audits from merged E3b results."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import pandas as pd

ALPHAS = (0.50, 0.75, 0.90)
CMNIST_ANCHORS = (0.1, 0.2, 0.5, 0.9)
CONDITION_LABELS = {
    "balanced_visible": "Balanced",
    "long_tail_visible": "Long-tail",
    "near_missing_tail": "Near-missing",
    "missing_tail": "Missing",
}


def cvar(values, weights, alpha):
    tail_mass = 1.0 - float(alpha)
    used = 0.0
    total = 0.0
    for value, weight in sorted(zip(values, weights), reverse=True):
        take = min(float(weight), tail_mass - used)
        total += float(value) * take
        used += take
        if used >= tail_mass - 1e-12:
            break
    return total / tail_mass


def audit_condition(frame, condition):
    x = frame[
        frame["condition"].eq(condition)
        & frame["domain_id"].isin(CMNIST_ANCHORS)
    ].copy()
    methods = sorted(x["algorithm"].unique())
    seeds = sorted(x["seed"].unique())
    domains = sorted(x["domain_id"].unique())
    assert len(methods) == 7, (condition, methods)
    assert len(seeds) == 5, (condition, seeds)
    assert domains == list(CMNIST_ANCHORS), (condition, domains)
    assert all((x.groupby(["algorithm", "seed"]).size() == 4))
    assert x.test_accuracy.between(0.0, 1.0).all()

    observed = set(x.loc[x.train_count > 0, "domain_id"])
    deployment_weights = {domain: 1.0 / len(CMNIST_ANCHORS) for domain in CMNIST_ANCHORS}
    epsilon = sum(weight for domain, weight in deployment_weights.items() if domain not in observed)
    records = []
    for _, group in x.groupby(["algorithm", "seed"]):
        assert set(group.loc[group.train_count > 0, "domain_id"]) == observed
        observed_rows = group[group.train_count > 0].sort_values("domain_id")
        values = (1.0 - observed_rows.test_accuracy.to_numpy(float)).tolist()
        observed_pi = [deployment_weights[domain] for domain in observed_rows.domain_id]
        assert abs(sum(observed_pi) - (1.0 - epsilon)) < 1e-12
        method = str(group.algorithm.iloc[0])
        seed = int(group.seed.iloc[0])
        for alpha in ALPHAS:
            lower = cvar(values + [0.0], observed_pi + [epsilon], alpha)
            upper = cvar(values + [1.0], observed_pi + [epsilon], alpha)
            records.append({
                "condition": condition,
                "algorithm": method,
                "seed": seed,
                "alpha": alpha,
                "lower": lower,
                "upper": upper,
                "width": upper - lower,
            })

    endpoints = pd.DataFrame(records)
    summary = []
    for alpha in ALPHAS:
        current = endpoints[endpoints.alpha.eq(alpha)].set_index(["algorithm", "seed"])
        certified = 0
        for seed in seeds:
            for first, second in itertools.combinations(methods, 2):
                left = current.loc[(first, seed)]
                right = current.loc[(second, seed)]
                certified += int(left.upper < right.lower or right.upper < left.lower)
        summary.append({
            "condition": condition,
            "alpha": alpha,
            "epsilon": epsilon,
            "observed_domains": ",".join(map(str, sorted(observed))),
            "unobserved_domains": ",".join(map(str, sorted(set(domains) - observed))),
            "mean_width": float(current.width.mean()),
            "certified_pairs": certified,
            "total_pairs": len(seeds) * len(list(itertools.combinations(methods, 2))),
        })
    return endpoints, pd.DataFrame(summary)


def tex_table(summary, condition, output):
    row = summary[summary.condition.eq(condition)].sort_values("alpha")
    label = CONDITION_LABELS[condition]
    epsilon = float(row.epsilon.iloc[0])
    lines = [
        "% Generated from raw_results_7methods.csv using run-level 0-1 losses and deployment weights.",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{\\textbf{Empirical plug-in identification audit under "+label.lower()+" source support.} For each method and seed, we compute the sharp plug-in CVaR interval under the observed-support completion model. ``Width'' is the interval width, averaged over all methods and five CMNIST seeds. A pair is certified when the two run-level intervals are disjoint; pairs with overlapping intervals have no identified deployment-CVaR ordering under the stated completion model. Results use the \\emph{"+label+"} support condition with missing mass $\\epsilon="+f"{epsilon:.4f}"+"$.}",
        "\\label{tab:identification-audit-"+condition.replace("_", "-")+"}",
        "\\begin{tabular}{@{}cccc@{}}",
        "\\toprule",
        "$\\alpha$ & Mean width & Certified pairs & Total pairs \\\\",
        "\\midrule",
    ]
    for _, item in row.iterrows():
        lines.append(f"${item.alpha:.2f}$ & {item.mean_width:.4f} & {int(item.certified_pairs)} / ${int(item.total_pairs)}$ & ${int(item.total_pairs)}$ \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    output.write_text("\n".join(lines), encoding="utf-8")


def supplement_tex(summary, output):
    lines = [
        "% Generated from raw_results_7methods.csv using run-level 0-1 losses and deployment weights.",
        "\\begin{table}[t]", "\\centering", "\\small",
        "\\caption{Supplementary run-level plug-in identification audit across E3b support conditions. Width is averaged over seven methods and five seeds; certification counts use disjoint run-level intervals.}",
        "\\label{tab:identification-audit-all}",
        "\\begin{tabular}{@{}lccccc@{}}", "\\toprule",
        "Condition & $\\alpha$ & $\\epsilon$ & Mean width & Certified pairs & Total pairs \\\\", "\\midrule",
    ]
    for _, item in summary.sort_values(["condition", "alpha"]).iterrows():
        lines.append(f"{CONDITION_LABELS[item.condition]} & ${item.alpha:.2f}$ & {item.epsilon:.4f} & {item.mean_width:.4f} & {int(item.certified_pairs)} & {int(item.total_pairs)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    output.write_text("\n".join(lines), encoding="utf-8")


def combined_tex(summary, output):
    lines = [
        "% Generated from raw_results_7methods.csv using run-level 0--1 losses",
        "% and deployment weights.",
        "\\begin{table}[t]", "\\centering", "\\small",
        "\\caption{\\textbf{Empirical plug-in identification audit under source support removal.} For each method--seed run, we compute the sharp plug-in deployment-CVaR interval under the observed-support completion model with $0$--$1$ loss ($M=1$). Width is averaged over seven methods and five CMNIST seeds. A method pair is certified only when its run-level intervals are disjoint. In both conditions, $\\epsilon>1-\\alpha$ for all reported $\\alpha$; consequently, every upper endpoint equals $1$, and no pairwise ordering is certified.}",
        "\\label{tab:identification-audit}",
        "\\begin{tabular}{@{}lccc@{}}", "\\toprule",
        "Support condition & $\\alpha$ & Mean width & Certified pairs \\\\", "\\midrule",
    ]
    for condition, epsilon_label in (("missing_tail", "8/11"), ("near_missing_tail", "7/11")):
        current = summary[summary.condition.eq(condition)].sort_values("alpha")
        lines.append("Missing" if condition == "missing_tail" else "Near-missing")
        for index, (_, item) in enumerate(current.iterrows()):
            if index == 1:
                lines.append(f"($\\epsilon={epsilon_label}$)")
            lines.append(f"  & ${item.alpha:.2f}$ & ${item.mean_width:.4f}$ & $0/{int(item.total_pairs)}$ \\\\")
        if condition == "missing_tail":
            lines.append("\\addlinespace")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    output.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_csv")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.raw_csv)
    all_endpoints = []
    all_summary = []
    for condition in CONDITION_LABELS:
        endpoints, summary = audit_condition(frame, condition)
        all_endpoints.append(endpoints)
        all_summary.append(summary)
    endpoints = pd.concat(all_endpoints, ignore_index=True)
    summary = pd.concat(all_summary, ignore_index=True)
    endpoints.to_csv(output_dir / "identification_audit_run_level.csv", index=False)
    summary.to_csv(output_dir / "identification_audit_summary.csv", index=False)
    tex_table(summary, "missing_tail", output_dir / "identification_audit_missing.tex")
    tex_table(summary, "near_missing_tail", output_dir / "identification_audit_near_missing.tex")
    combined_tex(summary, output_dir / "identification_audit.tex")
    supplement_tex(summary, output_dir / "identification_audit_all_conditions.tex")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
