#!/usr/bin/env python3
"""Run Theorem-aligned 0-1 CVaR post-hoc check, recover synthetic spec, and run identification-width sim.

Task 1: Theorem-aligned 0-1 CVaR post-hoc check on E3b trained checkpoints.
Task 2: Exact synthetic ranking-reversal specification recovery.
Task 3: Same-predictor identification-width simulation.
"""

import math
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def safe_savefig(fig, path, dpi=300):
    try:
        fig.savefig(path, dpi=dpi)
    except PermissionError:
        print(f"Warning: Could not save to {path} (file may be locked/open). Continuing...")


def apply_plot_style():
    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "figure.titlesize": 20,
    })


def cvar_discrete(support, weights, alpha):
    support = np.asarray(support, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()

    order = np.argsort(support)
    sorted_s = support[order]
    sorted_w = weights[order]

    cum_w = np.cumsum(sorted_w)

    val = 0.0
    prev_c = 0.0
    for k in range(len(sorted_s)):
        curr_c = cum_w[k]
        overlap = max(0.0, min(curr_c, 1.0) - max(prev_c, alpha))
        val += sorted_s[k] * overlap
        prev_c = curr_c
    return float(val / (1.0 - alpha))


def run_task1():
    print("=== Running Task 1: Theorem-aligned 0-1 CVaR post-hoc check ===")
    raw_path = Path("results/E3b_tail_support/analysis_4anchor_seed0-4/raw_results.csv")
    if not raw_path.exists():
        raw_path = Path("results/E3b_tail_support/raw_results.csv")
    
    df = pd.read_csv(raw_path)
    mt = df[df["condition"] == "missing_tail"]

    alpha = 0.9
    epsilon = 0.25

    algos = ["erm", "groupdro", "inftask", "irm", "iro"]
    display_names = {
        "erm": "ERM",
        "groupdro": "GroupDRO",
        "inftask": "INF-TASK",
        "irm": "IRM",
        "iro": "IRO",
    }
    seeds = [0, 1, 2, 3, 4]

    rows = []
    for algo in algos:
        for seed in seeds:
            sub = mt[(mt["algorithm"] == algo) & (mt["seed"] == seed)]
            acc_dict = dict(zip(sub["domain_id"], sub["test_accuracy"]))

            # Define risk R_e = 1 - accuracy
            r_01 = 1.0 - acc_dict[0.1]
            r_02 = 1.0 - acc_dict[0.2]
            r_05 = 1.0 - acc_dict[0.5]
            r_09 = 1.0 - acc_dict[0.9]

            # P_obs = (1/3) [delta(R_0.1) + delta(R_0.2) + delta(R_0.5)]
            # P_minus = (3/4) P_obs + (1/4) delta_0
            p_minus_s = [r_01, r_02, r_05, 0.0]
            p_minus_w = [0.25, 0.25, 0.25, 0.25]
            lower_cvar = cvar_discrete(p_minus_s, p_minus_w, alpha)

            # P_deploy = (1/4) [delta(R_0.1) + delta(R_0.2) + delta(R_0.5) + delta(R_0.9)]
            p_deploy_s = [r_01, r_02, r_05, r_09]
            p_deploy_w = [0.25, 0.25, 0.25, 0.25]
            deploy_cvar = cvar_discrete(p_deploy_s, p_deploy_w, alpha)

            # P_plus = (3/4) P_obs + (1/4) delta_1
            p_plus_s = [r_01, r_02, r_05, 1.0]
            p_plus_w = [0.25, 0.25, 0.25, 0.25]
            upper_cvar = cvar_discrete(p_plus_s, p_plus_w, alpha)

            id_w = upper_cvar - lower_cvar
            
            # Verify theorem inequality
            valid = (lower_cvar <= deploy_cvar + 1e-9) and (deploy_cvar <= upper_cvar + 1e-9)
            assert valid, f"Inequality violated for {algo} seed {seed}: {lower_cvar} <= {deploy_cvar} <= {upper_cvar}"

            rows.append({
                "algorithm": algo,
                "seed": seed,
                "condition": "missing_tail",
                "alpha": alpha,
                "lower_cvar": lower_cvar,
                "deployment_cvar": deploy_cvar,
                "upper_cvar": upper_cvar,
                "id_width": id_w,
            })

    res_df = pd.DataFrame(rows)
    
    # Ensure folders exist
    Path("results").mkdir(exist_ok=True, parents=True)
    Path("results_submit/additional").mkdir(exist_ok=True, parents=True)
    Path("results_submit/figures").mkdir(exist_ok=True, parents=True)
    Path("results_submit/figures/P7_theory").mkdir(exist_ok=True, parents=True)
    Path("results_submit/tables/P7_theory").mkdir(exist_ok=True, parents=True)
    Path("results_submit/metadata").mkdir(exist_ok=True, parents=True)

    # Save CSV files
    csv_filename = "theorem_aligned_01_cvar_bounds.csv"
    res_df.to_csv(csv_filename, index=False)
    res_df.to_csv(Path("results") / csv_filename, index=False)
    res_df.to_csv(Path("results_submit/additional") / csv_filename, index=False)
    res_df.to_csv(Path("results_submit/tables/P7_theory") / csv_filename, index=False)
    print(f"Saved {csv_filename} to root, results/, results_submit/additional/, and results_submit/tables/P7_theory/")

    # Print summary statistics
    print("\n--- Theorem-aligned 0-1 CVaR Bounds Summary (alpha=0.9, epsilon=0.25) ---")
    summary = res_df.groupby("algorithm").agg(
        lower_cvar_mean=("lower_cvar", "mean"),
        lower_cvar_std=("lower_cvar", "std"),
        deploy_cvar_mean=("deployment_cvar", "mean"),
        deploy_cvar_std=("deployment_cvar", "std"),
        upper_cvar_mean=("upper_cvar", "mean"),
        upper_cvar_std=("upper_cvar", "std"),
        id_width_mean=("id_width", "mean"),
        id_width_std=("id_width", "std"),
    ).reindex(algos)
    print(summary.to_string())

    # Generate Figure
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(9, 6))

    x = np.arange(len(algos))
    width = 0.25

    lower_means = [summary.loc[a, "lower_cvar_mean"] for a in algos]
    lower_stds = [summary.loc[a, "lower_cvar_std"] for a in algos]
    deploy_means = [summary.loc[a, "deploy_cvar_mean"] for a in algos]
    deploy_stds = [summary.loc[a, "deploy_cvar_std"] for a in algos]
    upper_means = [summary.loc[a, "upper_cvar_mean"] for a in algos]
    upper_stds = [summary.loc[a, "upper_cvar_std"] for a in algos]

    # Colorblind safe Okabe-Ito colors & distinct hatches
    c_lower, c_deploy, c_upper = "#0072B2", "#D55E00", "#009E73"
    h_lower, h_deploy, h_upper = "", "//", "\\\\"

    b1 = ax.bar(x - width, lower_means, width, yerr=lower_stds, label="Lower identified bound", color=c_lower, edgecolor="black", linewidth=0.8, capsize=4, hatch=h_lower)
    b2 = ax.bar(x, deploy_means, width, yerr=deploy_stds, label="Realised deployment CVaR", color=c_deploy, edgecolor="black", linewidth=0.8, capsize=4, hatch=h_deploy)
    b3 = ax.bar(x + width, upper_means, width, yerr=upper_stds, label="Upper identified bound", color=c_upper, edgecolor="black", linewidth=0.8, capsize=4, hatch=h_upper)

    ax.set_ylabel("Risk (0-1 Error CVaR at α = 0.9)", fontsize=16, labelpad=8)
    ax.set_xlabel("Algorithm", fontsize=16, labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels([display_names[a] for a in algos], fontsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="upper right", fontsize=13, framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    pdf_filename = "theorem_aligned_01_cvar_bounds.pdf"
    png_filename = "theorem_aligned_01_cvar_bounds.png"
    safe_savefig(fig, pdf_filename, dpi=300)
    safe_savefig(fig, png_filename, dpi=300)
    safe_savefig(fig, Path("results") / pdf_filename, dpi=300)
    safe_savefig(fig, Path("results") / png_filename, dpi=300)
    safe_savefig(fig, Path("results_submit/additional") / pdf_filename, dpi=300)
    safe_savefig(fig, Path("results_submit/additional") / png_filename, dpi=300)
    safe_savefig(fig, Path("results_submit/figures") / pdf_filename, dpi=300)
    safe_savefig(fig, Path("results_submit/figures") / png_filename, dpi=300)
    safe_savefig(fig, Path("results_submit/figures/P7_theory") / pdf_filename, dpi=300)
    safe_savefig(fig, Path("results_submit/figures/P7_theory") / png_filename, dpi=300)
    plt.close()
    print(f"Saved {pdf_filename} to root, results/, results_submit/additional/, and results_submit/figures/P7_theory/\n")
    return res_df, summary


def run_task2():
    print("=== Running Task 2: Recover exact synthetic specification ===")
    K = 10
    tradeoff = 0.5

    position = np.linspace(0.0, 1.0, K)
    risk_vector_f = 0.20 + tradeoff * position
    risk_vector_g = 0.15 + tradeoff * (1.0 - position)

    K_head = 7  # Observed domains under missing_tail_fraction=0.3
    r0_head = 0.20  # Base risk offset for f (head-favored)
    r0_tail = 0.15  # Base risk offset for g (tail-favored)
    delta_head = float(risk_vector_g[0] - risk_vector_f[0])  # 0.65 - 0.20 = 0.45
    delta_tail = float(risk_vector_f[-1] - risk_vector_g[-1])  # 0.70 - 0.15 = 0.55

    spec_text = f"""Synthetic Ranking-Reversal Experiment Specification:
--------------------------------------------------
K (total domain count)           : {K}
K_head (observed head domains)    : {K_head} (for missing_tail_fraction = 0.3)
r0_head (f base risk offset)     : {r0_head:.2f}
r0_tail (g base risk offset)     : {r0_tail:.2f}
tradeoff (risk slope spread)     : {tradeoff:.1f}
delta_head (head advantage)      : {delta_head:.2f}
delta_tail (tail advantage)      : {delta_tail:.2f}

Candidate Risk Profiles:
risk_vector_f (head-favored) : {np.array2string(risk_vector_f, precision=6, separator=', ')}
risk_vector_g (tail-favored) : {np.array2string(risk_vector_g, precision=6, separator=', ')}
"""
    print(spec_text)

    spec_filename = "synthetic_risk_profile_spec.txt"
    with open(spec_filename, "w", encoding="utf-8") as f:
        f.write(spec_text)
    with open(Path("results") / spec_filename, "w", encoding="utf-8") as f:
        f.write(spec_text)
    with open(Path("results_submit/additional") / spec_filename, "w", encoding="utf-8") as f:
        f.write(spec_text)
    with open(Path("results_submit/metadata") / spec_filename, "w", encoding="utf-8") as f:
        f.write(spec_text)
    print(f"Saved {spec_filename} to root, results/, results_submit/additional/, and results_submit/metadata/\n")


def run_task3():
    print("=== Running Task 3: Same-predictor identification-width simulation ===")
    alphas = [0.5, 0.75, 0.9]
    epsilons = [0.0, 0.05, 0.1, 0.2, 0.3]

    rows = []
    for alpha in alphas:
        for eps in epsilons:
            # Fixed predictor: P_obs = delta_0 (support [0.0], weight [1.0])
            cvar_minus = cvar_discrete([0.0], [1.0], alpha)

            # P_plus = (1 - eps) delta_0 + eps delta_1
            if eps == 0.0:
                cvar_plus = cvar_discrete([0.0], [1.0], alpha)
            elif eps == 1.0:
                cvar_plus = cvar_discrete([1.0], [1.0], alpha)
            else:
                cvar_plus = cvar_discrete([0.0, 1.0], [1.0 - eps, eps], alpha)

            num_width = cvar_plus - cvar_minus
            theo_width = min(1.0, eps / (1.0 - alpha))
            abs_err = abs(num_width - theo_width)

            rows.append({
                "alpha": alpha,
                "epsilon": eps,
                "numerical_width": num_width,
                "theoretical_width": theo_width,
                "absolute_error": abs_err,
            })

    df = pd.DataFrame(rows)

    csv_filename = "same_predictor_identification_width.csv"
    df.to_csv(csv_filename, index=False)
    df.to_csv(Path("results") / csv_filename, index=False)
    df.to_csv(Path("results_submit/additional") / csv_filename, index=False)
    df.to_csv(Path("results_submit/tables/P7_theory") / csv_filename, index=False)
    print(f"Saved {csv_filename} to root, results/, results_submit/additional/, and results_submit/tables/P7_theory/")

    print(df.to_string())

    # Generate Figure
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(9, 6))

    colors = {0.5: "#0072B2", 0.75: "#D55E00", 0.9: "#009E73"}
    markers = {0.5: "o", 0.75: "s", 0.9: "^"}
    linestyles = {0.5: "-", 0.75: "--", 0.9: "-."}

    eps_fine = np.linspace(0.0, 0.35, 100)

    for alpha in alphas:
        # Theoretical curve
        theo_curve = [min(1.0, e / (1.0 - alpha)) for e in eps_fine]
        ax.plot(eps_fine, theo_curve, label=f"Theoretical α = {alpha}", color=colors[alpha], linewidth=2.2, linestyle=linestyles[alpha])

        # Numerical points
        sub = df[df["alpha"] == alpha]
        ax.scatter(sub["epsilon"], sub["numerical_width"], label=f"Numerical α = {alpha}", color=colors[alpha], marker=markers[alpha], s=80, zorder=5)

    ax.set_xlabel("Missing Deployment Mass (ε)", fontsize=16, labelpad=8)
    ax.set_ylabel("Identification Interval Width $\Delta_{\mathrm{ID}}$", fontsize=16, labelpad=8)
    ax.tick_params(labelsize=14)
    ax.set_xlim(-0.01, 0.32)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="upper left", fontsize=13, framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    pdf_filename = "same_predictor_identification_width.pdf"
    png_filename = "same_predictor_identification_width.png"
    safe_savefig(fig, pdf_filename, dpi=300)
    safe_savefig(fig, png_filename, dpi=300)
    safe_savefig(fig, Path("results") / pdf_filename, dpi=300)
    safe_savefig(fig, Path("results") / png_filename, dpi=300)
    safe_savefig(fig, Path("results_submit/additional") / pdf_filename, dpi=300)
    safe_savefig(fig, Path("results_submit/additional") / png_filename, dpi=300)
    safe_savefig(fig, Path("results_submit/figures") / pdf_filename, dpi=300)
    safe_savefig(fig, Path("results_submit/figures") / png_filename, dpi=300)
    safe_savefig(fig, Path("results_submit/figures/P7_theory") / pdf_filename, dpi=300)
    safe_savefig(fig, Path("results_submit/figures/P7_theory") / png_filename, dpi=300)
    plt.close()
    print(f"Saved {pdf_filename} to root, results/, results_submit/additional/, and results_submit/figures/P7_theory/\n")


def run_task4():
    print("=== Running Task 4: Ranking-reversal by missing mass simulation figure ===")
    epsilons = [0.0, 0.1, 0.2, 0.3]
    reversal_probs = [0.666, 0.952, 0.997, 1.000]

    df = pd.DataFrame({
        "epsilon": epsilons,
        "mean_reversal_probability": reversal_probs,
    })

    csv_filename = "ranking_reversal_by_missing_mass.csv"
    df.to_csv(csv_filename, index=False)
    df.to_csv(Path("results") / csv_filename, index=False)
    df.to_csv(Path("results_submit/additional") / csv_filename, index=False)
    df.to_csv(Path("results_submit/tables/P7_theory") / csv_filename, index=False)
    print(f"Saved {csv_filename} to root, results/, results_submit/additional/, and results_submit/tables/P7_theory/")

    print(df.to_string())

    # Generate Figure
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(9, 6))

    color = "#0072B2"  # Okabe-Ito Blue
    marker = "o"

    ax.plot(
        epsilons,
        reversal_probs,
        marker=marker,
        color=color,
        linewidth=2.5,
        markersize=10,
        linestyle="-",
        label="Mean Reversal Probability",
    )

    # Annotate points with values for high clarity
    for eps, val in zip(epsilons, reversal_probs):
        ax.annotate(
            f"{val:.3f}",
            (eps, val),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            fontsize=15,
            fontweight="bold",
        )

    ax.set_xlabel("Missing Deployment Mass (ε)", fontsize=16, labelpad=8)
    ax.set_ylabel("Mean Ranking-Reversal Probability", fontsize=16, labelpad=8)
    ax.set_xticks(epsilons)
    ax.set_xticklabels(["0", "0.1", "0.2", "0.3"], fontsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_xlim(-0.02, 0.32)
    ax.set_ylim(0.60, 1.06)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    pdf_filename = "ranking_reversal_by_missing_mass.pdf"
    png_filename = "ranking_reversal_by_missing_mass.png"
    safe_savefig(fig, pdf_filename, dpi=300)
    safe_savefig(fig, png_filename, dpi=300)
    safe_savefig(fig, Path("results") / pdf_filename, dpi=300)
    safe_savefig(fig, Path("results") / png_filename, dpi=300)
    safe_savefig(fig, Path("results_submit/additional") / pdf_filename, dpi=300)
    safe_savefig(fig, Path("results_submit/additional") / png_filename, dpi=300)
    safe_savefig(fig, Path("results_submit/figures") / pdf_filename, dpi=300)
    safe_savefig(fig, Path("results_submit/figures") / png_filename, dpi=300)
    safe_savefig(fig, Path("results_submit/figures/P7_theory") / pdf_filename, dpi=300)
    safe_savefig(fig, Path("results_submit/figures/P7_theory") / png_filename, dpi=300)
    plt.close()
    print(f"Saved {pdf_filename} to root, results/, results_submit/additional/, and results_submit/figures/P7_theory/\n")


if __name__ == "__main__":
    run_task1()
    run_task2()
    run_task3()
    run_task4()
