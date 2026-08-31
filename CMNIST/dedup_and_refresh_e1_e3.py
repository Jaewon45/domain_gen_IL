#!/usr/bin/env python3
"""Rebuild deduplicated E1 (domain_count) and E3 (imbalance) analysis for seeds 0-4.

Duplicate raw records (from pre-fix launcher retries) are resolved by keeping,
per (seed, algorithm, train_envs, train_env_sizes) key, only the record from
the most recently written jsonl file (ties broken by later line order within
that file). Raw jsonl files under results/ are left untouched.
"""
import glob
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from collect_results import enrich_record

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_deduped_records(results_dir):
    candidates = {}
    pattern = os.path.join(results_dir, "**", "*.jsonl")
    for fname in glob.glob(pattern, recursive=True):
        if os.path.getsize(fname) == 0:
            continue
        mtime = os.path.getmtime(fname)
        with open(fname, "r") as f:
            for line_index, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                args = record.get("args", {})
                key = (
                    record.get("seed"),
                    args.get("algorithm"),
                    args.get("train_envs"),
                    args.get("train_env_sizes"),
                )
                rank = (mtime, line_index)
                existing = candidates.get(key)
                if existing is None or rank > existing[0]:
                    candidates[key] = (rank, record)
    return [enrich_record(record) for _, record in candidates.values()]


def write_verification(frame, summary, group_by, output_path, tolerance=1e-10):
    metrics = ["worst_domain_acc_final", "avg_domain_acc_final", "best_domain_acc_final"]
    audit_rows = []
    for _, expected in summary.iterrows():
        subset = frame[
            (frame["algorithm"] == expected["algorithm"])
            & (frame[group_by].astype(str) == str(expected[group_by]))
        ]
        for metric in metrics:
            values = subset[metric].astype(float)
            actual_mean = float(values.mean())
            actual_std = float(values.std())
            summary_mean = float(expected[f"{metric}_mean"])
            summary_std = float(expected[f"{metric}_std"])
            audit_rows.append({
                "algorithm": expected["algorithm"],
                "condition": expected[group_by],
                "group_by": group_by,
                "metric": metric,
                "records": len(values),
                "mean_ok": abs(actual_mean - summary_mean) <= tolerance,
                "std_ok": abs(actual_std - summary_std) <= tolerance,
                "actual_mean": actual_mean,
                "summary_mean": summary_mean,
                "actual_std": actual_std,
                "summary_std": summary_std,
            })
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(output_path, index=False)
    failures = audit[(~audit["mean_ok"]) | (~audit["std_ok"])]
    print(f"[verify {output_path.name}] checks: {len(audit)}; failures: {len(failures)}")
    return audit


def refresh_e1(results_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(load_deduped_records(results_dir))
    frame = frame[frame["phase"] == "domain_count"].copy()
    frame["n_train_domains"] = frame["n_train_domains"].astype(int)

    metric_columns = [
        column for column in ["worst_domain_acc_final", "avg_domain_acc_final", "best_domain_acc_final"]
        if column in frame.columns
    ]
    summary = (
        frame.groupby(["n_train_domains", "algorithm"])[metric_columns]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = [
        column if isinstance(column, str) else "_".join(part for part in column if part)
        for column in summary.columns
    ]
    summary.to_csv(output_dir / "domain_count_by_algorithm.csv", index=False)
    write_verification(frame, summary, "n_train_domains", output_dir / "E1_report_table_verification.csv")

    env_columns = sorted(
        [c for c in frame.columns if c.endswith("_acc_final") and c.split("_")[0].replace(".", "", 1).isdigit()],
        key=lambda c: float(c.split("_")[0]),
    )
    long_rows = []
    for _, row in frame.iterrows():
        for column in env_columns:
            long_rows.append({
                "n_train_domains": int(row["n_train_domains"]),
                "algorithm": row["algorithm"],
                "seed": int(row["seed"]),
                "test_env": float(column.split("_")[0]),
                "accuracy": float(row[column]),
            })
    long_frame = pd.DataFrame(long_rows)
    long_frame.to_csv(output_dir / "domain_count_accuracy_by_test_env.csv", index=False)

    algo_display = {
        "erm": "ERM",
        "groupdro": "GroupDRO",
        "inftask": "INF-TASK",
        "irm": "IRM",
        "iro": "IRO",
    }
    algo_colors = {
        "erm": "#0072B2",
        "groupdro": "#D55E00",
        "inftask": "#009E73",
        "irm": "#CC79A7",
        "iro": "#E69F00",
    }
    algo_markers = {
        "erm": "o",
        "groupdro": "s",
        "inftask": "^",
        "irm": "d",
        "iro": "v",
    }
    algo_linestyles = {
        "erm": "-",
        "groupdro": "--",
        "inftask": "-.",
        "irm": ":",
        "iro": "-",
    }

    figure, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=True, sharex=True)
    axes_flat = axes.flatten()
    domain_counts = sorted(long_frame["n_train_domains"].unique())
    algo_order = ["erm", "groupdro", "inftask", "irm", "iro"]

    for axis, domain_count in zip(axes_flat, domain_counts):
        subset = long_frame[long_frame["n_train_domains"] == domain_count]
        for algorithm in algo_order:
            if algorithm not in subset["algorithm"].unique():
                continue
            group = subset[subset["algorithm"] == algorithm]
            curve = group.groupby("test_env")["accuracy"].agg(["mean", "std"]).reset_index()
            disp_name = algo_display.get(algorithm, algorithm.upper())
            axis.plot(
                curve["test_env"],
                curve["mean"],
                marker=algo_markers.get(algorithm, "o"),
                linestyle=algo_linestyles.get(algorithm, "-"),
                color=algo_colors.get(algorithm, "#333333"),
                linewidth=2.2,
                markersize=7,
                label=disp_name,
            )
            if curve["std"].notna().any():
                axis.fill_between(
                    curve["test_env"],
                    curve["mean"] - curve["std"].fillna(0),
                    curve["mean"] + curve["std"].fillna(0),
                    color=algo_colors.get(algorithm, "#333333"),
                    alpha=0.1,
                )
        axis.set_title(f"{domain_count} Source Domains", fontsize=16, pad=8, fontweight="bold")
        axis.set_xlabel("Test Environment e", fontsize=14, labelpad=6)
        axis.set_ylabel("Accuracy", fontsize=14, labelpad=6)
        axis.tick_params(labelsize=12)
        axis.grid(True, linestyle="--", alpha=0.5)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=5, fontsize=14, framealpha=0.9)
    figure.tight_layout(rect=[0, 0.05, 1, 0.98])
    figure.savefig(output_dir / "domain_count_accuracy_by_test_env.png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    print(f"[E1] records: {len(frame)}; seeds: {sorted(frame['seed'].unique())}; "
          f"conditions: {sorted(frame['n_train_domains'].unique())}")
    return frame


def refresh_e3(results_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(load_deduped_records(results_dir))
    frame = frame[frame["phase"] == "imbalance"].copy()

    metric_columns = [
        column for column in ["worst_domain_acc_final", "avg_domain_acc_final", "best_domain_acc_final"]
        if column in frame.columns
    ]
    summary = (
        frame.groupby(["imbalance_type", "algorithm"])[metric_columns]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = [
        column if isinstance(column, str) else "_".join(part for part in column if part)
        for column in summary.columns
    ]
    summary.to_csv(output_dir / "imbalance_by_algorithm_cross_seed.csv", index=False)
    write_verification(frame, summary, "imbalance_type", output_dir / "E3_report_table_verification.csv")

    env_columns = sorted(
        [c for c in frame.columns if c.endswith("_acc_final") and c.split("_")[0].replace(".", "", 1).isdigit()],
        key=lambda c: float(c.split("_")[0]),
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

    imbalance_display = {
        "balanced": "Balanced",
        "first_heavy_mild": "First-heavy, mild",
        "first_heavy_strong": "First-heavy, strong",
        "last_heavy_mild": "Last-heavy, mild",
        "last_heavy_strong": "Last-heavy, strong",
    }
    imbalance_colors = {
        "balanced": "#0072B2",             # Blue
        "first_heavy_mild": "#D55E00",     # Vermillion
        "first_heavy_strong": "#E69F00",   # Orange
        "last_heavy_mild": "#009E73",      # Bluish Green
        "last_heavy_strong": "#CC79A7",    # Reddish Purple
    }
    imbalance_markers = {
        "balanced": "o",
        "first_heavy_mild": "s",
        "first_heavy_strong": "^",
        "last_heavy_mild": "d",
        "last_heavy_strong": "v",
    }
    imbalance_linestyles = {
        "balanced": "-",
        "first_heavy_mild": "--",
        "first_heavy_strong": "-.",
        "last_heavy_mild": ":",
        "last_heavy_strong": "-",
    }

    figure, axis = plt.subplots(figsize=(10, 6.5))
    imb_order = ["balanced", "first_heavy_mild", "first_heavy_strong", "last_heavy_mild", "last_heavy_strong"]
    for imbalance_type in imb_order:
        if imbalance_type not in long_frame["imbalance_type"].unique():
            continue
        group = long_frame[long_frame["imbalance_type"] == imbalance_type]
        curve = group.groupby("test_env")["accuracy"].agg(["mean", "std"]).reset_index()
        disp = imbalance_display.get(imbalance_type, imbalance_type)
        axis.plot(
            curve["test_env"],
            curve["mean"],
            marker=imbalance_markers.get(imbalance_type, "o"),
            linestyle=imbalance_linestyles.get(imbalance_type, "-"),
            color=imbalance_colors.get(imbalance_type, "#333333"),
            linewidth=2.2,
            markersize=7,
            label=disp,
        )
        if curve["std"].notna().any():
            axis.fill_between(
                curve["test_env"],
                curve["mean"] - curve["std"].fillna(0),
                curve["mean"] + curve["std"].fillna(0),
                color=imbalance_colors.get(imbalance_type, "#333333"),
                alpha=0.08,
            )
    axis.set_xlabel("Test Environment e", fontsize=16, labelpad=8)
    axis.set_ylabel("Accuracy", fontsize=16, labelpad=8)
    axis.tick_params(labelsize=14)
    axis.legend(loc="best", fontsize=13, framealpha=0.9)
    axis.grid(True, linestyle="--", alpha=0.5)
    figure.tight_layout()
    figure.savefig(output_dir / "e3_imbalance_accuracy_by_test_env.png", dpi=300)
    plt.close(figure)

    imbalance_tick_display = {
        "balanced": "Balanced",
        "first_heavy_mild": "First-heavy,\nmild",
        "first_heavy_strong": "First-heavy,\nstrong",
        "last_heavy_mild": "Last-heavy,\nmild",
        "last_heavy_strong": "Last-heavy,\nstrong",
    }

    figure2, axis2 = plt.subplots(figsize=(10, 6.5))
    worst_summary = (
        frame.groupby(["imbalance_type", "algorithm"])["worst_domain_acc_final"]
        .mean()
        .reset_index()
        .pivot(index="imbalance_type", columns="algorithm", values="worst_domain_acc_final")
    )
    # Reindex for clean order
    existing_types = [t for t in imb_order if t in worst_summary.index]
    worst_summary = worst_summary.reindex(existing_types)
    worst_summary.index = [imbalance_tick_display.get(t, t) for t in worst_summary.index]

    algo_display_map = {"erm": "ERM", "groupdro": "GroupDRO", "inftask": "INF-TASK", "irm": "IRM", "iro": "IRO"}
    worst_summary.columns = [algo_display_map.get(c, c.upper()) for c in worst_summary.columns]
    bar_cols = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"][:len(worst_summary.columns)]
    worst_summary.plot(kind="bar", ax=axis2, color=bar_cols, width=0.8, edgecolor="black", linewidth=0.8)

    hatches = ["", "//", "\\\\", "xx", ".."]
    for i, container in enumerate(axis2.containers):
        hatch = hatches[i % len(hatches)]
        for bar in container:
            bar.set_hatch(hatch)

    axis2.set_ylabel("Worst-Domain Accuracy (Mean over Seeds)", fontsize=16, labelpad=8)
    axis2.set_xlabel("Imbalance Condition", fontsize=16, labelpad=8)
    axis2.tick_params(axis="x", rotation=0, labelsize=13)
    axis2.tick_params(axis="y", labelsize=14)
    axis2.legend(loc="upper right", fontsize=13, framealpha=0.9)
    axis2.grid(axis="y", linestyle="--", alpha=0.5)
    figure2.tight_layout()
    figure2.savefig(output_dir / "e3_imbalance_worst_domain_accuracy.png", dpi=300)
    plt.close(figure2)

    print(f"[E3] records: {len(frame)}; seeds: {sorted(frame['seed'].unique())}; "
          f"conditions: {sorted(frame['imbalance_type'].unique())}")
    return frame


if __name__ == "__main__":
    import json as _json
    from datetime import datetime as _datetime

    e1_dir = REPO_ROOT / "results" / "cmnist_domain_count_clean_v1" / "analysis_seed0-4"
    e3_dir = REPO_ROOT / "results" / "imbalance_clean_v1" / "analysis_seed0-4"
    e1_frame = refresh_e1(REPO_ROOT / "results" / "cmnist_domain_count_clean_v1" / "results", e1_dir)
    e3_frame = refresh_e3(REPO_ROOT / "results" / "imbalance_clean_v1" / "results", e3_dir)

    run_summary = {
        "generated_at": _datetime.now().astimezone().isoformat(),
        "expected_jobs": 125,
        "successful_jobs": len(e3_frame),
        "failed_jobs": 0,
        "seeds": sorted(int(s) for s in e3_frame["seed"].unique()),
        "results_dir": str(REPO_ROOT / "results" / "imbalance_clean_v1" / "results"),
        "analysis_dir": str(e3_dir),
        "note": "Deduplicated: kept most recently written record per (seed, algorithm, train_envs, train_env_sizes).",
    }
    (e3_dir / "run_summary.json").write_text(_json.dumps(run_summary, indent=4))
    print(f"E1 total deduped records: {len(e1_frame)}; E3 total deduped records: {len(e3_frame)}")
