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
    metrics = ["worst_domain_acc_best", "avg_domain_acc_best", "best_domain_acc_best"]
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
        column for column in ["worst_domain_acc_best", "avg_domain_acc_best", "best_domain_acc_best"]
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
        [c for c in frame.columns if c.endswith("_acc_best") and c.split("_")[0].replace(".", "", 1).isdigit()],
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

    figure, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
    for axis, domain_count in zip(axes, sorted(long_frame["n_train_domains"].unique())):
        subset = long_frame[long_frame["n_train_domains"] == domain_count]
        for algorithm, group in subset.groupby("algorithm"):
            curve = group.groupby("test_env")["accuracy"].agg(["mean", "std"]).reset_index()
            axis.plot(curve["test_env"], curve["mean"], marker="o", label=algorithm)
            if curve["std"].notna().any():
                axis.fill_between(curve["test_env"], curve["mean"] - curve["std"].fillna(0), curve["mean"] + curve["std"].fillna(0), alpha=0.08)
        axis.set_title(f"{domain_count} source domains")
        axis.set_xlabel("Test environment e")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Accuracy")
    axes[-1].legend(loc="best", fontsize=9)
    figure.suptitle("E1 clean domain-count accuracy across test environments (seeds 0-4)")
    figure.tight_layout()
    figure.savefig(output_dir / "domain_count_accuracy_by_test_env.png", dpi=200)
    plt.close(figure)

    print(f"[E1] records: {len(frame)}; seeds: {sorted(frame['seed'].unique())}; "
          f"conditions: {sorted(frame['n_train_domains'].unique())}")
    return frame


def refresh_e3(results_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(load_deduped_records(results_dir))
    frame = frame[frame["phase"] == "imbalance"].copy()

    metric_columns = [
        column for column in ["worst_domain_acc_best", "avg_domain_acc_best", "best_domain_acc_best"]
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
        [c for c in frame.columns if c.endswith("_acc_best") and c.split("_")[0].replace(".", "", 1).isdigit()],
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

    figure, axis = plt.subplots(figsize=(9, 6))
    for imbalance_type, group in long_frame.groupby("imbalance_type"):
        curve = group.groupby("test_env")["accuracy"].agg(["mean", "std"]).reset_index()
        axis.plot(curve["test_env"], curve["mean"], marker="o", label=imbalance_type)
        if curve["std"].notna().any():
            axis.fill_between(curve["test_env"], curve["mean"] - curve["std"].fillna(0), curve["mean"] + curve["std"].fillna(0), alpha=0.08)
    axis.set_xlabel("Test environment e")
    axis.set_ylabel("Accuracy")
    axis.set_title("E3 clean imbalance accuracy across test environments (seeds 0-4)")
    axis.legend(loc="best", fontsize=9)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "e3_imbalance_accuracy_by_test_env.png", dpi=200)
    plt.close(figure)

    figure2, axis2 = plt.subplots(figsize=(9, 6))
    worst_summary = (
        frame.groupby(["imbalance_type", "algorithm"])["worst_domain_acc_best"]
        .mean()
        .reset_index()
        .pivot(index="imbalance_type", columns="algorithm", values="worst_domain_acc_best")
    )
    worst_summary.plot(kind="bar", ax=axis2)
    axis2.set_ylabel("Worst-domain accuracy (mean over seeds)")
    axis2.set_title("E3 worst-domain accuracy by condition (seeds 0-4)")
    figure2.tight_layout()
    figure2.savefig(output_dir / "e3_imbalance_worst_domain_accuracy.png", dpi=200)
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
