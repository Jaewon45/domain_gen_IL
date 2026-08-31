#!/usr/bin/env python3
"""Verify report summaries against clean CMNIST JSONL records."""

import argparse
from pathlib import Path

import pandas as pd

from collect_results import load_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir")
    parser.add_argument("summary_csv")
    parser.add_argument("--phase", default="domain_count")
    parser.add_argument("--group_by", default=None, help="Condition column; defaults to n_train_domains for domain_count or imbalance_type for imbalance.")
    parser.add_argument("--tolerance", type=float, default=1e-10)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    records = pd.DataFrame(load_records(args.results_dir))
    records = records[records["phase"] == args.phase].copy()
    summary = pd.read_csv(args.summary_csv)
    group_by = args.group_by or ("n_train_domains" if args.phase == "domain_count" else "imbalance_type")
    metrics = ["worst_domain_acc_best", "avg_domain_acc_best", "best_domain_acc_best"]
    audit_rows = []
    for _, expected in summary.iterrows():
        subset = records[
            (records["algorithm"] == expected["algorithm"])
            & (records[group_by].astype(str) == str(expected[group_by]))
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
                "mean_ok": abs(actual_mean - summary_mean) <= args.tolerance,
                "std_ok": abs(actual_std - summary_std) <= args.tolerance,
                "actual_mean": actual_mean,
                "summary_mean": summary_mean,
                "actual_std": actual_std,
                "summary_std": summary_std,
            })
    audit = pd.DataFrame(audit_rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output, index=False)
    failures = audit[(~audit["mean_ok"]) | (~audit["std_ok"])]
    print(f"Checks: {len(audit)}; failures: {len(failures)}; wrote {output}")
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
