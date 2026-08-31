#!/usr/bin/env python3
"""Build a CMNIST experiment artifact index without modifying outputs."""

import argparse
import csv
from pathlib import Path

EXPERIMENTS = [
    ("E1_corrected_domain_count", "results/cmnist_domain_count_clean_v1", "CMNIST/job_scripts/domain_count_clean.txt"),
    ("E3b_tail_support", "results/E3b_tail_support", "CMNIST/job_scripts/e3b_tail_support.txt"),
    ("E3_corrected_imbalance", "results/imbalance_clean_v1", "CMNIST/job_scripts/imbalance_clean.txt"),
    ("E4_lambda_predictions", "results/cmnist_lambda_prediction_eval_v2", "CMNIST/job_scripts/bashes/run_lambda_prediction_eval.ps1"),
    ("E4_pseudo_regret", "results/cmnist_lambda_pseudoregret_v2", "CMNIST/job_scripts/bashes/run_regret_logging.sh"),
    ("Priority7_theory", "results/cmnist_priority7_theory_v1", "CMNIST/job_scripts/bashes/run_priority7_theory.ps1"),
    ("GroupDRO_controlled", "results/cmnist_groupdro_control_v2", "CMNIST/job_scripts/groupdro_controlled_seed012_v2.txt"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/cmnist_artifact_index.csv")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    output = repo_root / args.output
    rows = []
    for name, relative_root, relative_manifest in EXPERIMENTS:
        root = repo_root / relative_root
        manifest = repo_root / relative_manifest
        files = [path for path in root.rglob("*") if path.is_file()] if root.exists() else []
        rows.append({
            "experiment": name,
            "result_root": relative_root,
            "manifest": relative_manifest,
            "result_root_exists": root.exists(),
            "manifest_exists": manifest.exists(),
            "file_count": len(files),
            "jsonl_count": sum(path.suffix == ".jsonl" for path in files),
            "csv_count": sum(path.suffix == ".csv" for path in files),
            "plot_count": sum(path.suffix == ".png" for path in files),
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Indexed {len(rows)} CMNIST experiment roots")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
