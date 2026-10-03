#!/usr/bin/env python3
"""Aggregate ImageNet-100-C evaluation JSONL and compare identification intervals."""

from __future__ import annotations

import argparse
import csv
import glob
import json
from itertools import combinations
from pathlib import Path

try:
    from .evaluate import intervals_separated
except ImportError:
    from evaluate import intervals_separated


def read_records(paths):
    records = []
    for pattern in paths:
        matches = [Path(pattern)] if Path(pattern).is_file() else [Path(value) for value in glob.glob(pattern, recursive=True)]
        for path in matches:
            with path.open("r", encoding="utf-8") as handle:
                records.extend(json.loads(line) for line in handle if line.strip())
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze ImageNet-100-C evaluations")
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    records = read_records(args.inputs)
    if not records:
        raise ValueError("No evaluation records found")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)

    rows = []
    for record in records:
        for alpha, interval in record["partial_identification"].items():
            rows.append({
                "manifest_id": record["manifest_id"], "experiment": record["experiment"],
                "condition": record["condition"], "protocol": record["protocol"],
                "algorithm": record["algorithm"], "seed": record["seed"], "lambda": record["lambda"],
                "alpha": alpha, "clean_top1": record["clean"]["top1"],
                "mean_domain_error": record["mean_domain_error"],
                "worst_domain_error": record["worst_domain_error"],
                "cvar_domain_error": record["domain_cvar_error"][alpha],
                "identification_lower": interval["lower"],
                "identification_upper": interval["upper"],
                "identification_width": interval["width"],
                "epsilon": interval["epsilon"],
                "realized_plugin_cvar": interval["realized_held_out_deployment_cvar_descriptive_plugin"],
            })
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    ranking_rows = []
    group_keys = sorted({(row["experiment"], row["condition"], row["seed"], row["lambda"], row["alpha"]) for row in rows})
    for group_key in group_keys:
        group = [row for row in rows if (row["experiment"], row["condition"], row["seed"], row["lambda"], row["alpha"]) == group_key]
        for first, second in combinations(group, 2):
            first_interval = [first["identification_lower"], first["identification_upper"]]
            second_interval = [second["identification_lower"], second["identification_upper"]]
            separated = intervals_separated(first_interval, second_interval)
            if first_interval[1] < second_interval[0]:
                preferred = first["algorithm"]
            elif second_interval[1] < first_interval[0]:
                preferred = second["algorithm"]
            else:
                preferred = "not_robustly_ranked"
            ranking_rows.append({
                "experiment": group_key[0], "condition": group_key[1], "seed": group_key[2],
                "lambda": group_key[3], "alpha": group_key[4],
                "algorithm_a": first["algorithm"], "algorithm_b": second["algorithm"],
                "intervals_separated": separated, "robustly_preferred_lower_loss": preferred,
            })
    if ranking_rows:
        with (output / "robust_ranking.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(ranking_rows[0]))
            writer.writeheader()
            writer.writerows(ranking_rows)
    with (output / "analysis_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump({"inputs": args.inputs, "records": len(records), "summary_rows": len(rows), "ranking_rows": len(ranking_rows)}, handle, indent=2)


if __name__ == "__main__":
    main()

