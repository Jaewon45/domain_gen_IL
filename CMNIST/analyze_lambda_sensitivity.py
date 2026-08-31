#!/usr/bin/env python3
"""Audit CMNIST lambda-evaluation records without loading model checkpoints."""

import argparse
import json
from pathlib import Path

import pandas as pd


ENVIRONMENTS = [str(index / 10.0) for index in range(11)]


def load_records(input_dir):
    records = []
    for path in sorted(Path(input_dir).rglob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    record = json.loads(line)
                    record["source_file"] = str(path)
                    record["source_line"] = line_number
                    records.append(record)
    return records


def summarize(frame):
    frame = frame.copy()
    frame["lambda_eval"] = frame["lambda_eval"].astype(float)
    frame["checkpoint_key"] = frame["checkpoint_path"].astype(str) + "|" + frame["checkpoint_type"].astype(str)
    dedup_columns = ["checkpoint_key", "algorithm", "seed", "lambda_eval"]
    frame = frame.sort_values(["checkpoint_key", "lambda_eval", "source_file", "source_line"])
    frame = frame.drop_duplicates(dedup_columns, keep="last")

    rows = []
    for keys, group in frame.groupby(["checkpoint_key", "algorithm", "seed"], dropna=False):
        group = group.sort_values("lambda_eval")
        for metric in ["avg_domain_acc", "worst_domain_acc", "aggregated_risk"]:
            values = group[metric].astype(float)
            changes = values.diff().abs().dropna()
            rows.append({
                "checkpoint_key": keys[0],
                "algorithm": keys[1],
                "seed": keys[2],
                "metric": metric,
                "lambda_count": int(group["lambda_eval"].nunique()),
                "lambda_min": float(group["lambda_eval"].min()),
                "lambda_max": float(group["lambda_eval"].max()),
                "best_value": float(values.max()),
                "worst_value": float(values.min()),
                "best_worst_range": float(values.max() - values.min()),
                "max_neighbor_change": float(changes.max()) if not changes.empty else 0.0,
                "mean_value": float(values.mean()),
            })
    return frame, pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lambda_results_dir")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(args.lambda_results_dir)
    if not records:
        raise ValueError("No lambda-evaluation records found")
    frame = pd.DataFrame(records)
    required = {"checkpoint_path", "checkpoint_type", "algorithm", "seed", "lambda_eval", "avg_domain_acc", "worst_domain_acc", "aggregated_risk"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    deduped, summary = summarize(frame)
    deduped.to_json(output_dir / "lambda_records_deduplicated.jsonl", orient="records", lines=True)
    summary.to_csv(output_dir / "lambda_sensitivity_summary.csv", index=False)

    coverage = (
        deduped.groupby(["algorithm", "seed"], dropna=False)
        .agg(checkpoints=("checkpoint_key", "nunique"), rows=("lambda_eval", "size"), lambda_values=("lambda_eval", "nunique"))
        .reset_index()
    )
    coverage.to_csv(output_dir / "lambda_coverage.csv", index=False)
    print(f"Raw records: {len(records)}")
    print(f"Deduplicated records: {len(deduped)}")
    print(f"Sensitivity rows: {len(summary)}")
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
