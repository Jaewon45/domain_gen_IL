#!/usr/bin/env python3
"""Create non-invasive CMNIST lambda pseudo-regret logs from post-training metrics.

Pseudo-regret uses one deployment-wide best lambda on the same checkpoint as an
oracle. True operator regret is not computed by this pipeline.
No training or checkpoint processing is performed here.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def load_lambda_metrics(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    rows = []
    for jsonl_path in sorted(path.rglob("*.jsonl")):
        with jsonl_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    return pd.DataFrame(rows)


def normalize_metrics(frame):
    required = {"checkpoint_path", "algorithm", "seed", "lambda_eval", "test_env", "accuracy"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required lambda metric columns: {sorted(missing)}")
    frame = frame.copy()
    frame["lambda_eval"] = frame["lambda_eval"].astype(float)
    frame["test_env"] = frame["test_env"].astype(float)
    frame["accuracy"] = frame["accuracy"].astype(float)
    frame["checkpoint_key"] = frame["checkpoint_path"].astype(str)
    return frame


def build_pseudo_regret(frame):
    rows = []
    deployment_groups = ["checkpoint_key", "algorithm", "seed", "lambda_eval"]
    deployment_accuracy = (
        frame.groupby(deployment_groups, dropna=False)["accuracy"]
        .mean()
        .reset_index(name="deployment_mean_accuracy")
    )
    oracle_groups = ["checkpoint_key", "algorithm", "seed"]
    for keys, group in deployment_accuracy.groupby(oracle_groups, dropna=False):
        group = group.sort_values("lambda_eval")
        oracle_row = group.loc[group["deployment_mean_accuracy"].idxmax()]
        oracle_lambda = float(oracle_row["lambda_eval"])
        oracle_accuracy = float(oracle_row["deployment_mean_accuracy"])
        checkpoint_rows = frame[
            (frame["checkpoint_key"] == keys[0])
            & (frame["algorithm"] == keys[1])
            & (frame["seed"] == keys[2])
        ].merge(
            deployment_accuracy,
            on=["checkpoint_key", "algorithm", "seed", "lambda_eval"],
            how="left",
        )
        for _, row in checkpoint_rows.iterrows():
            rows.append({
                "checkpoint_path": keys[0],
                "algorithm": keys[1],
                "seed": keys[2],
                "test_env": float(row["test_env"]),
                "lambda_used": float(row["lambda_eval"]),
                "oracle_lambda": oracle_lambda,
                "metric": "accuracy",
                "oracle_value": oracle_accuracy,
                "used_value": float(row["accuracy"]),
                "used_deployment_mean_accuracy": float(row["deployment_mean_accuracy"]),
                "oracle_deployment_mean_accuracy": oracle_accuracy,
                "deployment_wide_pseudo_regret": oracle_accuracy - float(row["deployment_mean_accuracy"]),
                "pseudo_regret": oracle_accuracy - float(row["deployment_mean_accuracy"]),
                "regret_type": "same_checkpoint_deployment_wide_lambda_oracle",
            })
    return pd.DataFrame(rows)


def add_true_regret(pseudo, reference_path):
    reference = pd.read_csv(reference_path)
    required = {"algorithm", "seed", "test_env", "lambda_reference", "accuracy"}
    missing = required.difference(reference.columns)
    if missing:
        raise ValueError(f"Missing reference columns: {sorted(missing)}")
    reference = reference.copy()
    reference["seed"] = reference["seed"].astype(int)
    reference["test_env"] = reference["test_env"].astype(float)
    reference["lambda_reference"] = reference["lambda_reference"].astype(float)
    reference["accuracy"] = reference["accuracy"].astype(float)
    reference = reference.rename(columns={"accuracy": "reference_accuracy"})
    merged = pseudo.merge(reference, left_on=["algorithm", "seed", "test_env", "lambda_used"], right_on=["algorithm", "seed", "test_env", "lambda_reference"], how="left")
    merged["true_regret"] = merged["reference_accuracy"] - merged["used_value"]
    merged["regret_type"] = merged["reference_accuracy"].notna().map({True: "fixed_lambda_reference", False: "same_checkpoint_lambda_oracle"})
    return merged.drop(columns=["lambda_reference"], errors="ignore")


def summarize(regret):
    group_columns = ["algorithm", "seed", "lambda_used", "regret_type"]
    summary = regret.groupby(group_columns, dropna=False).agg(
        mean_pseudo_regret=("pseudo_regret", "mean"),
        worst_pseudo_regret=("pseudo_regret", "max"),
        mean_true_regret=("true_regret", "mean"),
        worst_true_regret=("true_regret", "max"),
        test_environments=("test_env", "nunique"),
    ).reset_index()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lambda_metrics", help="CSV file or directory of lambda JSONL records.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--fixed_reference_csv", help="Optional reference CSV for separate future analysis; not used for pseudo-regret.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = normalize_metrics(load_lambda_metrics(args.lambda_metrics))
    if frame.empty:
        raise ValueError("No lambda metrics found")
    regret = build_pseudo_regret(frame)
    regret["true_regret"] = float("nan")
    if args.fixed_reference_csv:
        regret = add_true_regret(regret, args.fixed_reference_csv)

    regret.to_csv(output_dir / "regret_log.csv", index=False)
    summarize(regret).to_csv(output_dir / "regret_summary.csv", index=False)
    metadata = {
        "regret_type": "fixed_lambda_reference" if args.fixed_reference_csv else "same_checkpoint_deployment_wide_lambda_oracle",
        "lambda_metrics": str(args.lambda_metrics),
        "fixed_reference_csv": args.fixed_reference_csv,
        "rows": int(len(regret)),
        "checkpoints": int(regret["checkpoint_path"].nunique()),
    }
    (output_dir / "regret_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Lambda metric rows: {len(frame)}")
    print(f"Regret rows: {len(regret)}")
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
