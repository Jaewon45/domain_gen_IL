#!/usr/bin/env python3
"""Export CMNIST JSONL experiment results to reportable CSV files."""

import argparse
import json
import os
import re
from typing import Dict, List, Tuple

import pandas as pd

from collect_results import load_records


ENV_METRIC_RE = re.compile(r"^(?P<env>.+)_(?P<metric>acc|loss)_(?P<ms_type>best|final)$")
EXP_E_TAG_RE = re.compile(r"(?:^|[_-])e(?P<idx>[0-4])(?:$|[_-])", re.IGNORECASE)


PHASE_TO_E_TAG = {
    "reproduction": "E0",
    "domain_count": "E1",
    "sample_size": "E2",
    "imbalance": "E3",
    "lambda_eval": "E4",
}


def infer_e_tags(df: pd.DataFrame) -> List[str]:
    tags = set()

    if "phase" in df.columns:
        for phase in df["phase"].dropna().astype(str).unique().tolist():
            mapped = PHASE_TO_E_TAG.get(phase.lower())
            if mapped is not None:
                tags.add(mapped)

    if "exp_name" in df.columns:
        for exp_name in df["exp_name"].dropna().astype(str).unique().tolist():
            for match in EXP_E_TAG_RE.finditer(exp_name):
                tags.add(f"E{match.group('idx')}")

            lowered = exp_name.lower()
            if "phase1" in lowered:
                tags.add("E1")
            if "phase2" in lowered:
                tags.add("E2")
            if "phase3" in lowered:
                tags.add("E3")
            if "phase4" in lowered or "lambda" in lowered:
                tags.add("E4")

    return sorted(tags, key=lambda tag: int(tag[1:]))


def infer_seed_tag(df: pd.DataFrame) -> str:
    if "seed" not in df.columns:
        return ""

    seeds = []
    for seed in df["seed"].dropna().tolist():
        try:
            seeds.append(int(seed))
        except (TypeError, ValueError):
            continue

    if not seeds:
        return ""

    unique_sorted = sorted(set(seeds))
    return f"seed_{''.join(str(seed) for seed in unique_sorted)}"


def infer_e_compact_tag(df: pd.DataFrame) -> str:
    e_tags = infer_e_tags(df)
    if not e_tags:
        return ""
    return f"E_{''.join(tag[1:] for tag in e_tags)}"


def with_standardized_prefix(prefix: str, df: pd.DataFrame) -> str:
    # Strip trailing legacy tags so we can append canonical tags once.
    cleaned = prefix
    cleaned = re.sub(r"(?:_E\d+)+$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"_E_\d+$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(?:_seed\d+)+$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"_seed_\d+$", "", cleaned, flags=re.IGNORECASE)

    suffix_parts = []
    seed_tag = infer_seed_tag(df)
    if seed_tag:
        suffix_parts.append(seed_tag)

    e_tag = infer_e_compact_tag(df)
    if e_tag:
        suffix_parts.append(e_tag)

    if not suffix_parts:
        return cleaned
    return f"{cleaned}_{'_'.join(suffix_parts)}"


def flatten_args_column(df: pd.DataFrame) -> pd.DataFrame:
    if "args" not in df.columns:
        return df

    args_rows: List[Dict[str, object]] = []
    for args in df["args"].tolist():
        args = args or {}
        row: Dict[str, object] = {}
        for key, value in args.items():
            if isinstance(value, (list, tuple, dict)):
                row[f"arg_{key}"] = json.dumps(value, sort_keys=True)
            else:
                row[f"arg_{key}"] = value
        args_rows.append(row)

    args_df = pd.DataFrame(args_rows)
    return pd.concat([df.drop(columns=["args"]), args_df], axis=1)


def build_env_metric_long(df: pd.DataFrame) -> pd.DataFrame:
    env_metric_cols = [
        col for col in df.columns
        if isinstance(col, str) and ENV_METRIC_RE.match(col)
    ]
    if not env_metric_cols:
        return pd.DataFrame()

    base_cols = [
        c for c in [
            "algorithm",
            "seed",
            "exp_name",
            "phase",
            "args_id",
            "n_train_domains",
            "sample_size_per_domain",
            "imbalance_type",
            "train_envs",
            "train_env_sizes",
        ]
        if c in df.columns
    ]

    long_rows: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        for col in env_metric_cols:
            value = row[col]
            if pd.isna(value):
                continue
            match = ENV_METRIC_RE.match(col)
            if match is None:
                continue
            long_rows.append(
                {
                    **{k: row[k] for k in base_cols},
                    "test_env": match.group("env"),
                    "metric": match.group("metric"),
                    "model_selection": match.group("ms_type"),
                    "value": value,
                }
            )

    return pd.DataFrame(long_rows)


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary_metrics = [
        c for c in [
            "worst_domain_acc_best",
            "avg_domain_acc_best",
            "best_domain_acc_best",
            "worst_domain_acc_final",
            "avg_domain_acc_final",
            "best_domain_acc_final",
        ]
        if c in df.columns
    ]
    if not summary_metrics:
        return pd.DataFrame()

    group_cols = [
        c for c in [
            "phase",
            "algorithm",
            "exp_name",
            "n_train_domains",
            "sample_size_per_domain",
            "imbalance_type",
        ]
        if c in df.columns
    ]

    grouped = (
        df.groupby(group_cols)[summary_metrics]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    # Flatten MultiIndex columns produced by agg.
    grouped.columns = [
        col if isinstance(col, str) else "_".join([part for part in col if part])
        for col in grouped.columns
    ]
    return grouped


def export_results(input_dir: str, output_dir: str, prefix: str) -> Tuple[str, List[str]]:
    os.makedirs(output_dir, exist_ok=True)
    records = load_records(input_dir)
    if not records:
        raise ValueError(f"No JSONL records found under: {input_dir}")

    df = pd.DataFrame.from_records(records)

    if "train_envs" in df.columns:
        df["train_envs"] = df["train_envs"].apply(
            lambda x: json.dumps(x) if isinstance(x, list) else x
        )
    if "train_env_sizes" in df.columns:
        df["train_env_sizes"] = df["train_env_sizes"].apply(
            lambda x: json.dumps(x) if isinstance(x, list) else x
        )

    flattened_df = flatten_args_column(df.copy())
    long_df = build_env_metric_long(df)
    summary_df = build_summary(df)
    tagged_prefix = with_standardized_prefix(prefix, df)

    written_paths: List[str] = []

    run_level_path = os.path.join(output_dir, f"{tagged_prefix}_run_level.csv")
    flattened_df.to_csv(run_level_path, index=False)
    written_paths.append(run_level_path)

    if not long_df.empty:
        long_path = os.path.join(output_dir, f"{tagged_prefix}_env_metric_long.csv")
        long_df.to_csv(long_path, index=False)
        written_paths.append(long_path)

    if not summary_df.empty:
        summary_path = os.path.join(output_dir, f"{tagged_prefix}_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        written_paths.append(summary_path)

    return run_level_path, written_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Export CMNIST results JSONL files to CSV")
    parser.add_argument(
        "input_dir",
        nargs="?",
        default="../cmnist_exp_small/results",
        help="Path to results root containing phase subfolders with JSONL files.",
    )
    parser.add_argument(
        "--output_dir",
        default="../results/export",
        help="Directory where reportable CSV files are written.",
    )
    parser.add_argument(
        "--prefix",
        default="cmnist_exp_small",
        help="Filename prefix for generated CSVs.",
    )
    args = parser.parse_args()

    _, written_paths = export_results(args.input_dir, args.output_dir, args.prefix)
    print("Wrote CSV files:")
    for path in written_paths:
        print(path)


if __name__ == "__main__":
    main()
