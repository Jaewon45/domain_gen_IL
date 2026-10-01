#!/usr/bin/env python3
"""Merge the matched VREx/EQRM E3b export with the frozen five-method table."""

import argparse
from pathlib import Path

import pandas as pd


CONDITIONS = {"balanced_visible", "long_tail_visible", "near_missing_tail", "missing_tail"}
EXISTING_METHODS = {"erm", "irm", "groupdro", "inftask", "iro"}
NEW_METHODS = {"vrex", "eqrm"}
SEEDS = {0, 1, 2, 3, 4}


def validate(frame: pd.DataFrame, methods: set[str], label: str, require_seeds: bool) -> None:
    required = {"condition", "algorithm"}
    if require_seeds:
        required.add("seed")
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")
    observed_methods = set(frame["algorithm"].astype(str).str.lower())
    if observed_methods != methods:
        raise ValueError(f"{label} methods {sorted(observed_methods)} != {sorted(methods)}")
    observed_conditions = set(frame["condition"].astype(str))
    if observed_conditions != CONDITIONS:
        raise ValueError(f"{label} conditions {sorted(observed_conditions)} != {sorted(CONDITIONS)}")
    if require_seeds:
        observed_seeds = set(pd.to_numeric(frame["seed"]).astype(int))
        if observed_seeds != SEEDS:
            raise ValueError(f"{label} seeds {sorted(observed_seeds)} != {sorted(SEEDS)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing_raw", required=True)
    parser.add_argument("--existing_summary", required=True)
    parser.add_argument("--new_raw", required=True)
    parser.add_argument("--new_summary", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    existing_raw = pd.read_csv(args.existing_raw)
    new_raw = pd.read_csv(args.new_raw)
    existing_summary = pd.read_csv(args.existing_summary)
    new_summary = pd.read_csv(args.new_summary)
    validate(existing_raw, EXISTING_METHODS, "existing raw export", require_seeds=True)
    validate(new_raw, NEW_METHODS, "new raw export", require_seeds=True)
    validate(existing_summary, EXISTING_METHODS, "existing summary", require_seeds=False)
    validate(new_summary, NEW_METHODS, "new summary", require_seeds=False)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_raw = pd.concat([existing_raw, new_raw], ignore_index=True)
    combined_summary = pd.concat([existing_summary, new_summary], ignore_index=True)
    combined_raw.to_csv(output_dir / "raw_results_7methods.csv", index=False)
    combined_summary.to_csv(output_dir / "summary_by_condition_7methods.csv", index=False)
    print(f"Wrote {output_dir / 'raw_results_7methods.csv'}")
    print(f"Wrote {output_dir / 'summary_by_condition_7methods.csv'}")


if __name__ == "__main__":
    main()
