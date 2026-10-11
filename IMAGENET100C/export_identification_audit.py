#!/usr/bin/env python3
"""Export the ImageNet-100-C missing-support identification audit table."""

from __future__ import annotations

import argparse
import glob
import json
import itertools
from pathlib import Path

import pandas as pd


ALPHAS = (0.5, 0.75, 0.9)
EXPECTED_METHODS = {"erm", "irm", "vrex", "eqrm", "groupdro", "inftask", "iro"}


def read_records(patterns: list[str]) -> list[dict]:
    records = []
    for pattern in patterns:
        matches = [Path(pattern)] if Path(pattern).is_file() else [Path(value) for value in glob.glob(pattern, recursive=True)]
        for path in matches:
            with path.open("r", encoding="utf-8") as handle:
                records.extend(json.loads(line) for line in handle if line.strip())
    return records


def build_run_level(records: list[dict], condition: str, evaluation_lambda: float) -> pd.DataFrame:
    rows = []
    for record in records:
        if record.get("condition") != condition or float(record.get("lambda")) != evaluation_lambda:
            continue
        for alpha, interval in record.get("partial_identification", {}).items():
            alpha_value = float(alpha)
            if alpha_value in ALPHAS:
                rows.append({
                    "condition": condition,
                    "algorithm": str(record["algorithm"]),
                    "seed": int(record["seed"]),
                    "alpha": alpha_value,
                    "lower": float(interval["lower"]),
                    "upper": float(interval["upper"]),
                    "width": float(interval["width"]),
                    "epsilon": float(interval["epsilon"]),
                    "lambda": evaluation_lambda,
                })
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise FileNotFoundError(f"No identification records found for condition={condition!r}")
    if set(frame["algorithm"]) != EXPECTED_METHODS:
        raise ValueError(f"Expected methods {sorted(EXPECTED_METHODS)}, found {sorted(frame['algorithm'].unique())}")
    if set(frame["alpha"]) != set(ALPHAS):
        raise ValueError(f"Expected alpha grid {ALPHAS}, found {sorted(frame['alpha'].unique())}")
    expected = len(EXPECTED_METHODS) * frame["seed"].nunique() * len(ALPHAS)
    if len(frame) != expected or frame.duplicated(["algorithm", "seed", "alpha"]).any():
        raise ValueError("ImageNet identification audit is incomplete or duplicated.")
    if not frame["epsilon"].eq(0.25).all():
        raise ValueError("The ImageNet missing-support audit must use epsilon=1/4.")
    return frame.sort_values(["algorithm", "seed", "alpha"]).reset_index(drop=True)


def summarize_intervals(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["algorithm", "alpha"], as_index=False)[["lower", "upper", "width"]]
        .mean()
        .sort_values(["algorithm", "alpha"])
    )


def summarize_audit(frame: pd.DataFrame) -> pd.DataFrame:
    methods = sorted(frame["algorithm"].unique())
    seeds = sorted(frame["seed"].unique())
    total_pairs = len(seeds) * len(list(itertools.combinations(methods, 2)))
    if len(methods) != 7 or len(seeds) != 3 or total_pairs != 63:
        raise ValueError("ImageNet audit requires seven methods and three seeds (63 pairs).")
    rows = []
    for alpha in ALPHAS:
        current = frame[frame["alpha"].eq(alpha)].set_index(["algorithm", "seed"])
        certified = 0
        for seed in seeds:
            for first, second in itertools.combinations(methods, 2):
                left = current.loc[(first, seed)]
                right = current.loc[(second, seed)]
                certified += int(left["upper"] < right["lower"] or right["upper"] < left["lower"])
        rows.append({
            "alpha": alpha,
            "mean_width": float(frame.loc[frame["alpha"].eq(alpha), "width"].mean()),
            "certified_pairs": certified,
            "total_pairs": total_pairs,
            "epsilon": float(frame["epsilon"].iloc[0]),
            "seeds": len(seeds),
            "methods": len(methods),
        })
    return pd.DataFrame(rows)


def write_tex(summary: pd.DataFrame, output: Path) -> None:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{\\textbf{ImageNet-100-C missing-support identification audit.} Entries are three-seed plug-in summaries using domain-level $0$--$1$ loss and the declared missing mass $\\epsilon=1/4$. A pair is certified only when its same-seed completion intervals are disjoint; $63=\\binom{7}{2}\\times3$.}",
        "\\label{tab:imagenet100c-identification-audit}",
        "\\begin{tabular}{@{}cccc@{}}",
        "\\toprule",
          "$\\alpha$ & Mean width & Certified / total \\\\",
        "\\midrule",
    ]
    for row in summary.itertuples(index=False):
          lines.append(f"{row.alpha:.2f} & {row.mean_width:.4f} & {int(row.certified_pairs)} / {int(row.total_pairs)} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--condition", default="missing")
    parser.add_argument("--lambda", dest="evaluation_lambda", type=float, default=0.9)
    args = parser.parse_args()
    frame = build_run_level(read_records(args.inputs), args.condition, args.evaluation_lambda)
    interval_summary = summarize_intervals(frame)
    audit_summary = summarize_audit(frame)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "imagenet100c_missing_tail_intervals_run_level.csv", index=False)
    interval_summary.to_csv(args.output_dir / "imagenet100c_missing_tail_intervals_summary.csv", index=False)
    audit_summary.to_csv(args.output_dir / "imagenet100c_identification_audit_summary.csv", index=False)
    write_tex(audit_summary, args.output_dir / "imagenet100c_identification_audit.tex")
    print(f"Runs: {frame[['algorithm', 'seed']].drop_duplicates().shape[0]}")
    print(f"Wrote {args.output_dir / 'imagenet100c_identification_audit.tex'}")


if __name__ == "__main__":
    main()