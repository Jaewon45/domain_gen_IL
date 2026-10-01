#!/usr/bin/env python3
"""Export the combined CMNIST/ImageNet100C worst-environment-accuracy table."""

import argparse
import json
from pathlib import Path

import pandas as pd


METHODS = ["erm", "irm", "vrex", "eqrm", "groupdro", "inftask", "iro"]
DISPLAY = {
    "erm": "ERM", "irm": "IRM", "vrex": "VREx", "eqrm": "EQRM",
    "groupdro": "GroupDRO", "inftask": "INF-TASK", "iro": "IRO",
}
CMNIST_CONDITIONS = ["balanced_visible", "long_tail_visible", "near_missing_tail", "missing_tail"]
IMAGENET_CONDITIONS = ["balanced", "long_tail", "near_missing", "missing"]
CONDITION_DISPLAY = ["Balanced", "Long-tail", "Near-missing", "Missing"]


def cell(mean: float, std: float) -> str:
    return f"{100 * mean:.1f} $\\pm$ {100 * std:.1f}"


def imagenet_cell(root: Path, method: str, condition: str, seeds: list[int]) -> str | None:
    values = []
    for seed in seeds:
        path = root / f"seed{seed}" / f"E3b_{condition}_{method}" / "evaluation_minimal_anchor1000" / "evaluation.jsonl"
        if not path.is_file():
            return None
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(records) != 1:
            raise ValueError(f"Expected one record in {path}, found {len(records)}")
        values.append(1.0 - float(records[0]["worst_domain_error"]))
    series = pd.Series(values, dtype=float)
    return cell(float(series.mean()), float(series.std(ddof=1)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmnist_summary", required=True)
    parser.add_argument("--imagenet_root", required=True)
    parser.add_argument("--imagenet_seeds", default="1,2")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cmnist = pd.read_csv(args.cmnist_summary)
    seeds = [int(value) for value in args.imagenet_seeds.split(",")]
    if len(seeds) != 2:
        raise ValueError("This export is configured for exactly two ImageNet100C seeds.")
    lookup = {(str(row.algorithm), str(row.condition)): row for _, row in cmnist.iterrows()}
    rows, missing = [], []
    for method in METHODS:
        cmnist_values = []
        for condition in CMNIST_CONDITIONS:
            row = lookup[(method, condition)]
            cmnist_values.append(cell(float(row.worst_accuracy_mean), float(row.worst_accuracy_std)))
        imagenet_values = []
        for condition in IMAGENET_CONDITIONS:
            value = imagenet_cell(Path(args.imagenet_root), method, condition, seeds)
            if value is None:
                value = "[TBD]"
                missing.append(f"{method}/{condition}")
            imagenet_values.append(value)
        rows.append(" & ".join([DISPLAY[method], *cmnist_values, *imagenet_values]) + r" \\")

    missing_comment = "" if not missing else "% ImageNet100C unavailable cells: " + ", ".join(missing) + "\n"
    table = "\n".join([
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Worst-environment accuracy (\%) under source-support degradation. Entries are mean $\pm$ standard deviation over five CMNIST seeds and two ImageNet100C seeds.}",
        r"\label{tab:support-performance}",
        r"\begin{tabular}{@{}lcccccccc@{}}",
        r"\toprule",
        r"& \multicolumn{4}{c}{CMNIST (5 seeds)} & \multicolumn{4}{c}{ImageNet100C (2 seeds)} \\",
        r"\cmidrule(lr){2-5}\cmidrule(l){6-9}",
        r"Method & " + " & ".join(CONDITION_DISPLAY + CONDITION_DISPLAY) + r" \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(missing_comment + table)
    print(f"Wrote {output}")
    print(f"ImageNet100C unavailable cells: {len(missing)}")


if __name__ == "__main__":
    main()
