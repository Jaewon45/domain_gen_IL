#!/usr/bin/env python3
"""Create clearly labeled seed-0 pilot figures from four-anchor evaluations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ALGORITHMS = ["erm", "groupdro", "inftask", "iro"]
LABELS = {"erm": "ERM", "groupdro": "GroupDRO", "inftask": "INF-TASK", "iro": "IRO"}
COLORS = {"erm": "#4C78A8", "groupdro": "#F58518", "inftask": "#54A24B", "iro": "#E45756"}
ANCHORS = ["gaussian_noise", "defocus_blur", "snow", "contrast"]
E3_ORDER = ["balanced", "mild_imbalance", "strong_imbalance"]
E3B_ORDER = ["balanced", "long_tail", "near_missing", "missing"]


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_records(runs_root: Path):
    records = []
    for path in sorted(runs_root.glob("E3*/evaluation_pilot_anchor100/evaluation.jsonl")):
        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(lines) != 1:
            raise ValueError(f"Expected one lambda record in {path}, found {len(lines)}")
        record = lines[0]
        scope = record.get("evaluation_scope", {})
        if scope.get("kind") != "pilot" or scope.get("max_eval_images") != 100:
            raise ValueError(f"Unexpected evaluation scope in {path}: {scope}")
        if scope.get("corruption_types") != ANCHORS or int(record["seed"]) != 0:
            raise ValueError(f"Unexpected anchors or seed in {path}")
        records.append(record)
    if len(records) != 28:
        raise ValueError(f"Expected 28 E3/E3b pilot evaluations, found {len(records)}")
    return records


def per_type(record, name):
    return next(row for row in record["per_type"] if row["corruption"] == name)


def plot_lines(rows, conditions, value_key, ylabel, title, output):
    positions = np.arange(len(conditions))
    fig, axis = plt.subplots(figsize=(10, 6.5))
    for algorithm in ALGORITHMS:
        lookup = {row["condition"]: row for row in rows if row["algorithm"] == algorithm}
        values = [float(lookup[condition][value_key]) for condition in conditions]
        axis.plot(positions, values, marker="o", linewidth=2.5, markersize=7,
                  label=LABELS[algorithm], color=COLORS[algorithm])
    axis.set_xticks(positions, [value.replace("_", " ") for value in conditions])
    axis.set_ylabel(ylabel)
    axis.set_xlabel("Source condition")
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot ImageNet-100-C four-anchor pilot results")
    parser.add_argument("--runs_root", default="results/imagenet100c_seed0")
    parser.add_argument("--output_dir", default="results_submit_img100")
    args = parser.parse_args()
    runs_root, output = Path(args.runs_root), Path(args.output_dir)
    records = load_records(runs_root)
    e3b_rows = []
    e3_rows = []
    interval_rows = []
    for record in records:
        row = {
            "seed": int(record["seed"]),
            "condition": record["condition"],
            "algorithm": record["algorithm"],
            "designated_tail_type": "contrast",
            "designated_tail_accuracy_percent": 100.0 * float(per_type(record, "contrast")["severity_averaged_top1"]),
            "worst_anchor_accuracy_percent": 100.0 * (1.0 - float(record["worst_domain_error"])),
            "mean_anchor_accuracy_percent": 100.0 * (1.0 - float(record["mean_domain_error"])),
            "alpha_0.9_anchor_cvar_error": float(record["domain_cvar_error"]["0.9"]),
            "pilot_images": 100,
            "seed_count": 1,
            "sd_available": False,
        }
        if record["experiment"] == "E3b":
            e3b_rows.append(row)
            interval = record["partial_identification"]["0.9"]
            interval_rows.append({
                "seed": int(record["seed"]),
                "condition": record["condition"],
                "algorithm": record["algorithm"],
                "alpha": 0.9,
                "lower": interval["lower"],
                "realized_deployment_cvar": interval["realized_held_out_deployment_cvar_descriptive_plugin"],
                "upper": interval["upper"],
                "width": interval["width"],
                "epsilon": interval["epsilon"],
                "upper_endpoint_expected_vacuous": interval["upper_endpoint_expected_vacuous"],
            })
        elif record["experiment"] == "E3":
            e3_rows.append(row)
    e3b_rows.sort(key=lambda row: (E3B_ORDER.index(row["condition"]), ALGORITHMS.index(row["algorithm"])))
    e3_rows.sort(key=lambda row: (E3_ORDER.index(row["condition"]), ALGORITHMS.index(row["algorithm"])))
    interval_rows.sort(key=lambda row: (E3B_ORDER.index(row["condition"]), ALGORITHMS.index(row["algorithm"])))
    write_csv(output / "tables" / "E3b_tail_support" / "pilot_anchor100_performance.csv", e3b_rows)
    write_csv(output / "tables" / "E3b_tail_support" / "pilot_anchor100_intervals_alpha0.9.csv", interval_rows)
    write_csv(output / "tables" / "E3_imbalance" / "pilot_anchor100_worst_domain.csv", e3_rows)
    plot_lines(
        e3b_rows, E3B_ORDER, "designated_tail_accuracy_percent", "Designated-tail accuracy (%)",
        "E3b four-anchor pilot — contrast tail, seed 0",
        output / "figures" / "E3b_tail_support" / "tail_accuracy_by_condition_pilot_anchor100.png",
    )
    plot_lines(
        e3b_rows, E3B_ORDER, "worst_anchor_accuracy_percent", "Worst-anchor accuracy (%)",
        "E3b four-anchor pilot — seed 0",
        output / "figures" / "E3b_tail_support" / "worst_accuracy_by_condition_pilot_anchor100.png",
    )
    plot_lines(
        e3_rows, E3_ORDER, "worst_anchor_accuracy_percent", "Worst-anchor accuracy (%)",
        "E3 visible-imbalance four-anchor pilot — seed 0",
        output / "figures" / "E3_imbalance" / "e3_imbalance_worst_domain_accuracy_pilot_anchor100.png",
    )
    missing = [row for row in interval_rows if row["condition"] == "missing"]
    positions = np.arange(len(ALGORITHMS))
    width = 0.24
    fig, axis = plt.subplots(figsize=(10, 6.5))
    for offset, key, label, color in (
        (-width, "lower", "Lower endpoint", "#4C78A8"),
        (0.0, "realized_deployment_cvar", "Realized pilot CVaR", "#F58518"),
        (width, "upper", "Upper endpoint", "#E45756"),
    ):
        lookup = {row["algorithm"]: row for row in missing}
        axis.bar(positions + offset, [lookup[name][key] for name in ALGORITHMS], width=width, label=label, color=color)
    axis.set_xticks(positions, [LABELS[name] for name in ALGORITHMS])
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("CVaR error")
    axis.set_title("E3b missing-support identification interval — alpha=0.9, pilot seed 0")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    interval_path = output / "figures" / "E3b_tail_support" / "cvar_interval_missing_alpha0.9_pilot_anchor100.png"
    interval_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(interval_path, dpi=300)
    plt.close(fig)
    metadata = {
        "schema_version": 1,
        "scope": "pilot",
        "seed": 0,
        "seed_count": 1,
        "validation_images": 100,
        "validation_sampling": "one image per ImageNet-100 class",
        "corruption_types": ANCHORS,
        "severities": [1, 2, 3, 4, 5],
        "evaluated_checkpoints": len(records),
        "limitations": [
            "Single seed; no cross-seed standard deviation.",
            "One validation image per class; metrics are high-variance pilot estimates.",
            "Worst-domain means worst among the four registered anchors, not all 15 corruption types.",
            "Not report-grade and must not be mixed with full evaluation results.",
        ],
    }
    metadata_path = output / "metadata" / "pilot_anchor100_summary.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
