#!/usr/bin/env python3
"""Build a submission-oriented bundle from completed ImageNet-100-C runs.

This builder deliberately separates training diagnostics from deployment
evaluation.  It can package completed training runs immediately and will index
evaluation JSONL files when they exist, but it never labels source-training
losses as held-out corruption robustness.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ALGORITHM_ORDER = ["erm", "groupdro", "inftask", "iro"]
ALGORITHM_LABELS = {
    "erm": "ERM",
    "groupdro": "GroupDRO",
    "inftask": "INF-TASK",
    "iro": "IRO",
}
ALGORITHM_COLORS = {
    "erm": "#4C78A8",
    "groupdro": "#F58518",
    "inftask": "#54A24B",
    "iro": "#E45756",
}
EXPERIMENT_DIRS = {
    "E0": "E0_baseline",
    "E1": "E1_domain_count",
    "E2": "E2_sample_size",
    "E3": "E3_imbalance",
    "E3b": "E3b_tail_support",
    "severity_support": "severity_support",
}
CONDITION_ORDER = {
    "E3": ["balanced", "mild_imbalance", "strong_imbalance"],
    "E3b": ["balanced", "long_tail", "near_missing", "missing"],
}


def write_csv(path: Path, rows: List[Mapping[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else float("nan")


def independent_value(manifest: Mapping[str, object]) -> object:
    experiment = str(manifest["experiment"])
    counts = {str(key): int(value) for key, value in manifest["active_domain_counts"].items()}
    if experiment == "E1":
        return len(counts)
    if experiment == "E2":
        unique_counts = sorted(set(counts.values()))
        return unique_counts[0] if len(unique_counts) == 1 else sum(counts.values())
    if experiment in {"E3", "E3b"}:
        return str(manifest["condition"])
    if experiment == "severity_support":
        return "severity_domains"
    return "baseline"


def load_runs(runs_root: Path):
    summaries: List[Dict[str, object]] = []
    histories: List[Dict[str, object]] = []
    artifacts: List[Dict[str, object]] = []
    for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        manifest_path = run_dir / "manifest.json"
        history_path = run_dir / "history.jsonl"
        checkpoint_path = run_dir / "checkpoints" / "final.pt"
        missing = [str(path) for path in (manifest_path, history_path, checkpoint_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Incomplete run {run_dir.name}; missing: {missing}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(records) != int(manifest["run_configuration"]["steps"]):
            raise ValueError(f"History length mismatch for {run_dir.name}: {len(records)}")
        algorithm = str(manifest["run_configuration"]["algorithm"])
        experiment = str(manifest["experiment"])
        value = independent_value(manifest)
        enriched = []
        for record in records:
            environment_values = [float(item) for item in record["environment_losses"].values()]
            row = {
                "run": run_dir.name,
                "manifest_id": manifest["manifest_id"],
                "seed": int(manifest["global_seed"]),
                "experiment": experiment,
                "condition": str(manifest["condition"]),
                "independent_value": value,
                "algorithm": algorithm,
                "step": int(record["step"]),
                "objective": float(record["objective"]),
                "mean_logged_source_ce": mean(environment_values),
                "worst_logged_source_ce": max(environment_values),
            }
            histories.append(row)
            enriched.append((record, row))
        tail = enriched[-100:]
        final_record, final_row = enriched[-1]
        beta = final_record.get("adaptive_beta") or {}
        summaries.append({
            "run": run_dir.name,
            "manifest_id": manifest["manifest_id"],
            "seed": int(manifest["global_seed"]),
            "experiment": experiment,
            "condition": str(manifest["condition"]),
            "protocol": str(manifest["protocol"]),
            "independent_value": value,
            "algorithm": algorithm,
            "backbone_mode": manifest["run_configuration"]["backbone_mode"],
            "steps": len(records),
            "batch_size": int(manifest["run_configuration"]["batch_size"]),
            "num_source_domains": len(manifest["active_domain_counts"]),
            "total_source_images": sum(int(item) for item in manifest["active_domain_counts"].values()),
            "final_objective": final_row["objective"],
            "last100_mean_objective": mean(item[1]["objective"] for item in tail),
            "minimum_objective": min(float(item[1]["objective"]) for item in enriched),
            "final_mean_logged_source_ce": final_row["mean_logged_source_ce"],
            "final_worst_logged_source_ce": final_row["worst_logged_source_ce"],
            "last100_mean_logged_source_ce": mean(item[1]["mean_logged_source_ce"] for item in tail),
            "last100_mean_worst_logged_source_ce": mean(item[1]["worst_logged_source_ce"] for item in tail),
            "adaptive_beta_a_final": beta.get("a", ""),
            "adaptive_beta_b_final": beta.get("b", ""),
            "checkpoint_selection": manifest["run_configuration"]["checkpoint_selection"],
            "checkpoint_path": checkpoint_path.resolve().as_posix(),
            "history_path": history_path.resolve().as_posix(),
            "manifest_path": manifest_path.resolve().as_posix(),
        })
        for kind, path in (("manifest", manifest_path), ("history", history_path), ("checkpoint", checkpoint_path)):
            artifacts.append({
                "run": run_dir.name,
                "seed": int(manifest["global_seed"]),
                "experiment": experiment,
                "condition": str(manifest["condition"]),
                "algorithm": algorithm,
                "artifact_type": kind,
                "path": path.resolve().as_posix(),
                "bytes": path.stat().st_size,
            })
    return summaries, histories, artifacts


def plot_condition_summary(rows: List[Mapping[str, object]], experiment: str, output: Path) -> None:
    subset = [row for row in rows if row["experiment"] == experiment]
    if not subset:
        return
    if experiment == "E1":
        x_values = [2, 4, 8, 12]
        labels = [str(value) for value in x_values]
        x_label = "Number of source corruption types"
    elif experiment == "E2":
        x_values = [1000, 5000, 10000]
        labels = [f"{value:,}" for value in x_values]
        x_label = "Source images per corruption type"
    elif experiment in CONDITION_ORDER:
        x_values = CONDITION_ORDER[experiment]
        labels = [str(value).replace("_", " ") for value in x_values]
        x_label = "Source condition"
    else:
        x_values = ["baseline" if experiment == "E0" else "severity_domains"]
        labels = ["baseline" if experiment == "E0" else "severity support"]
        x_label = "Configuration"
    positions = np.arange(len(x_values), dtype=float)
    fig, axis = plt.subplots(figsize=(8.5, 5.2))
    if len(x_values) == 1:
        width = 0.18
        for index, algorithm in enumerate(ALGORITHM_ORDER):
            match = [row for row in subset if row["algorithm"] == algorithm]
            if match:
                axis.bar(
                    positions[0] + (index - 1.5) * width,
                    float(match[0]["last100_mean_logged_source_ce"]),
                    width=width,
                    label=ALGORITHM_LABELS[algorithm],
                    color=ALGORITHM_COLORS[algorithm],
                )
    else:
        for algorithm in ALGORITHM_ORDER:
            algorithm_rows = {str(row["independent_value"]): row for row in subset if row["algorithm"] == algorithm}
            values = [float(algorithm_rows[str(value)]["last100_mean_logged_source_ce"]) for value in x_values]
            axis.plot(
                positions, values, marker="o", linewidth=2, label=ALGORITHM_LABELS[algorithm],
                color=ALGORITHM_COLORS[algorithm],
            )
    axis.set_xticks(positions, labels)
    axis.set_xlabel(x_label)
    axis.set_ylabel("Mean logged source CE (last 100 steps)")
    axis.set_title(f"{experiment}: seed-0 training diagnostic")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_e0_convergence(histories: List[Mapping[str, object]], output: Path) -> None:
    fig, axis = plt.subplots(figsize=(8.5, 5.2))
    for algorithm in ALGORITHM_ORDER:
        rows = [row for row in histories if row["experiment"] == "E0" and row["algorithm"] == algorithm]
        if not rows:
            continue
        steps = np.asarray([int(row["step"]) for row in rows])
        values = np.asarray([float(row["objective"]) for row in rows])
        window = 25
        smoothed = np.convolve(values, np.ones(window) / window, mode="valid")
        axis.plot(steps[window - 1 :], smoothed, linewidth=2, label=ALGORITHM_LABELS[algorithm], color=ALGORITHM_COLORS[algorithm])
    axis.set_xlabel("Training step")
    axis.set_ylabel("Algorithm objective (25-step moving mean)")
    axis.set_title("E0 convergence diagnostic — seed 0")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def build_readme(output: Path, runs_root: Path, summaries: List[Mapping[str, object]], evaluation_count: int) -> None:
    seeds = sorted({int(row["seed"]) for row in summaries})
    text = f"""# ImageNet-100-C Submission Results

This bundle was generated from `{runs_root.resolve().as_posix()}`.

## Scope

- Completed training runs: **{len(summaries)}/64**
- Seeds: **{', '.join(map(str, seeds))}** (one seed only)
- Backbone: ImageNet-1k-pretrained ResNet-50 V2 with the last stage fine-tuned
- Checkpoint policy: fixed 1,000-step budget, final checkpoint
- Evaluation JSONL files discovered: **{evaluation_count}**

## Critical interpretation

The current tables and figures are **training diagnostics**, derived from source-domain
losses logged during optimization. They are not held-out ImageNet-100-C deployment
results and must not be reported as clean accuracy, corruption accuracy, worst-domain
error, CVaR, or partial-identification evidence.

Final scientific tables require running `IMAGENET100C.evaluate` on the checkpoints.
That evaluation covers clean validation plus all 15 corruption types at five severity
levels. `IMAGENET100C.analyze` can then create deployment `summary.csv` and
`robust_ranking.csv` tables.

## Layout

- `tables/training_run_summary.csv`: one row per completed run.
- `tables/training_history.csv`: long-form 64,000-step diagnostic table.
- `tables/<experiment>/training_summary.csv`: experiment-specific tables.
- `figures/<experiment>/source_training_diagnostic.png`: source-loss diagnostics.
- `figures/E0_baseline/convergence.png`: smoothed E0 training objectives.
- `metadata/artifact_index.csv`: paths and sizes of all raw training artifacts.
- `metadata/completion_summary.json`: completeness and provenance summary.
- `metadata/evaluation_inventory.csv`: discovered deployment evaluation files, if any.

Raw checkpoints are referenced rather than copied, avoiding a duplicate of roughly
20 GiB of model files.
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ImageNet-100-C submission artifacts")
    parser.add_argument("--runs_root", default="results/imagenet100c_seed0")
    parser.add_argument("--output_dir", default="results_submit_img100")
    args = parser.parse_args()
    runs_root = Path(args.runs_root)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    summaries, histories, artifacts = load_runs(runs_root)
    if len(summaries) != 64:
        raise ValueError(f"Expected 64 completed runs, found {len(summaries)}")
    output.mkdir(parents=True)
    write_csv(output / "tables" / "training_run_summary.csv", summaries)
    write_csv(output / "tables" / "training_history.csv", histories)
    write_csv(output / "metadata" / "artifact_index.csv", artifacts)
    for experiment, dirname in EXPERIMENT_DIRS.items():
        rows = [row for row in summaries if row["experiment"] == experiment]
        write_csv(output / "tables" / dirname / "training_summary.csv", rows)
        plot_condition_summary(rows, experiment, output / "figures" / dirname / "source_training_diagnostic.png")
    plot_e0_convergence(histories, output / "figures" / "E0_baseline" / "convergence.png")
    evaluation_files = sorted(runs_root.glob("**/evaluation.jsonl"))
    evaluation_inventory = [
        {"path": path.resolve().as_posix(), "bytes": path.stat().st_size} for path in evaluation_files
    ]
    if evaluation_inventory:
        write_csv(output / "metadata" / "evaluation_inventory.csv", evaluation_inventory)
    else:
        (output / "metadata").mkdir(parents=True, exist_ok=True)
        (output / "metadata" / "evaluation_inventory.csv").write_text("path,bytes\n", encoding="utf-8")
    completion = {
        "schema_version": 1,
        "runs_root": runs_root.resolve().as_posix(),
        "completed_runs": len(summaries),
        "expected_runs": 64,
        "seeds": sorted({int(row["seed"]) for row in summaries}),
        "algorithms": ALGORITHM_ORDER,
        "experiments": {key: sum(row["experiment"] == key for row in summaries) for key in EXPERIMENT_DIRS},
        "evaluation_jsonl_count": len(evaluation_files),
        "deployment_evaluation_complete": len(evaluation_files) == len(summaries),
        "bundle_scope": "source-training diagnostics; deployment evaluation pending",
    }
    (output / "metadata" / "completion_summary.json").write_text(
        json.dumps(completion, indent=2) + "\n", encoding="utf-8"
    )
    build_readme(output, runs_root, summaries, len(evaluation_files))
    print(json.dumps(completion, indent=2))


if __name__ == "__main__":
    main()
