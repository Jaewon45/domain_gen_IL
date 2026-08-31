#!/usr/bin/env python3
"""Audit CMNIST result rows without modifying source artifacts."""

import argparse
import ast
import csv
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from collect_results import enrich_record


def parse_value(value):
    if isinstance(value, (list, dict, tuple, int, float, bool)) or value is None:
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def canonical_run_key(record):
    args = dict(record.get("args", {}))
    key_data = {
        "algorithm": record.get("algorithm", args.get("algorithm")),
        "seed": record.get("seed", args.get("seed")),
        "exp_name": args.get("exp_name"),
        "train_env_ps": parse_value(args.get("train_env_ps")),
        "train_env_sizes_parsed": parse_value(args.get("train_env_sizes_parsed")),
        "train_env_size_mode": args.get("train_env_size_mode"),
        "test_envs": args.get("test_envs"),
        "steps": args.get("steps"),
        "batch_size": args.get("batch_size"),
        "lr": args.get("lr"),
        "weight_decay": args.get("weight_decay"),
        "erm_pretrain_iters": args.get("erm_pretrain_iters"),
        "penalty_weight": args.get("penalty_weight"),
        "groupdro_eta": args.get("groupdro_eta"),
        "alpha": args.get("alpha"),
        "lr_cos_sched": args.get("lr_cos_sched"),
        "test_env_ms": args.get("test_env_ms"),
        "tail_support_condition": record.get("tail_support_condition") or args.get("tail_support_condition"),
        "model_selection": "best_and_final",
    }
    encoded = json.dumps(key_data, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16], key_data


def load_records(results_dir):
    records = []
    for path in sorted(Path(results_dir).rglob("*.jsonl")):
        if path.stat().st_size == 0:
            continue
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = enrich_record(json.loads(line))
                run_key, key_data = canonical_run_key(record)
                records.append({
                    "source_file": str(path),
                    "source_line": line_number,
                    "run_key": run_key,
                    "algorithm": record.get("algorithm"),
                    "seed": record.get("seed"),
                    "phase": record.get("phase"),
                    "exp_name": key_data["exp_name"],
                    "step": record.get("step"),
                    "has_best": any(key.endswith("_acc_best") for key in record),
                    "has_final": any(key.endswith("_acc_final") for key in record),
                    "record": record,
                })
    return records


def load_lambda_records(lambda_dir):
    records = []
    if not lambda_dir:
        return records
    for path in sorted(Path(lambda_dir).rglob("*.jsonl")):
        if path.stat().st_size == 0:
            continue
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    record = json.loads(line)
                    record["source_file"] = str(path)
                    record["source_line"] = line_number
                    records.append(record)
    return records


def audit_records(records):
    grouped = {}
    for record in records:
        grouped.setdefault(record["run_key"], []).append(record)

    rows = []
    for run_key, candidates in grouped.items():
        candidates = sorted(
            candidates,
            key=lambda item: (item["has_best"], item["has_final"], item["step"] or 0),
            reverse=True,
        )
        selected = candidates[0]
        status = "complete" if selected["has_best"] and selected["has_final"] else "partial"
        if len(candidates) > 1:
            status = "duplicate_or_recovered"
        row = dict(selected)
        row.pop("record", None)
        row["candidate_count"] = len(candidates)
        row["status"] = status
        row["candidate_files"] = ";".join(sorted({item["source_file"] for item in candidates}))
        rows.append(row)
    return sorted(rows, key=lambda item: (str(item["exp_name"]), str(item["algorithm"]), str(item["seed"]), item["run_key"]))


def selected_records(records):
    grouped = {}
    for record in records:
        grouped.setdefault(record["run_key"], []).append(record)
    selected = []
    for candidates in grouped.values():
        candidates = sorted(
            candidates,
            key=lambda item: (item["has_best"], item["has_final"], item["step"] or 0),
            reverse=True,
        )
        candidate = candidates[0]
        if candidate["has_best"] and candidate["has_final"]:
            selected.append(candidate["record"])
    return selected


def manifest_identity(command):
    values = {}
    for name in [
        "algorithm", "seed", "exp_name", "train_envs", "train_env_sizes", "steps",
        "batch_size", "lr", "erm_pretrain_iters", "penalty_weight", "groupdro_eta",
    ]:
        match = re.search(rf"--{name}\s+([^\s]+)", command)
        values[name] = match.group(1) if match else None
    return values


def record_identity(record):
    args = dict(record.get("args", {}))
    train_envs = args.get("train_env_ps")
    if isinstance(train_envs, (list, tuple)):
        train_envs = ",".join(str(float(value)) for value in train_envs)
    sizes = args.get("train_env_sizes_parsed")
    if isinstance(sizes, (list, tuple)):
        sizes = ",".join(str(int(value)) for value in sizes)
    identity = {
        "algorithm": record.get("algorithm", args.get("algorithm")),
        "seed": str(record.get("seed", args.get("seed"))),
        "exp_name": args.get("exp_name"),
        "train_envs": train_envs,
        "train_env_sizes": sizes,
        "steps": str(args.get("steps")),
        "batch_size": str(args.get("batch_size")),
        "lr": str(args.get("lr")),
        "erm_pretrain_iters": str(args.get("erm_pretrain_iters")),
        "penalty_weight": str(args.get("penalty_weight")),
        "groupdro_eta": str(args.get("groupdro_eta")),
    }
    for name in ["lr", "penalty_weight", "groupdro_eta"]:
        identity[name] = format(float(identity[name]), ".15g")
    return identity


def normalize_manifest_identity(identity):
    normalized = dict(identity)
    defaults = {
        "penalty_weight": "1000",
        "groupdro_eta": "1.0",
    }
    for name, value in defaults.items():
        if normalized[name] is None:
            normalized[name] = value
    for name in ["train_envs", "train_env_sizes"]:
        if normalized[name]:
            normalized[name] = ",".join(str(float(value)) if name == "train_envs" else str(int(value)) for value in normalized[name].split(","))
    for name in ["lr", "penalty_weight", "groupdro_eta"]:
        if normalized[name] is not None:
            normalized[name] = format(float(normalized[name]), ".15g")
    return normalized


def write_manifest_audit(manifests, records, output):
    record_keys = {tuple(record_identity(record).items()) for record in records}
    rows = []
    command_index = 0
    for manifest in manifests:
        with open(manifest, encoding="utf-8") as handle:
            for line in handle:
                command = line.strip()
                if not command or command.startswith("#"):
                    continue
                command_index += 1
                identity = normalize_manifest_identity(manifest_identity(command))
                rows.append({
                    "command_index": command_index,
                    "manifest": str(manifest),
                    "matched_clean_record": tuple(identity.items()) in record_keys,
                    "algorithm": identity["algorithm"],
                    "seed": identity["seed"],
                    "exp_name": identity["exp_name"],
                    "train_envs": identity["train_envs"],
                    "train_env_sizes": identity["train_env_sizes"],
                    "command": command,
                })
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["command_index", "manifest", "matched_clean_record"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_log_audit(log_dir, output):
    rows = []
    for path in sorted(Path(log_dir).rglob("*")):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        rows.append({
            "log_file": str(path),
            "size_bytes": path.stat().st_size,
            "traceback_count": text.count("Traceback"),
            "error_count": len(re.findall(r"(?im)^.*\b(error|exception)\b.*$", text)),
            "completion_marker_count": len(re.findall(r"(?im)^(saved|finished|smoke|wrote|generated).*", text)),
        })
    with output.open("w", newline="", encoding="utf-8") as handle:
        fields = ["log_file", "size_bytes", "traceback_count", "error_count", "completion_marker_count"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def audit_log_runs(log_path):
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"(?m)^Args:\s*$", text)[1:]
    rows = []
    for index, block in enumerate(blocks, 1):
        args = {}
        for line in block.splitlines():
            match = re.match(r"^\s{1,2}([^:]+):\s*(.*)$", line)
            if match:
                args[match.group(1).strip()] = match.group(2).strip()
        has_final = "final accuracies:" in block
        has_best = "best accuracies:" in block
        errors = re.findall(r"(?im)^.*\b(error|exception|traceback)\b.*$", block)
        status = "complete" if has_final and has_best and not errors else "failed_or_partial"
        rows.append({
            "log_sequence": index,
            "algorithm": args.get("algorithm"),
            "seed": args.get("seed"),
            "exp_name": args.get("exp_name"),
            "train_envs": args.get("train_envs"),
            "train_env_sizes": args.get("train_env_sizes"),
            "steps": args.get("steps"),
            "has_final_accuracies": has_final,
            "has_best_accuracies": has_best,
            "error_markers": len(errors),
            "status": status,
        })
    return rows


def write_log_run_audit(log_dir, output):
    rows = []
    for path in sorted(Path(log_dir).rglob("*.txt")):
        for row in audit_log_runs(path):
            row["log_file"] = str(path)
            rows.append(row)
    fields = [
        "log_file", "log_sequence", "algorithm", "seed", "exp_name",
        "train_envs", "train_env_sizes", "steps", "has_final_accuracies",
        "has_best_accuracies", "error_markers", "status",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_clean_metric_summary(clean_records, output):
    rows = []
    for record in clean_records:
        metadata = {
            "phase": record.get("phase"),
            "algorithm": record.get("algorithm"),
            "seed": record.get("seed"),
            "n_train_domains": record.get("n_train_domains"),
            "sample_size_per_domain": record.get("sample_size_per_domain"),
            "imbalance_type": record.get("imbalance_type"),
            "train_envs": json.dumps(record.get("train_envs"), sort_keys=True),
            "train_env_sizes": json.dumps(record.get("train_env_sizes"), sort_keys=True),
        }
        for key, value in record.items():
            if key.endswith("_acc_best") or key.endswith("_loss_best"):
                try:
                    metadata[key] = float(value)
                except (TypeError, ValueError):
                    pass
        rows.append(metadata)
    frame = pd.DataFrame(rows)
    metric_columns = [column for column in frame.columns if column.endswith("_acc_best") or column.endswith("_loss_best")]
    group_columns = [
        "phase", "algorithm", "n_train_domains", "sample_size_per_domain",
        "imbalance_type", "train_envs", "train_env_sizes",
    ]
    summary = frame.groupby(group_columns, dropna=False)[metric_columns].agg(["mean", "std", "count"]).reset_index()
    summary.columns = [
        column if isinstance(column, str) else "_".join(token for token in column if token)
        for column in summary.columns
    ]
    summary.to_csv(output, index=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir")
    parser.add_argument("--output", required=True, help="CSV audit path; existing files are not overwritten.")
    parser.add_argument("--clean_output", help="Optional directory for selected complete JSONL records.")
    parser.add_argument("--lambda_results_dir", help="Optional post-training lambda JSONL directory.")
    parser.add_argument("--log_dir", help="Optional log directory for aggregate error/completion evidence.")
    parser.add_argument("--manifest", action="append", default=[], help="Manifest to audit; may be repeated.")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit: {output}")
    records = load_records(args.results_dir)
    rows = audit_records(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_key", "algorithm", "seed", "phase", "exp_name", "step",
        "has_best", "has_final", "candidate_count", "status",
        "source_file", "source_line", "candidate_files",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fieldnames} for row in rows)

    records_count = sum(int(row["candidate_count"]) for row in rows)
    print(f"Read {records_count} result rows")
    print(f"Canonical runs: {len(rows)}")
    print(f"Wrote audit: {output}")

    if args.clean_output:
        clean_dir = Path(args.clean_output)
        if clean_dir.exists() and any(clean_dir.iterdir()):
            raise FileExistsError(f"Refusing to overwrite non-empty clean output: {clean_dir}")
        clean_dir.mkdir(parents=True, exist_ok=True)
        clean_records = selected_records(records)
        with (clean_dir / "clean_training_results.jsonl").open("w", encoding="utf-8") as handle:
            for record in clean_records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        summary = {}
        for record in clean_records:
            key = (record.get("algorithm"), record.get("seed"))
            summary[key] = summary.get(key, 0) + 1
        with (clean_dir / "clean_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["algorithm", "seed", "retained_records"])
            for (algorithm, seed), count in sorted(summary.items()):
                writer.writerow([algorithm, seed, count])
        print(f"Wrote {len(clean_records)} complete records to {clean_dir}")

        lambda_records = load_lambda_records(args.lambda_results_dir)
        with (clean_dir / "lambda_results.jsonl").open("w", encoding="utf-8") as handle:
            for record in lambda_records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(f"Wrote {len(lambda_records)} post-training lambda records to {clean_dir}")

    if args.manifest:
        manifest_output = output.with_name("manifest_audit.csv")
        manifest_rows = write_manifest_audit(args.manifest, selected_records(records), manifest_output)
        matched = sum(row["matched_clean_record"] for row in manifest_rows)
        print(f"Manifest commands matched to clean records: {matched}/{len(manifest_rows)}")
        print(f"Wrote manifest audit: {manifest_output}")

    if args.log_dir:
        log_output = output.with_name("log_audit.csv")
        log_rows = write_log_audit(args.log_dir, log_output)
        print(f"Audited {len(log_rows)} log files; wrote log audit: {log_output}")
        log_run_output = output.with_name("log_run_audit.csv")
        log_run_rows = write_log_run_audit(args.log_dir, log_run_output)
        print(f"Classified {len(log_run_rows)} log run blocks; wrote log-run audit: {log_run_output}")

    if args.clean_output:
        clean_metric_output = Path(args.clean_output) / "clean_metric_summary.csv"
        write_clean_metric_summary(clean_records, clean_metric_output)
        print(f"Wrote clean mean/std metric summary: {clean_metric_output}")


if __name__ == "__main__":
    main()
