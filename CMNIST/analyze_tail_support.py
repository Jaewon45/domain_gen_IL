#!/usr/bin/env python3
"""Analyze CMNIST tail-support stress tests and export slide-ready artifacts."""

import argparse
import math
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from collect_results import load_records


DEFAULT_CONDITION_ORDER = [
    "balanced_visible",
    "long_tail_visible",
    "near_missing_tail",
    "missing_tail",
]
DEFAULT_ALPHAS = [0.5, 0.75, 0.9]

ALGORITHM_DISPLAY_NAMES = {
    "erm": "ERM",
    "irm": "IRM",
    "groupdro": "GroupDRO",
    "iro": "IRO",
    "inftask": "InfTask",
}

BRIGHT_BAR_COLORS = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#EECA3B",
    "#B279A2",
    "#FF9DA6",
]


def apply_plot_style() -> None:
    plt.rcParams.update({
        "font.size": 22,
        "axes.titlesize": 24,
        "axes.labelsize": 22,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "figure.titlesize": 24,
    })


def format_algorithm_name(name: str) -> str:
    key = str(name).strip().lower()
    return ALGORITHM_DISPLAY_NAMES.get(key, key.upper())


def reportable_text(value: str) -> str:
    text = str(value).replace("_", " ").strip()
    token_map = {
        "acc": "accuracy",
        "avg": "average",
        "lambda": "Lambda",
    }
    words: List[str] = []
    for token in text.split():
        lower = token.lower()
        if lower in token_map:
            words.append(token_map[lower])
            continue
        if lower in ALGORITHM_DISPLAY_NAMES:
            words.append(ALGORITHM_DISPLAY_NAMES[lower])
            continue
        words.append(token.capitalize())
    return " ".join(words)


def format_condition_tick(value: str) -> str:
    return reportable_text(str(value))


def parse_float_list(text: str) -> List[float]:
    return [float(token.strip()) for token in text.split(",") if token.strip()]


def parse_map(value) -> Dict[str, float]:
    if isinstance(value, dict):
        return {str(k): float(v) for k, v in value.items()}
    return {}


def parse_source_envs(row: pd.Series) -> List[float]:
    source_envs = row.get("tail_support_source_envs")
    if isinstance(source_envs, list):
        return [float(value) for value in source_envs]
    args = row.get("args", {}) or {}
    args_source = args.get("tail_support_source_envs_parsed")
    if isinstance(args_source, (list, tuple)):
        return [float(value) for value in args_source]
    train_envs = row.get("train_envs")
    if isinstance(train_envs, list):
        return [float(value) for value in train_envs]
    return [0.1, 0.2, 0.5, 0.9]


def parse_count_map(row: pd.Series, source_envs: List[float]) -> Dict[str, int]:
    count_map = row.get("tail_support_train_count_map")
    if isinstance(count_map, dict):
        return {str(k): int(v) for k, v in count_map.items()}

    sizes = row.get("train_env_sizes")
    derived = {str(env): 0 for env in source_envs}
    if isinstance(sizes, list):
        for env, size in zip(source_envs, sizes):
            derived[str(env)] = int(size)
    return derived


def infer_condition(row: pd.Series) -> str:
    condition = row.get("tail_support_condition")
    if isinstance(condition, str) and condition:
        return condition
    exp_name = str(row.get("exp_name", ""))
    for token in DEFAULT_CONDITION_ORDER:
        if token in exp_name:
            return token
    return "unknown_condition"


def infer_head_tail_envs(source_envs: List[float], count_map: Dict[str, int], row: pd.Series) -> Tuple[str, str]:
    head_env = row.get("tail_support_head_env")
    tail_env = row.get("tail_support_tail_env")
    if head_env is not None and tail_env is not None and not pd.isna(head_env) and not pd.isna(tail_env):
        return str(float(head_env)), str(float(tail_env))

    count_pairs = [(str(env), int(count_map.get(str(env), 0))) for env in source_envs]
    min_count = min(count for _, count in count_pairs)
    max_count = max(count for _, count in count_pairs)

    tail_candidates = [env for env, count in count_pairs if count == min_count]
    head_candidates = [env for env, count in count_pairs if count == max_count]

    # Tie-break: largest env value for tail, smallest for head.
    tail_env = sorted(tail_candidates, key=lambda x: float(x))[-1]
    head_env = sorted(head_candidates, key=lambda x: float(x))[0]
    return head_env, tail_env


def env_metric_columns(frame: pd.DataFrame, suffix: str) -> List[str]:
    cols = []
    for col in frame.columns:
        if not isinstance(col, str) or not col.endswith(suffix):
            continue
        env_token = col.split("_")[0]
        try:
            float(env_token)
        except ValueError:
            continue
        cols.append(col)
    return sorted(cols, key=lambda x: float(x.split("_")[0]))


def weighted_cvar(losses: np.ndarray, weights: np.ndarray, alpha: float) -> float:
    weights = np.asarray(weights, dtype=float)
    losses = np.asarray(losses, dtype=float)
    if np.all(weights == 0):
        return float("nan")

    weights = weights / weights.sum()
    order = np.argsort(losses)
    sorted_losses = losses[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    var_idx = int(np.searchsorted(cumulative, alpha, side="left"))
    var_idx = min(var_idx, len(sorted_losses) - 1)
    var_threshold = sorted_losses[var_idx]

    tail_mask = losses >= var_threshold
    tail_weights = weights[tail_mask]
    tail_losses = losses[tail_mask]
    tail_mass = tail_weights.sum()
    if tail_mass <= 0:
        return float(var_threshold)
    return float(np.dot(tail_losses, tail_weights) / tail_mass)


def metrics_from_rows(group_rows: pd.DataFrame, alphas: List[float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    domain_acc = group_rows["test_accuracy"].astype(float).to_numpy()
    domain_loss = group_rows["loss"].astype(float).to_numpy()
    empirical_weights = group_rows["empirical_weight"].astype(float).to_numpy()
    deployment_weights = group_rows["deployment_weight"].astype(float).to_numpy()

    out["avg_accuracy"] = float(np.nanmean(domain_acc))
    out["worst_accuracy"] = float(np.nanmin(domain_acc))

    tail_rows = group_rows[group_rows["is_tail_domain"] == 1]
    head_rows = group_rows[group_rows["is_head_domain"] == 1]
    out["tail_accuracy"] = float(tail_rows["test_accuracy"].iloc[0]) if not tail_rows.empty else float("nan")
    out["head_accuracy"] = float(head_rows["test_accuracy"].iloc[0]) if not head_rows.empty else float("nan")
    out["head_tail_gap"] = out["head_accuracy"] - out["tail_accuracy"]

    for alpha in alphas:
        suffix = str(alpha).replace(".", "")
        emp_cvar = weighted_cvar(domain_loss, empirical_weights, alpha)
        dep_cvar = weighted_cvar(domain_loss, deployment_weights, alpha)
        out[f"empirical_cvar_a{suffix}"] = emp_cvar
        out[f"deployment_cvar_a{suffix}"] = dep_cvar
        out[f"cvar_gap_a{suffix}"] = abs(dep_cvar - emp_cvar)

    return out


def build_raw_rows_from_main(frame: pd.DataFrame, eval_envs: List[float]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    acc_cols = env_metric_columns(frame, "_acc_best")
    envs = [float(col.split("_")[0]) for col in acc_cols]
    if eval_envs:
        envs = eval_envs

    for _, row in frame.iterrows():
        source_envs = parse_source_envs(row)
        count_map = parse_count_map(row, source_envs)
        empirical_total = float(sum(count_map.values()))

        empirical_prior = {
            str(env): (float(count_map.get(str(env), 0)) / empirical_total if empirical_total > 0 else 0.0)
            for env in envs
        }
        deployment_prior = {str(env): 1.0 / len(envs) for env in envs}

        condition = infer_condition(row)
        head_env, tail_env = infer_head_tail_envs(source_envs, count_map, row)

        for env in envs:
            env_key = str(env)
            acc_key = f"{env_key}_acc_best"
            loss_key = f"{env_key}_loss_best"
            if acc_key not in row or loss_key not in row:
                continue

            rows.append(
                {
                    "source": "main_eval",
                    "condition": condition,
                    "algorithm": row["algorithm"],
                    "seed": int(row["seed"]),
                    "lambda": np.nan,
                    "alpha_eval": np.nan,
                    "domain_id": env_key,
                    "train_count": int(count_map.get(env_key, 0)),
                    "test_accuracy": float(row[acc_key]),
                    "loss": float(row[loss_key]),
                    "empirical_weight": float(empirical_prior.get(env_key, 0.0)),
                    "deployment_weight": float(deployment_prior.get(env_key, 0.0)),
                    "head_domain_id": head_env,
                    "tail_domain_id": tail_env,
                    "is_head_domain": int(env_key == head_env),
                    "is_tail_domain": int(env_key == tail_env),
                }
            )

    return rows


def build_raw_rows_from_lambda(frame: pd.DataFrame, eval_envs: List[float]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    envs = eval_envs

    for _, row in frame.iterrows():
        condition = infer_condition(row)
        if condition == "unknown_condition":
            continue

        source_envs = parse_source_envs(row)
        count_map = parse_count_map(row, source_envs)
        empirical_total = float(sum(count_map.values()))
        empirical_prior = {
            str(env): (float(count_map.get(str(env), 0)) / empirical_total if empirical_total > 0 else 0.0)
            for env in envs
        }
        deployment_prior = {str(env): 1.0 / len(envs) for env in envs}
        head_env, tail_env = infer_head_tail_envs(source_envs, count_map, row)

        lambda_value = row.get("lambda_eval")
        for env in envs:
            env_key = str(env)
            acc_key = f"{env_key}_acc"
            loss_key = f"{env_key}_loss"
            if acc_key not in row or loss_key not in row:
                continue
            rows.append(
                {
                    "source": "lambda_eval",
                    "condition": condition,
                    "algorithm": row["algorithm"],
                    "seed": int(row["seed"]),
                    "lambda": float(lambda_value),
                    "alpha_eval": np.nan,
                    "domain_id": env_key,
                    "train_count": int(count_map.get(env_key, 0)),
                    "test_accuracy": float(row[acc_key]),
                    "loss": float(row[loss_key]),
                    "empirical_weight": float(empirical_prior.get(env_key, 0.0)),
                    "deployment_weight": float(deployment_prior.get(env_key, 0.0)),
                    "head_domain_id": head_env,
                    "tail_domain_id": tail_env,
                    "is_head_domain": int(env_key == head_env),
                    "is_tail_domain": int(env_key == tail_env),
                }
            )

    return rows


def ensure_condition_order(frame: pd.DataFrame) -> pd.DataFrame:
    categories = [c for c in DEFAULT_CONDITION_ORDER if c in frame["condition"].unique().tolist()]
    rest = [c for c in frame["condition"].unique().tolist() if c not in categories]
    frame = frame.copy()
    frame["condition"] = pd.Categorical(frame["condition"], categories=categories + sorted(rest), ordered=True)
    return frame


def plot_metric(summary: pd.DataFrame, metric: str, out_path: str, title: str, ylabel: str) -> None:
    plot_df = summary[summary["lambda_label"] == "none"].copy()
    if plot_df.empty:
        return
    plot_df = ensure_condition_order(plot_df)
    pivot = plot_df.pivot(index="condition", columns="algorithm", values=f"{metric}_mean")
    if pivot.empty:
        return

    pivot.columns = [format_algorithm_name(column) for column in pivot.columns]
    bar_colors = BRIGHT_BAR_COLORS[:len(pivot.columns)]
    ax = pivot.plot(kind="bar", figsize=(11, 6), color=bar_colors)
    ax.set_title(reportable_text(title))
    ax.set_xlabel("Condition")
    ax.set_ylabel(reportable_text(ylabel))
    ax.set_xticklabels([format_condition_tick(value) for value in pivot.index], rotation=0, ha="center")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_iro_lambda(raw_df: pd.DataFrame, metric: str, out_path: str, title: str, ylabel: str, alpha_suffix: str) -> None:
    lambda_rows = raw_df[
        (raw_df["algorithm"] == "iro")
        & (raw_df["source"] == "lambda_eval")
        & (raw_df["lambda"].notna())
    ].copy()
    if lambda_rows.empty:
        return

    group_cols = ["condition", "seed", "lambda"]
    metric_rows = []
    for (condition, seed, lambda_value), group in lambda_rows.groupby(group_cols):
        metrics = metrics_from_rows(group, DEFAULT_ALPHAS)
        metric_rows.append(
            {
                "condition": condition,
                "seed": seed,
                "lambda": lambda_value,
                "tail_accuracy": metrics["tail_accuracy"],
                "deployment_cvar": metrics[f"deployment_cvar_a{alpha_suffix}"],
                "empirical_cvar": metrics[f"empirical_cvar_a{alpha_suffix}"],
            }
        )

    metric_df = pd.DataFrame(metric_rows)
    if metric_df.empty:
        return

    plt.figure(figsize=(8, 5))
    for condition, condition_df in metric_df.groupby("condition"):
        grouped = condition_df.groupby("lambda")[metric].mean().reset_index().sort_values("lambda")
        plt.plot(grouped["lambda"], grouped[metric], marker="o", label=format_condition_tick(condition))

    plt.xlabel("Lambda")
    plt.ylabel(reportable_text(ylabel))
    plt.title(reportable_text(title))
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def build_slide_table(summary: pd.DataFrame) -> pd.DataFrame:
    slide_df = summary[summary["lambda_label"] == "none"].copy()
    if slide_df.empty:
        return pd.DataFrame()

    slide_df = ensure_condition_order(slide_df)
    algorithms = sorted(slide_df["algorithm"].unique().tolist())

    out_rows: List[Dict[str, object]] = []
    for condition, condition_df in slide_df.groupby("condition"):
        row = {"condition": str(condition)}
        for algorithm in algorithms:
            alg_df = condition_df[condition_df["algorithm"] == algorithm]
            if alg_df.empty:
                row[algorithm] = ""
                continue
            worst = float(alg_df["worst_accuracy_mean"].iloc[0])
            avg = float(alg_df["avg_accuracy_mean"].iloc[0])
            tail = float(alg_df["tail_accuracy_mean"].iloc[0])
            row[algorithm] = f"{worst:.3f}/{avg:.3f} (tail:{tail:.3f})"
        out_rows.append(row)

    return pd.DataFrame(out_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze E3b/E2 tail-support CMNIST results.")
    parser.add_argument("results_dir", help="Path to training JSONL outputs.")
    parser.add_argument("--lambda_results_dir", default=None, help="Optional lambda-eval JSONL directory.")
    parser.add_argument("--output_dir", default="../results/E3b_tail_support", help="Output directory for CSV/plots.")
    parser.add_argument("--eval_envs", default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--alphas", default="0.5,0.75,0.9")
    parser.add_argument("--plot_alpha", type=float, default=0.9)
    args = parser.parse_args()

    eval_envs = parse_float_list(args.eval_envs)
    alphas = parse_float_list(args.alphas)
    if not alphas:
        raise ValueError("At least one alpha is required.")
    alpha_suffix = str(args.plot_alpha).replace(".", "")

    apply_plot_style()
    os.makedirs(args.output_dir, exist_ok=True)

    main_records = load_records(args.results_dir)
    main_frame = pd.DataFrame.from_records(main_records)
    if main_frame.empty:
        raise ValueError(f"No training records found under: {args.results_dir}")
    main_frame = main_frame[main_frame["phase"] == "tail_support"]
    if main_frame.empty:
        raise ValueError("No tail_support records found in main results.")

    raw_rows = build_raw_rows_from_main(main_frame, eval_envs)

    if args.lambda_results_dir is not None:
        lambda_records = load_records(args.lambda_results_dir)
        lambda_frame = pd.DataFrame.from_records(lambda_records)
        if not lambda_frame.empty:
            raw_rows.extend(build_raw_rows_from_lambda(lambda_frame, eval_envs))

    raw_df = pd.DataFrame(raw_rows)
    if raw_df.empty:
        raise ValueError("No raw rows could be constructed from the provided inputs.")

    # Required file 1: raw_results.csv
    raw_columns = [
        "condition",
        "algorithm",
        "seed",
        "lambda",
        "domain_id",
        "train_count",
        "test_accuracy",
        "loss",
        "empirical_weight",
        "deployment_weight",
    ]
    optional_columns = [
        "source",
        "alpha_eval",
        "head_domain_id",
        "tail_domain_id",
        "is_head_domain",
        "is_tail_domain",
    ]
    ordered_columns = raw_columns + [column for column in optional_columns if column in raw_df.columns]
    raw_df[ordered_columns].to_csv(os.path.join(args.output_dir, "raw_results.csv"), index=False)

    # Build per-run metrics first.
    summary_rows: List[Dict[str, object]] = []
    for (condition, algorithm, seed, lambda_value), group in raw_df.groupby(["condition", "algorithm", "seed", "lambda"], dropna=False):
        run_metrics = metrics_from_rows(group, alphas)
        run_metrics.update(
            {
                "condition": condition,
                "algorithm": algorithm,
                "seed": int(seed),
                "lambda": lambda_value,
                "lambda_label": "none" if pd.isna(lambda_value) else str(lambda_value),
            }
        )
        summary_rows.append(run_metrics)

    run_metrics_df = pd.DataFrame(summary_rows)

    # Required file 2: summary_by_condition.csv (mean/std across seeds)
    metric_columns = [
        column for column in run_metrics_df.columns
        if column
        not in ["condition", "algorithm", "seed", "lambda", "lambda_label"]
    ]

    summary_by_condition = (
        run_metrics_df.groupby(["condition", "algorithm", "lambda_label"])[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary_by_condition.columns = [
        column if isinstance(column, str) else "_".join([token for token in column if token])
        for column in summary_by_condition.columns
    ]
    summary_by_condition = ensure_condition_order(summary_by_condition)
    summary_by_condition.to_csv(os.path.join(args.output_dir, "summary_by_condition.csv"), index=False)

    # Required file 3: slide_table.csv
    slide_table = build_slide_table(summary_by_condition)
    slide_table.to_csv(os.path.join(args.output_dir, "slide_table.csv"), index=False)

    # Required plots.
    plot_metric(
        summary_by_condition,
        "tail_accuracy",
        os.path.join(args.output_dir, "tail_accuracy_by_condition.png"),
        "Tail-domain accuracy by condition",
        "Tail-domain accuracy",
    )
    plot_metric(
        summary_by_condition,
        "worst_accuracy",
        os.path.join(args.output_dir, "worst_accuracy_by_condition.png"),
        "Worst-domain accuracy by condition",
        "Worst-domain accuracy",
    )
    plot_metric(
        summary_by_condition,
        "head_tail_gap",
        os.path.join(args.output_dir, "head_tail_gap_by_condition.png"),
        "Head-tail accuracy gap by condition",
        "Head-tail gap",
    )
    plot_metric(
        summary_by_condition,
        f"cvar_gap_a{alpha_suffix}",
        os.path.join(args.output_dir, "cvar_gap_by_condition.png"),
        f"CVaR gap by condition (alpha={args.plot_alpha})",
        "|deployment CVaR - empirical CVaR|",
    )

    plot_iro_lambda(
        raw_df,
        "tail_accuracy",
        os.path.join(args.output_dir, "iro_lambda_tail_accuracy_by_condition.png"),
        "IRO lambda sensitivity: tail-domain accuracy",
        "Tail-domain accuracy",
        alpha_suffix,
    )
    plot_iro_lambda(
        raw_df,
        "deployment_cvar",
        os.path.join(args.output_dir, "iro_lambda_cvar_by_condition.png"),
        f"IRO lambda sensitivity: deployment CVaR (alpha={args.plot_alpha})",
        "Deployment CVaR",
        alpha_suffix,
    )

    print(f"Wrote outputs to {args.output_dir}")
    print(f"Rows in raw_results.csv: {len(raw_df)}")


if __name__ == "__main__":
    main()
