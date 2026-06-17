import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

from collect_results import load_records


def build_frame(results_dir, algorithms=None):
    records = load_records(results_dir)
    frame = pd.DataFrame.from_records(records)
    if algorithms is not None:
        frame = frame[frame["algorithm"].isin(algorithms)]
    return frame


def ensure_output_dir(output_dir):
    os.makedirs(output_dir, exist_ok=True)


def env_metric_columns(frame, suffix):
    env_columns = []
    for column in frame.columns:
        if not column.endswith(suffix):
            continue
        prefix = column.split("_")[0]
        try:
            float(prefix)
        except ValueError:
            continue
        env_columns.append(column)
    return sorted(env_columns, key=lambda value: float(value.split("_")[0]))


def plot_e0_accuracy(frame, output_dir):
    phase_frame = frame[frame["phase"].isin(["reproduction", "validation_smoke"])]
    if phase_frame.empty:
        return None

    acc_columns = env_metric_columns(phase_frame, "_acc_best")
    if not acc_columns:
        return None

    plt.figure(figsize=(8, 5))
    for algorithm, algorithm_frame in phase_frame.groupby("algorithm"):
        row = algorithm_frame.iloc[0]
        xs = [float(column.split("_")[0]) for column in acc_columns]
        ys = [row[column] for column in acc_columns]
        plt.plot(xs, ys, marker="o", label=algorithm)

    plt.xlabel("Test environment e")
    plt.ylabel("Accuracy")
    plt.title("E0 / Validation accuracy across test environments")
    plt.legend()
    plt.tight_layout()
    output_path = os.path.join(output_dir, "e0_accuracy_by_test_env.png")
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_grouped_metric(frame, phase_name, x_column, y_column, output_name, title, xlabel, output_dir):
    phase_frame = frame[frame["phase"] == phase_name].copy()
    if phase_frame.empty or x_column not in phase_frame.columns or y_column not in phase_frame.columns:
        return None

    phase_frame = phase_frame.dropna(subset=[x_column, y_column])
    if phase_frame.empty:
        return None

    grouped = (phase_frame.groupby([x_column, "algorithm"])[y_column]
               .mean()
               .reset_index())
    pivot = grouped.pivot(index=x_column, columns="algorithm", values=y_column).sort_index()
    if pivot.empty:
        return None

    ax = pivot.plot(kind="bar", figsize=(9, 5))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(y_column)
    plt.tight_layout()
    output_path = os.path.join(output_dir, output_name)
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_lambda_risk(lambda_results_dir, output_dir):
    lambda_records = load_records(lambda_results_dir)
    frame = pd.DataFrame.from_records(lambda_records)
    if frame.empty or "lambda_eval" not in frame.columns or "aggregated_risk" not in frame.columns:
        return None

    plt.figure(figsize=(8, 5))
    for algorithm, algorithm_frame in frame.groupby("algorithm"):
        grouped = algorithm_frame.groupby("lambda_eval")["aggregated_risk"].mean().reset_index().sort_values("lambda_eval")
        plt.plot(grouped["lambda_eval"], grouped["aggregated_risk"], marker="o", label=algorithm)

    plt.xlabel("Lambda")
    plt.ylabel("Aggregated risk")
    plt.title("E4 aggregated risk over lambda")
    plt.legend()
    plt.tight_layout()
    output_path = os.path.join(output_dir, "e4_lambda_aggregated_risk.png")
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot CMNIST stress-test summaries.")
    parser.add_argument("results_dir")
    parser.add_argument("--output_dir", default="plots")
    parser.add_argument("--lambda_results_dir", default=None)
    parser.add_argument("--algorithms", nargs="+", default=None)
    parser.add_argument("--metric", default="worst_domain_acc_best")
    args = parser.parse_args()

    ensure_output_dir(args.output_dir)
    frame = build_frame(args.results_dir, args.algorithms)

    generated = []
    for output_path in [
        plot_e0_accuracy(frame, args.output_dir),
        plot_grouped_metric(
            frame,
            "domain_count",
            "n_train_domains",
            args.metric,
            "e1_domain_count_worst_domain_accuracy.png",
            "E1 worst-domain accuracy by number of training domains",
            "Number of training domains",
            args.output_dir,
        ),
        plot_grouped_metric(
            frame,
            "sample_size",
            "sample_size_per_domain",
            args.metric,
            "e2_sample_size_worst_domain_accuracy.png",
            "E2 worst-domain accuracy by samples per domain",
            "Samples per domain",
            args.output_dir,
        ),
        plot_grouped_metric(
            frame,
            "imbalance",
            "imbalance_type",
            args.metric,
            "e3_imbalance_worst_domain_accuracy.png",
            "E3 worst-domain accuracy by imbalance condition",
            "Imbalance condition",
            args.output_dir,
        ),
    ]:
        if output_path is not None:
            generated.append(output_path)

    if args.lambda_results_dir is not None:
        lambda_output = plot_lambda_risk(args.lambda_results_dir, args.output_dir)
        if lambda_output is not None:
            generated.append(lambda_output)

    if generated:
        for path in generated:
            print(f"Generated {path}")
    else:
        print("No plots were generated from the provided inputs.")