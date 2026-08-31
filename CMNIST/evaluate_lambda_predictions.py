#!/usr/bin/env python3
"""Evaluate CMNIST checkpoints on fixed test tensors across lambda values."""

import argparse
import copy
import glob
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import algorithms
import networks
from datasets import get_cmnist_datasets
from lib.fast_data_loader import FastDataLoader


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_grid(text):
    if ":" in text:
        start, stop, step = [float(value) for value in text.split(":")]
        values = []
        current = start
        while current <= stop + 1e-9:
            values.append(round(current, 10))
            current += step
        return values
    return [float(value) for value in text.split(",") if value.strip()]


def build_network(args_dict, input_shape, n_targets):
    if args_dict["network"] == "MLP":
        return networks.MLP(np.prod(input_shape), args_dict["mlp_hidden_dim"], n_targets, dropout=args_dict["dropout_p"])
    if args_dict["network"] == "FiLMedMLP":
        return networks.FiLMedMLP(np.prod(input_shape), args_dict["mlp_hidden_dim"], n_targets, dropout=args_dict["dropout_p"], film_dim=1)
    if args_dict["network"] == "CNN":
        return networks.CNN(input_shape)
    raise ValueError(f"Unknown network: {args_dict['network']}")


def resolve_paths(path_arg, algorithms_filter, max_checkpoints, distinct_algorithms):
    paths = sorted(glob.glob(path_arg)) if not os.path.isdir(path_arg) else sorted(glob.glob(os.path.join(path_arg, "*.pkl")))
    selected = []
    selected_algorithms = set()
    for path in paths:
        checkpoint = torch.load(path, map_location="cpu")
        algorithm = str(checkpoint.get("args", {}).get("algorithm", "")).lower()
        if algorithms_filter and algorithm not in algorithms_filter:
            continue
        if distinct_algorithms and algorithm in selected_algorithms:
            continue
        selected.append(path)
        selected_algorithms.add(algorithm)
        if max_checkpoints and len(selected) >= max_checkpoints:
            break
    return selected


def evaluate_checkpoint(path, lambdas, eval_envs, batch_size, device_name, output_dir):
    checkpoint = torch.load(path, map_location="cpu")
    args_dict = copy.deepcopy(checkpoint["args"])
    seed_all(int(args_dict.get("seed", 0)))
    device = torch.device(device_name)
    int_target = args_dict["loss_fn"] != "nll"
    loss_fn = F.cross_entropy if int_target else F.binary_cross_entropy_with_logits

    datasets = get_cmnist_datasets(
        args_dict["data_dir"],
        train_envs=[],
        test_envs=tuple(eval_envs),
        label_noise_rate=0.25,
        int_target=int_target,
        subsample=not args_dict.get("full_resolution", False),
        cuda=False,
        use_test_set=True,
    )
    input_shape = datasets[0].tensors[0].size()[1:]
    n_targets = 2 if int_target else 1
    model = build_network(args_dict, input_shape, n_targets)
    algorithm_class = algorithms.get_algorithm_class(args_dict["algorithm"])
    algorithm = algorithm_class(model, args_dict, loss_fn)
    algorithm.load_state_dict(checkpoint["model_dict"], strict=False)
    algorithm.to(device)
    algorithm.eval()

    labels = []
    all_predictions = []
    rows = []
    algorithm_name = str(args_dict["algorithm"]).lower()
    for env_index, dataset in enumerate(datasets):
        loader = FastDataLoader(dataset=dataset, batch_size=batch_size, num_workers=0)
        env_labels = []
        env_predictions = []
        for x, y in loader:
            x = x.to(device)
            with torch.no_grad():
                for lambda_index, lambda_value in enumerate(lambdas):
                    if algorithm_name in ["iro", "inftask"]:
                        alpha = torch.full((len(x), 1), float(lambda_value), device=device)
                        logits = algorithm.predict(x, alpha)
                    else:
                        logits = algorithm.predict(x)
                    if n_targets == 1:
                        predictions = (logits > 0).long().view(-1)
                    else:
                        predictions = logits.argmax(dim=1)
                    while len(env_predictions) <= lambda_index:
                        env_predictions.append([])
                    env_predictions[lambda_index].append(predictions.cpu().numpy())
            env_labels.append(y.view(-1).cpu().numpy())
        env_labels = np.concatenate(env_labels)
        env_predictions = [np.concatenate(values) for values in env_predictions]
        labels.append(env_labels)
        all_predictions.append(env_predictions)
        for lambda_index, lambda_value in enumerate(lambdas):
            prediction = env_predictions[lambda_index]
            rows.append({
                "checkpoint_path": path,
                "algorithm": algorithm_name,
                "seed": int(args_dict.get("seed", 0)),
                "test_env": float(eval_envs[env_index]),
                "lambda_eval": float(lambda_value),
                "accuracy": float(np.mean(prediction == env_labels)),
            })

    prediction_path = output_dir / (Path(path).stem + "_predictions.npz")
    np.savez_compressed(
        prediction_path,
        labels=np.asarray(labels, dtype=object),
        predictions=np.asarray(all_predictions, dtype=object),
        eval_envs=np.asarray(eval_envs),
        lambdas=np.asarray(lambdas),
    )

    for env_index, env_predictions in enumerate(all_predictions):
        reference = env_predictions[0]
        for lambda_index, prediction in enumerate(env_predictions):
            rows_for_env = [row for row in rows if row["test_env"] == float(eval_envs[env_index]) and row["lambda_eval"] == float(lambdas[lambda_index])]
            rows_for_env[0]["disagreement_from_lambda_0"] = float(np.mean(prediction != reference))
            if lambda_index > 0:
                rows_for_env[0]["neighbor_disagreement"] = float(np.mean(prediction != env_predictions[lambda_index - 1]))
            else:
                rows_for_env[0]["neighbor_disagreement"] = 0.0

    return rows, prediction_path


def summarize(rows):
    frame = __import__("pandas").DataFrame(rows)
    summaries = []
    for (path, algorithm, seed), group in frame.groupby(["checkpoint_path", "algorithm", "seed"]):
        for metric in ["accuracy", "disagreement_from_lambda_0", "neighbor_disagreement"]:
            values = group.groupby("lambda_eval")[metric].mean().sort_index()
            summaries.append({
                "checkpoint_path": path,
                "algorithm": algorithm,
                "seed": seed,
                "metric": metric,
                "best_value": float(values.max()),
                "worst_value": float(values.min()),
                "best_worst_range": float(values.max() - values.min()),
                "max_neighbor_change": float(values.diff().abs().fillna(0).max()),
                "mean_value": float(values.mean()),
            })
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_path")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--lambda_grid", default="0.0:1.0:0.1")
    parser.add_argument("--eval_envs", default="0.0,0.1,0.5,0.9,1.0")
    parser.add_argument("--batch_size", type=int, default=5000)
    parser.add_argument("--algorithms", default="iro,inftask")
    parser.add_argument("--max_checkpoints", type=int, default=2)
    parser.add_argument("--distinct_algorithms", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    existing_outputs = [path for path in output_dir.iterdir() if path.name != "runner_logs"] if output_dir.exists() else []
    if existing_outputs:
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    lambdas = parse_grid(args.lambda_grid)
    eval_envs = [float(value) for value in args.eval_envs.split(",")]
    algorithms_filter = {value.strip().lower() for value in args.algorithms.split(",") if value.strip()}
    paths = resolve_paths(args.checkpoint_path, algorithms_filter, args.max_checkpoints, args.distinct_algorithms)
    if not paths:
        raise FileNotFoundError("No matching checkpoints found")

    rows = []
    for path in paths:
        checkpoint_rows, prediction_path = evaluate_checkpoint(path, lambdas, eval_envs, args.batch_size, args.device, output_dir)
        rows.extend(checkpoint_rows)
        print(f"Evaluated {path}; saved {prediction_path}")

    import pandas as pd
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "prediction_lambda_metrics.csv", index=False)
    with (output_dir / "prediction_lambda_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    pd.DataFrame(summarize(rows)).to_csv(output_dir / "prediction_lambda_sensitivity_summary.csv", index=False)
    print(f"Checkpoints: {len(paths)}; rows: {len(rows)}; lambdas: {len(lambdas)}; environments: {len(eval_envs)}")


if __name__ == "__main__":
    main()
