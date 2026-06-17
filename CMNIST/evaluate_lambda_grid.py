import argparse
import copy
import glob
import hashlib
import json
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

import algorithms as algorithms
import networks as networks
from collect_results import enrich_record
from datasets import get_cmnist_datasets
from lib.fast_data_loader import FastDataLoader
from lib.iro_utils import aggregation_function
from lib.misc import accuracy, loss


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalize_train_env_sizes(args_dict):
    value = args_dict.get("train_env_sizes_parsed")
    if value is not None:
        return tuple(int(size) for size in value)

    raw_value = args_dict.get("train_env_sizes", "")
    if raw_value in [None, ""]:
        return None
    return tuple(int(size) for size in str(raw_value).split(","))


def build_loss_and_target_mode(args_dict):
    if args_dict["loss_fn"] == "nll":
        return F.binary_cross_entropy_with_logits, False
    return F.cross_entropy, True


def build_network(args_dict, input_shape, n_targets):
    if args_dict["network"] == "MLP":
        return networks.MLP(
            np.prod(input_shape),
            args_dict["mlp_hidden_dim"],
            n_targets,
            dropout=args_dict["dropout_p"],
        )
    if args_dict["network"] == "FiLMedMLP":
        return networks.FiLMedMLP(
            np.prod(input_shape),
            args_dict["mlp_hidden_dim"],
            n_targets,
            dropout=args_dict["dropout_p"],
            film_dim=1,
        )
    if args_dict["network"] == "CNN":
        return networks.CNN(input_shape)
    raise NotImplementedError(f"Unknown network: {args_dict['network']}")


def parse_checkpoint_type(checkpoint_path):
    stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
    if stem.endswith("_best"):
        return "best"
    if stem.endswith("_final"):
        return "final"
    return "unknown"


def build_lambda_grid(grid_spec):
    if "," in grid_spec:
        return [float(value) for value in grid_spec.split(",")]
    if ":" in grid_spec:
        start, stop, step = [float(value) for value in grid_spec.split(":")]
        grid = []
        current = start
        while current <= stop + 1e-9:
            grid.append(round(current, 10))
            current += step
        return grid
    raise ValueError("lambda_grid must be a comma-separated list or start:stop:step")


def evaluate_checkpoint(checkpoint_path, lambda_grid, batch_size, n_workers, eval_envs):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    args_dict = copy.deepcopy(checkpoint["args"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seed_all(args_dict.get("seed", 0))
    if args_dict.get("deterministic", False):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    loss_fn, int_target = build_loss_and_target_mode(args_dict)
    train_envs = tuple(args_dict.get("train_env_ps", []))
    train_env_sizes = normalize_train_env_sizes(args_dict)

    reference_envs = get_cmnist_datasets(
        args_dict["data_dir"],
        train_envs=train_envs,
        test_envs=(eval_envs[0],),
        label_noise_rate=0.25,
        int_target=int_target,
        subsample=not args_dict.get("full_resolution", False),
        cuda=(device == "cuda"),
        train_env_sizes=train_env_sizes,
        train_env_size_mode=args_dict.get("train_env_size_mode", "random"),
    )
    input_shape = reference_envs[0].tensors[0].size()[1:]
    n_targets = 1 if args_dict["loss_fn"] == "nll" else 2

    net = build_network(args_dict, input_shape, n_targets)
    algorithm_class = algorithms.get_algorithm_class(args_dict["algorithm"])
    algorithm = algorithm_class(net, args_dict, loss_fn)
    algorithm.load_state_dict(checkpoint["model_dict"], strict=False)
    algorithm.to(device)

    all_envs = get_cmnist_datasets(
        args_dict["data_dir"],
        train_envs=[],
        test_envs=tuple(eval_envs),
        label_noise_rate=0.25,
        int_target=int_target,
        subsample=not args_dict.get("full_resolution", False),
        cuda=(device == "cuda"),
        use_test_set=True,
    )
    loaders = [FastDataLoader(dataset=env, batch_size=batch_size, num_workers=n_workers) for env in all_envs]
    aggregator = aggregation_function(name="cvar")
    checkpoint_type = parse_checkpoint_type(checkpoint_path)

    args_no_seed = copy.deepcopy(args_dict)
    if "seed" in args_no_seed:
        del args_no_seed["seed"]
    args_id = hashlib.md5(str(args_no_seed).encode("utf-8")).hexdigest()

    lambda_records = []
    algorithm_name = args_dict["algorithm"].lower()
    for lambda_value in lambda_grid:
        env_losses = []
        env_accs = []
        record = {
            "algorithm": algorithm_name,
            "seed": args_dict.get("seed", 0),
            "args": args_no_seed,
            "args_id": args_id,
            "checkpoint_path": checkpoint_path,
            "checkpoint_type": checkpoint_type,
            "lambda_eval": float(lambda_value),
        }

        for env_name, loader in zip([str(env) for env in eval_envs], loaders):
            if algorithm_name in ["iro", "inftask"]:
                env_acc = accuracy(algorithm, loader, device, alpha=lambda_value)
                env_loss = loss(algorithm, loader, loss_fn, device, alpha=lambda_value)
            else:
                env_acc = accuracy(algorithm, loader, device)
                env_loss = loss(algorithm, loader, loss_fn, device)
            record[f"{env_name}_acc"] = env_acc
            record[f"{env_name}_loss"] = env_loss
            env_accs.append(env_acc)
            env_losses.append(env_loss)

        risks = torch.tensor(env_losses, dtype=torch.float32)
        record["aggregated_risk"] = float(aggregator.aggregate(risks, torch.tensor(lambda_value)).item())
        record["avg_domain_acc"] = float(sum(env_accs) / len(env_accs))
        record["worst_domain_acc"] = float(min(env_accs))
        record["best_domain_acc"] = float(max(env_accs))
        record["phase"] = "lambda_eval"
        lambda_records.append(enrich_record(record))

    return lambda_records, args_dict


def resolve_checkpoint_paths(checkpoint_arg):
    if os.path.isdir(checkpoint_arg):
        pattern = os.path.join(checkpoint_arg, "*.pkl")
        return sorted(glob.glob(pattern))
    return sorted(glob.glob(checkpoint_arg))


def default_output_path(args_dict, checkpoint_path):
    checkpoint_stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
    output_dir = os.path.join(args_dict["output_dir"], "lambda_eval", args_dict["exp_name"])
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"{checkpoint_stem}.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CMNIST checkpoints across a lambda grid.")
    parser.add_argument("checkpoint_path", help="Checkpoint path, glob, or checkpoint directory.")
    parser.add_argument("--lambda_grid", default="0.0:1.0:0.1")
    parser.add_argument("--batch_size", type=int, default=5000)
    parser.add_argument("--n_workers", type=int, default=0)
    parser.add_argument("--eval_envs", type=str, default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--output_dir", default=None, help="Optional override for lambda-eval output directory.")
    args = parser.parse_args()

    lambda_grid = build_lambda_grid(args.lambda_grid)
    eval_envs = [float(value) for value in args.eval_envs.split(",")]
    checkpoint_paths = resolve_checkpoint_paths(args.checkpoint_path)
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints found for: {args.checkpoint_path}")

    for checkpoint_path in checkpoint_paths:
        lambda_records, checkpoint_args = evaluate_checkpoint(
            checkpoint_path,
            lambda_grid,
            args.batch_size,
            args.n_workers,
            eval_envs,
        )
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            output_path = os.path.join(
                args.output_dir,
                f"{os.path.splitext(os.path.basename(checkpoint_path))[0]}.jsonl",
            )
        else:
            output_path = default_output_path(checkpoint_args, checkpoint_path)

        with open(output_path, "w") as output_file:
            for record in lambda_records:
                output_file.write(json.dumps(record, sort_keys=True) + "\n")

        print(f"Saved {len(lambda_records)} lambda-eval rows to {output_path}")