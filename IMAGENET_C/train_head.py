#!/usr/bin/env python3
"""Smoke-capable training entry point for ImageNet-C frozen-feature heads."""

import argparse
import copy
import json
import os
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from datasets import build_real_feature_splits, ensure_output_subdirs, smoke_feature_splits, write_manifest
from models import LinearClassifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_csv_list(text: str, cast_fn=str) -> List:
    return [cast_fn(token.strip()) for token in text.split(",") if token.strip()]


def domain_loss_vector(model: LinearClassifier, train_domains: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    losses = {}
    for domain_name, tensors in train_domains.items():
        logits = model(tensors["x"])
        losses[domain_name] = F.cross_entropy(logits, tensors["y"])
    return losses


def cvar_objective(losses: torch.Tensor, lambda_value: float) -> torch.Tensor:
    if losses.numel() == 0:
        raise ValueError("No domain losses provided for CVaR objective")
    quantile = torch.quantile(losses, torch.tensor(float(lambda_value), dtype=losses.dtype))
    tail_losses = losses[losses >= quantile]
    return tail_losses.mean()


def train_one_model(algorithm: str, split_bundle: Dict[str, object], args: argparse.Namespace, seed: int) -> Dict[str, object]:
    set_seed(seed)
    train_domains = split_bundle["train_domains"]
    val_conditions = split_bundle["val_conditions"]
    model = LinearClassifier(split_bundle["feature_dim"], split_bundle["num_classes"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    lambda_grid = parse_csv_list(args.lambda_grid, float)
    q_weights = torch.ones(len(train_domains), dtype=torch.float32)
    domain_names = list(train_domains.keys())

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    patience_left = args.patience
    history = []

    for epoch in range(args.max_epochs):
        model.train()
        optimizer.zero_grad()
        losses_by_domain = domain_loss_vector(model, train_domains)
        loss_tensor = torch.stack([losses_by_domain[name] for name in domain_names])

        if algorithm == "erm":
            objective = loss_tensor.mean()
        elif algorithm == "groupdro":
            q_weights = q_weights * torch.exp(args.groupdro_eta * loss_tensor.detach())
            q_weights = q_weights / q_weights.sum()
            objective = torch.dot(q_weights, loss_tensor)
        elif algorithm == "iro":
            sampled_lambdas = lambda_grid[: min(len(lambda_grid), 3)]
            cvar_losses = [cvar_objective(loss_tensor, lambda_value) for lambda_value in sampled_lambdas]
            objective = torch.stack(cvar_losses).mean()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        objective.backward()
        optimizer.step()

        val_loss = evaluate_visible_validation(model, val_conditions)
        history.append({"epoch": epoch + 1, "train_objective": float(objective.item()), "val_loss": float(val_loss)})
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state)
    return {
        "model": model,
        "best_val_loss": best_val_loss,
        "history": history,
    }


def evaluate_visible_validation(model: LinearClassifier, val_conditions: List[Dict[str, object]]) -> float:
    if not val_conditions:
        return float("nan")
    model.eval()
    losses = []
    with torch.no_grad():
        for condition in val_conditions:
            logits = model(condition["x"])
            losses.append(F.cross_entropy(logits, condition["y"]).item())
    return float(np.mean(losses))


def evaluate_conditions(model: LinearClassifier, split_bundle: Dict[str, object]) -> Dict[str, object]:
    model.eval()
    rows = []
    all_conditions = list(split_bundle["eval_conditions"]) + [split_bundle["clean_condition"]]
    with torch.no_grad():
        for condition in all_conditions:
            logits = model(condition["x"])
            loss = F.cross_entropy(logits, condition["y"]).item()
            accuracy = (logits.argmax(dim=1) == condition["y"]).float().mean().item()
            rows.append(
                {
                    "corruption": condition["corruption"],
                    "severity": int(condition["severity"]),
                    "split": condition["split"],
                    "domain_seen": bool(condition["domain_seen"]),
                    "accuracy": float(accuracy),
                    "loss": float(loss),
                }
            )

    held_out_rows = [row for row in rows if row["split"] == "held_out"]
    all_corruption_rows = [row for row in rows if row["split"] in {"held_out", "all"}]
    clean_rows = [row for row in rows if row["split"] == "clean"]

    def summarize(target_rows: List[Dict[str, object]], prefix: str) -> Dict[str, float]:
        if not target_rows:
            return {
                f"{prefix}_mean_accuracy": float("nan"),
                f"{prefix}_worst_accuracy": float("nan"),
                f"{prefix}_mean_loss": float("nan"),
                f"{prefix}_worst_loss": float("nan"),
            }
        accuracies = [float(row["accuracy"]) for row in target_rows]
        losses = [float(row["loss"]) for row in target_rows]
        return {
            f"{prefix}_mean_accuracy": float(np.mean(accuracies)),
            f"{prefix}_worst_accuracy": float(np.min(accuracies)),
            f"{prefix}_mean_loss": float(np.mean(losses)),
            f"{prefix}_worst_loss": float(np.max(losses)),
        }

    summary = {}
    summary.update(summarize(held_out_rows, "held_out"))
    summary.update(summarize(all_corruption_rows, "all_corruptions"))
    summary["clean_accuracy"] = float(clean_rows[0]["accuracy"]) if clean_rows else float("nan")
    summary["clean_loss"] = float(clean_rows[0]["loss"]) if clean_rows else float("nan")
    return {
        "rows": rows,
        "summary": summary,
    }


def save_checkpoint(output_dir: Path, record: Dict[str, object], model: LinearClassifier, args: argparse.Namespace) -> str:
    ckpt_dir = output_dir / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ckpt_dir / f"{record['run_id']}_best.pt"
    checkpoint = {
        "args": {
            "data_mode": record["data_mode"],
            "fold": record["fold"],
            "seed": record["seed"],
            "algorithm": record["algorithm"],
            "feature_dim": record["feature_dim"],
            "num_classes": record["num_classes"],
            "visible_domains": record["visible_domains"],
            "held_out_domains": record["held_out_domains"],
            "lambda_grid": args.lambda_grid,
            "smoke_feature_dim": getattr(args, "smoke_feature_dim", None),
            "smoke_num_classes": getattr(args, "smoke_num_classes", None),
            "smoke_train_samples_per_class": getattr(args, "smoke_train_samples_per_class", None),
            "smoke_val_samples_per_class": getattr(args, "smoke_val_samples_per_class", None),
            "smoke_test_samples_per_class": getattr(args, "smoke_test_samples_per_class", None),
            "folds_path": args.folds_path,
            "imagenet_train_corrupted_root": getattr(args, "imagenet_train_corrupted_root", None),
            "imagenet_val_root": getattr(args, "imagenet_val_root", None),
            "imagenet_c_root": getattr(args, "imagenet_c_root", None),
            "batch_size": getattr(args, "batch_size", None),
            "max_images_per_condition": getattr(args, "max_images_per_condition", None),
            "clean_loader_backend": getattr(args, "clean_loader_backend", "imagenet"),
        },
        "model_dict": model.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)
    return str(checkpoint_path)


def write_records(output_dir: Path, records: List[Dict[str, object]]) -> None:
    raw_path = output_dir / "raw" / "train_runs.jsonl"
    with raw_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def run_smoke(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    ensure_output_subdirs(output_dir, ["raw", "summary", "plots", "manifests", "ckpts"], allow_existing_files=False)
    algorithms = parse_csv_list(args.algorithms, str)
    seeds = parse_csv_list(args.seeds, int)

    records = []
    for seed in seeds:
        split_bundle = smoke_feature_splits(
            folds_path=args.folds_path,
            fold_name=args.fold,
            seed=seed,
            feature_dim=args.smoke_feature_dim,
            num_classes=args.smoke_num_classes,
            train_samples_per_class=args.smoke_train_samples_per_class,
            val_samples_per_class=args.smoke_val_samples_per_class,
            test_samples_per_class=args.smoke_test_samples_per_class,
        )
        for algorithm in algorithms:
            trained = train_one_model(algorithm, split_bundle, args, seed)
            evaluation = evaluate_conditions(trained["model"], split_bundle)
            record = {
                "dataset": "imagenet_c_smoke",
                "experiment": "fold_generalization_smoke",
                "data_mode": "smoke",
                "fold": args.fold,
                "seed": seed,
                "algorithm": algorithm,
                "run_id": f"imagenet_c__fold_generalization_smoke__{args.fold}__{algorithm}__seed{seed}",
                "visible_domains": split_bundle["visible_domains"],
                "held_out_domains": split_bundle["held_out_domains"],
                "feature_dim": split_bundle["feature_dim"],
                "num_classes": split_bundle["num_classes"],
                "best_val_loss": float(trained["best_val_loss"]),
                "training_history": trained["history"],
                "eval_rows": evaluation["rows"],
            }
            record.update(evaluation["summary"])
            record["checkpoint_path"] = save_checkpoint(output_dir, record, trained["model"], args)
            records.append(record)

    write_records(output_dir, records)
    write_manifest(output_dir, f"{args.fold}_{'_'.join(f'seed{seed}' for seed in seeds)}_smoke.txt", os.sys.argv, "Fold smoke manifest")
    print(f"Saved {len(records)} smoke training records to {output_dir / 'raw' / 'train_runs.jsonl'}")


def run_real(args: argparse.Namespace) -> None:
    imagenet_root = args.imagenet_root or args.imagenet_val_root
    if args.imagenet_train_corrupted_root is None or imagenet_root is None or args.imagenet_c_root is None:
        raise ValueError(
            "Real mode requires --imagenet_train_corrupted_root, --imagenet_root, and --imagenet_c_root"
        )

    output_dir = Path(args.output_dir)
    ensure_output_subdirs(output_dir, ["raw", "summary", "plots", "manifests", "ckpts"], allow_existing_files=False)
    algorithms = parse_csv_list(args.algorithms, str)
    seeds = parse_csv_list(args.seeds, int)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    records = []
    for seed in seeds:
        set_seed(seed)
        split_bundle = build_real_feature_splits(
            folds_path=args.folds_path,
            fold_name=args.fold,
            imagenet_train_corrupted_root=args.imagenet_train_corrupted_root,
            imagenet_root=imagenet_root,
            imagenet_c_root=args.imagenet_c_root,
            batch_size=args.batch_size,
            max_images_per_condition=args.max_images_per_condition,
            device=device,
            clean_loader_backend=args.clean_loader_backend,
        )
        for algorithm in algorithms:
            trained = train_one_model(algorithm, split_bundle, args, seed)
            evaluation = evaluate_conditions(trained["model"], split_bundle)
            record = {
                "dataset": "imagenet_c",
                "experiment": "fold_generalization",
                "data_mode": "real",
                "fold": args.fold,
                "seed": seed,
                "algorithm": algorithm,
                "run_id": f"imagenet_c__fold_generalization__{args.fold}__{algorithm}__seed{seed}",
                "visible_domains": split_bundle["visible_domains"],
                "held_out_domains": split_bundle["held_out_domains"],
                "feature_dim": split_bundle["feature_dim"],
                "num_classes": split_bundle["num_classes"],
                "best_val_loss": float(trained["best_val_loss"]),
                "training_history": trained["history"],
                "eval_rows": evaluation["rows"],
            }
            record.update(evaluation["summary"])
            record["checkpoint_path"] = save_checkpoint(output_dir, record, trained["model"], args)
            records.append(record)

    write_records(output_dir, records)
    write_manifest(output_dir, f"{args.fold}_{'_'.join(f'seed{seed}' for seed in seeds)}.txt", os.sys.argv, "Fold manifest")
    print(f"Saved {len(records)} real training records to {output_dir / 'raw' / 'train_runs.jsonl'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ImageNet-C frozen-head training entry point.")
    parser.add_argument("--output_dir", default="results/imagenet_c_fold_generalization_smoke_v1")
    parser.add_argument("--folds_path", default="IMAGENET_C/folds.json")
    parser.add_argument("--fold", default="fold_a")
    parser.add_argument("--algorithms", default="erm,groupdro,iro")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--imagenet_train_corrupted_root", default=None)
    parser.add_argument("--imagenet_root", default=None)
    # Backward-compat alias; prefer --imagenet_root.
    parser.add_argument("--imagenet_val_root", default=None)
    parser.add_argument("--imagenet_c_root", default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_images_per_condition", type=int, default=None)
    parser.add_argument("--clean_loader_backend", choices=["imagenet", "imagefolder", "auto"], default="imagenet")
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=3e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--groupdro_eta", type=float, default=0.1)
    parser.add_argument("--lambda_grid", default="0.0,0.5,0.9")
    parser.add_argument("--smoke_feature_dim", type=int, default=64)
    parser.add_argument("--smoke_num_classes", type=int, default=20)
    parser.add_argument("--smoke_train_samples_per_class", type=int, default=4)
    parser.add_argument("--smoke_val_samples_per_class", type=int, default=2)
    parser.add_argument("--smoke_test_samples_per_class", type=int, default=3)
    args = parser.parse_args()

    if not args.smoke:
        run_real(args)
        return

    run_smoke(args)


if __name__ == "__main__":
    main()