#!/usr/bin/env python3
"""Generate reduced-scope per-seed command files from domain_stress.txt."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


SEED_RE = re.compile(r"--seed\s+(\d+)")
ENVS_RE = re.compile(r"--train_envs\s+([^\s]+)")
SIZES_RE = re.compile(r"--train_env_sizes\s+([^\s]+)")


E1_ENVS = {
    "0.1,0.2",
    "0.01,0.12,0.5,0.99",
    "0.01,0.12,0.0,0.0,0.14,0.5,0.7,0.99",
}

E2_SIZES = {
    "2000,2000,2000,2000",
    "4000,4000,4000,4000",
    "8000,8000,8000,8000",
}

E3_SIZES = {
    "balanced": "2000,2000,2000,2000",
    "mild_imbalance": "2000,2000,2000,4000",
    "strong_imbalance": "2000,2000,2000,10000",
}


def include_line(line: str, heavy_size: int) -> tuple[bool, str]:
    env_match = ENVS_RE.search(line)
    if not env_match:
        return False, line
    envs = env_match.group(1)

    sizes_match = SIZES_RE.search(line)
    if not sizes_match:
        return (envs in E1_ENVS), line

    if envs != "0.01,0.12,0.5,0.99":
        return False, line

    sizes = sizes_match.group(1)
    if sizes in E2_SIZES:
        return True, line

    # Check E3 sizes (severity-based imbalance)
    if sizes in E3_SIZES.values():
        return True, line

    return False, line


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reduced-scope per-seed command files")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("domain_stress.txt"),
        help="Path to full command file",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated seeds to generate",
    )
    parser.add_argument(
        "--heavy_size",
        type=int,
        default=10000,
        help="Strong-heavy domain size for reduced E3 conditions",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="domain_stress_main_seed",
        help="Output filename prefix",
    )
    args = parser.parse_args()

    source = args.source
    if not source.exists():
        raise FileNotFoundError(f"Source command file not found: {source}")

    target_seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    lines = source.read_text(encoding="utf-8").splitlines()

    out = {seed: [] for seed in target_seeds}
    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        seed_match = SEED_RE.search(line)
        if not seed_match:
            continue
        seed = int(seed_match.group(1))
        if seed not in out:
            continue

        keep, transformed = include_line(line, args.heavy_size)
        if keep:
            out[seed].append(transformed)

    for seed in target_seeds:
        output_path = source.parent / f"{args.output_prefix}{seed}.txt"
        content = "\n".join(out[seed])
        output_path.write_text(content, encoding="utf-8")
        print(f"seed={seed} jobs={len(out[seed])} file={output_path.name}")


if __name__ == "__main__":
    main()
