#!/usr/bin/env python3
"""Extract completed jobs from log and comment them out in command file."""

from __future__ import annotations

import re
from pathlib import Path


def parse_log_jobs(log_path: Path) -> list[dict[str, str]]:
    """Extract completed job parameters from log."""
    jobs = []
    content = log_path.read_text(encoding="utf-8")
    
    # Find all "best accuracies:" markers (indicates job completed)
    best_acc_indices = []
    for m in re.finditer(r"best accuracies:", content):
        best_acc_indices.append(m.start())
    
    print(f"Found {len(best_acc_indices)} 'best accuracies:' markers in log")
    
    for idx in best_acc_indices:
        # Search backwards for the most recent "Args:" block
        args_idx = content.rfind("Args:\n\t", 0, idx)
        if args_idx == -1:
            continue
        
        # Extract from "Args:" to the first non-indented line
        args_block = content[args_idx:idx]
        lines = args_block.split("\n")[1:]  # Skip "Args:" line
        job_params = {}
        
        for line in lines:
            if not line.startswith("\t"):
                break
            line = line.strip()
            if ":" in line:
                key, val = line.split(":", 1)
                job_params[key.strip()] = val.strip()
        
        # Extract identifying fields
        algo = job_params.get("algorithm", "").lower()
        seed = job_params.get("seed", "")
        steps = job_params.get("steps", "")
        train_envs = job_params.get("train_envs", "")
        train_sizes = job_params.get("train_env_sizes", "")
        
        if seed == "0" and algo and steps and train_envs:
            jobs.append({
                "algorithm": algo,
                "steps": steps,
                "train_envs": train_envs,
                "train_env_sizes": train_sizes,
            })
            print(f"  Job: {algo} steps={steps} envs={train_envs} sizes={train_sizes if train_sizes else '(none)'}")
        
    return jobs


def match_command(cmd: str, job_params: dict[str, str]) -> bool:
    """Check if command matches job parameters."""
    algo_re = rf"--algorithm\s+{job_params['algorithm']}\b"
    steps_re = rf"--steps\s+{job_params['steps']}\b"
    envs_re = rf"--train_envs\s+{re.escape(job_params['train_envs'])}\b"
    
    if not (re.search(algo_re, cmd) and re.search(steps_re, cmd) and re.search(envs_re, cmd)):
        return False
    
    # If sizes are specified, must match
    if job_params["train_env_sizes"]:
        sizes_re = rf"--train_env_sizes\s+{re.escape(job_params['train_env_sizes'])}\b"
        return bool(re.search(sizes_re, cmd))
    
    # If sizes not in log, command should also not have them (or have empty string)
    return "--train_env_sizes" not in cmd or "--train_env_sizes\s+$" in cmd


def main() -> None:
    log_path = Path("../../domain_stress_main_seed0.log").resolve()
    cmd_path = Path("domain_stress_main_seed0.txt").resolve()
    
    if not log_path.exists():
        raise FileNotFoundError(f"Log not found: {log_path}")
    if not cmd_path.exists():
        raise FileNotFoundError(f"Command file not found: {cmd_path}")
    
    # Parse completed jobs
    completed = parse_log_jobs(log_path)
    print(f"Found {len(completed)} completed jobs for seed 0 in log.")
    
    # Read command file
    lines = cmd_path.read_text(encoding="utf-8").splitlines(keepends=True)
    
    # Match and comment out
    commented = 0
    for i, line in enumerate(lines):
        for job in completed:
            if match_command(line, job):
                if not line.strip().startswith("#"):
                    lines[i] = "# " + line
                    commented += 1
                    print(f"Commented: {line.strip()[:100]}")
                break
    
    # Write back
    cmd_path.write_text("".join(lines), encoding="utf-8")
    print(f"\nCommented out {commented} lines.")
    print(f"Remaining jobs: {len([l for l in lines if l.strip() and not l.strip().startswith('#')])}")


if __name__ == "__main__":
    main()
