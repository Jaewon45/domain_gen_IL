#!/usr/bin/env bash
# Run a generated CMNIST command file with a configurable GPU queue.
set -euo pipefail

command_file=""
gpus="0"
workers_per_gpu=1
skip_lines=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --command-file) command_file="$2"; shift 2 ;;
    --gpus) gpus="$2"; shift 2 ;;
    --workers-per-gpu) workers_per_gpu="$2"; shift 2 ;;
    --skip-lines) skip_lines="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 --command-file FILE [--gpus 0,1] [--workers-per-gpu N] [--skip-lines N]"
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$command_file" && -f "$command_file" ]] || { echo "Missing --command-file" >&2; exit 2; }
[[ "$workers_per_gpu" =~ ^[1-9][0-9]*$ && "$skip_lines" =~ ^[0-9]+$ ]] || { echo "Invalid worker or skip count" >&2; exit 2; }
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
IFS=',' read -r -a gpu_list <<< "$gpus"
mapfile -t commands < "$command_file"
log_dir="${command_file%.*}_logs"
mkdir -p "$log_dir"
active=0
failed=0
slot=0
for index in "${!commands[@]}"; do
  [[ "$index" -ge "$skip_lines" && -n "${commands[index]// /}" ]] || continue
  while [[ "$active" -ge $(( ${#gpu_list[@]} * workers_per_gpu )) ]]; do
    wait -n || failed=$((failed + 1))
    active=$((active - 1))
  done
  gpu="${gpu_list[$((slot % ${#gpu_list[@]}))]}"
  ( CUDA_VISIBLE_DEVICES="$gpu" bash -lc "${commands[index]}" ) >"$log_dir/job_$((index + 1)).log" 2>&1 &
  active=$((active + 1))
  slot=$((slot + 1))
done
while [[ "$active" -gt 0 ]]; do
  wait -n || failed=$((failed + 1))
  active=$((active - 1))
done
[[ "$failed" -eq 0 ]] || { echo "$failed command(s) failed; inspect $log_dir" >&2; exit 1; }
