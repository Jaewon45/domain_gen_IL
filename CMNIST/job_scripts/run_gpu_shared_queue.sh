#!/usr/bin/env bash
# Run a generated CMNIST command file with a fixed number of processes per GPU.
set -uo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 COMMAND_FILE GPU_IDS WORKERS_PER_GPU [SKIP_LINES]" >&2
  exit 2
fi

command_file="$1"
gpu_ids="$2"
workers_per_gpu="$3"
skip_lines="${4:-0}"
command_dir="$(cd "$(dirname "$command_file")" && pwd)"
run_dir="$(cd "$command_dir/.." && pwd)"
command_file="$command_dir/$(basename "$command_file")"

IFS=',' read -r -a gpus <<< "$gpu_ids"
if [[ ${#gpus[@]} -eq 0 || "$workers_per_gpu" -lt 1 ]]; then
  echo "At least one GPU and one worker per GPU are required." >&2
  exit 2
fi

log_dir="$(dirname "$command_file")/logs_$(basename "$command_file" .txt)"
mkdir -p "$log_dir"
mapfile -t commands < "$command_file"
active=0
failed=0
slot=0

for index in "${!commands[@]}"; do
  if [[ "$index" -lt "$skip_lines" || -z "${commands[$index]// /}" ]]; then
    continue
  fi
  if [[ "$active" -ge $(( ${#gpus[@]} * workers_per_gpu )) ]]; then
    if ! wait -n; then
      failed=$((failed + 1))
    fi
    active=$((active - 1))
  fi
  gpu="${gpus[$((slot % ${#gpus[@]}))]}"
  echo "Launching command $((index + 1)) on GPU ${gpu}"
  (
    cd "$run_dir"
    CUDA_VISIBLE_DEVICES="$gpu" bash -c "${commands[$index]}"
  ) > "$log_dir/job_$((index + 1)).log" 2>&1 &
  active=$((active + 1))
  slot=$((slot + 1))
done

while [[ "$active" -gt 0 ]]; do
  if ! wait -n; then
    failed=$((failed + 1))
  fi
  active=$((active - 1))
done

if [[ "$failed" -gt 0 ]]; then
  echo "$failed job(s) failed; inspect $log_dir." >&2
  exit 1
fi
echo "All queued jobs completed. Logs: $log_dir"
