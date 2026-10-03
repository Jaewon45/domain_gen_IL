#!/usr/bin/env bash
# Evaluate all final checkpoints for one ImageNet100C seed.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_bin="python"
results_root="results/ImgNet/e3b"
logs_root="logs/ImgNet/e3b"
seed=""
gpus="0"
batch_size=64
workers=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) python_bin="$2"; shift 2 ;;
    --results-root) results_root="$2"; shift 2 ;;
    --logs-root) logs_root="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --gpus) gpus="$2"; shift 2 ;;
    --batch-size) batch_size="$2"; shift 2 ;;
    --workers) workers="$2"; shift 2 ;;
    -h|--help) echo "Usage: $0 --seed N [--results-root DIR] [--gpus 0,1] [--python PATH]"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ "$seed" =~ ^[0-9]+$ ]] || { echo "--seed must be a non-negative integer" >&2; exit 2; }
[[ "$results_root" = /* ]] || results_root="$repo_root/$results_root"
[[ "$logs_root" = /* ]] || logs_root="$repo_root/$logs_root"
seed_root="$results_root/seed${seed}"
mapfile -t checkpoints < <(find "$seed_root" -mindepth 3 -maxdepth 3 -type f -path '*/checkpoints/final.pt' | sort)
[[ "${#checkpoints[@]}" -gt 0 ]] || { echo "No final checkpoints found under $seed_root" >&2; exit 1; }
IFS=',' read -r -a gpu_list <<< "$gpus"
for index in "${!checkpoints[@]}"; do
  checkpoint="${checkpoints[index]}"
  run_dir="$(dirname "$(dirname "$checkpoint")")"
  output_dir="$run_dir/evaluation_anchor1000"
  log_file="$logs_root/seed${seed}_$(basename "$run_dir")_evaluation.log"
  [[ -f "$output_dir/evaluation.jsonl" ]] && continue
  gpu="${gpu_list[$((index % ${#gpu_list[@]}))]}"
  mkdir -p "$output_dir"
  mkdir -p "$(dirname "$log_file")"
  CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -m IMAGENET100C.evaluate "$checkpoint" \
    --corruption_types gaussian_noise,defocus_blur,snow,contrast \
    --max_eval_images 1000 --batch_size "$batch_size" --workers "$workers" \
    --output_dir "$output_dir" >"$log_file" 2>&1 &
done
wait
