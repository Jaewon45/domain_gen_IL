#!/usr/bin/env bash
# Run source-only IRM/VREx calibration; seed 42 is the default, not a stored artifact.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_bin="python"
manifest=""
checkpoint=""
output_root="results/imagenet100c_source_calibration"
seed=42
methods="irm,vrex"
penalties="0.1,0.3,1,3"
steps=100
num_lambda_samples=4
iro_sampler_learning_rate=1e-6
while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) python_bin="$2"; shift 2 ;;
    --manifest) manifest="$2"; shift 2 ;;
    --checkpoint) checkpoint="$2"; shift 2 ;;
    --output-root) output_root="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --methods) methods="$2"; shift 2 ;;
    --penalties) penalties="$2"; shift 2 ;;
    --steps) steps="$2"; shift 2 ;;
    --num-lambda-samples) num_lambda_samples="$2"; shift 2 ;;
    --iro-sampler-learning-rate) iro_sampler_learning_rate="$2"; shift 2 ;;
    -h|--help) echo "Usage: $0 --manifest FILE --checkpoint FILE [--seed 42] [--penalties 0.1,0.3,1,3]"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$manifest" && -f "$manifest" ]] || { echo "--manifest is required" >&2; exit 2; }
[[ -n "$checkpoint" && -f "$checkpoint" ]] || { echo "--checkpoint is required" >&2; exit 2; }
[[ "$seed" =~ ^[0-9]+$ ]] || { echo "--seed must be numeric" >&2; exit 2; }
[[ "$output_root" = /* ]] || output_root="$repo_root/$output_root"
IFS=',' read -r -a method_list <<< "$methods"
IFS=',' read -r -a penalty_list <<< "$penalties"
for method in "${method_list[@]}"; do
  for penalty in "${penalty_list[@]}"; do
    tag="${penalty//./p}"
    output_dir="$output_root/seed${seed}_${method}_penalty${tag}"
    [[ -e "$output_dir" ]] && { echo "Output exists: $output_dir" >&2; exit 1; }
    "$python_bin" -m IMAGENET100C.investigate_VRExIRM.probe \
      --manifest "$manifest" --checkpoint "$checkpoint" --output-dir "$output_dir" \
      --method "$method" --penalty-weight "$penalty" --steps "$steps" --seed "$seed" \
      --num-lambda-samples "$num_lambda_samples" \
      --iro-sampler-learning-rate "$iro_sampler_learning_rate"
  done
done
