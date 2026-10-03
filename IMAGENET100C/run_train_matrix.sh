#!/usr/bin/env bash
# Reusable ImageNet100C training launcher; outputs are selected by CLI.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_bin="python"
results_root="results/ImgNet/e3b"
logs_root="logs/ImgNet/e3b"
gpus="0"
seeds="0,1,2"
conditions="balanced,long_tail,scarce_tail,missing"
algorithms="erm,irm,vrex,eqrm,groupdro,inftask,iro"
steps=1000
batch_size=64
workers=1
num_lambda_samples=4
iro_sampler_learning_rate=1e-6
erm_pretrain_iters=400
lr_cos_sched=1
erm_shared_schedule=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) python_bin="$2"; shift 2 ;;
    --results-root) results_root="$2"; shift 2 ;;
    --logs-root) logs_root="$2"; shift 2 ;;
    --gpus) gpus="$2"; shift 2 ;;
    --seeds) seeds="$2"; shift 2 ;;
    --conditions) conditions="$2"; shift 2 ;;
    --algorithms) algorithms="$2"; shift 2 ;;
    --steps) steps="$2"; shift 2 ;;
    --batch-size) batch_size="$2"; shift 2 ;;
    --workers) workers="$2"; shift 2 ;;
    --num-lambda-samples) num_lambda_samples="$2"; shift 2 ;;
    --iro-sampler-learning-rate) iro_sampler_learning_rate="$2"; shift 2 ;;
    --erm-pretrain-iters) erm_pretrain_iters="$2"; shift 2 ;;
    --lr-cos-sched) lr_cos_sched=1; shift ;;
    --no-lr-cos-sched) lr_cos_sched=0; shift ;;
    --erm-shared-schedule) erm_shared_schedule=1; shift ;;
    --no-erm-shared-schedule) erm_shared_schedule=0; shift ;;
    -h|--help)
      sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ "$results_root" = /* ]] || results_root="$repo_root/$results_root"
[[ "$logs_root" = /* ]] || logs_root="$repo_root/$logs_root"
mkdir -p "$results_root"
IFS=',' read -r -a gpu_list <<< "$gpus"
IFS=',' read -r -a seed_list <<< "$seeds"
IFS=',' read -r -a condition_list <<< "$conditions"
IFS=',' read -r -a algorithm_list <<< "$algorithms"
runs=()
for seed in "${seed_list[@]}"; do
  for condition in "${condition_list[@]}"; do
    for algorithm in "${algorithm_list[@]}"; do runs+=("$seed|$condition|$algorithm"); done
  done
done
run_one() {
  local gpu="$1" spec="$2" seed condition algorithm output_dir log_file
  IFS='|' read -r seed condition algorithm <<< "$spec"
  output_dir="$results_root/seed${seed}/E3b_${condition}_${algorithm}"
  log_file="$logs_root/seed${seed}_E3b_${condition}_${algorithm}.log"
  [[ -f "$output_dir/checkpoints/final.pt" ]] && return 0
  [[ ! -e "$output_dir" ]] || { echo "Incomplete output exists: $output_dir" >&2; return 1; }
  method_pretrain_iters="$erm_pretrain_iters"
  method_lr_cos_sched="$lr_cos_sched"
  method_shared_schedule="$erm_shared_schedule"
  case "$algorithm" in
    erm)
      method_pretrain_iters=0
      method_lr_cos_sched=0
      method_shared_schedule=0
      ;;
    irm|vrex|eqrm)
      method_pretrain_iters=400
      method_lr_cos_sched=1
      method_shared_schedule=0
      ;;
    groupdro|iro|inftask)
      method_pretrain_iters=0
      method_lr_cos_sched=1
      method_shared_schedule=0
      ;;
  esac
  args=(--seed "$seed" --algorithm "$algorithm" --backbone_mode finetune_last_stage --experiment E3b --condition "$condition" --steps "$steps" --batch_size "$batch_size" --num_lambda_samples "$num_lambda_samples" --iro_sampler_learning_rate "$iro_sampler_learning_rate" --eqrm_alpha 0.9 --penalty_weight 1000 --erm_pretrain_iters "$method_pretrain_iters" --workers "$workers" --checkpoint_selection final --output_dir "$output_dir")
  [[ "$method_lr_cos_sched" -eq 1 ]] && args+=(--lr_cos_sched)
  [[ "$method_shared_schedule" -eq 1 ]] && args+=(--erm_shared_schedule)
  mkdir -p "$(dirname "$log_file")"
  CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -m IMAGENET100C.train "${args[@]}" >"$log_file" 2>&1
}
active=0
next_run=0
failed=0
while [[ "$next_run" -lt "${#runs[@]}" || "$active" -gt 0 ]]; do
  while [[ "$next_run" -lt "${#runs[@]}" && "$active" -lt $(( ${#gpu_list[@]} )) ]]; do
    gpu="${gpu_list[$((next_run % ${#gpu_list[@]}))]}"
    run_one "$gpu" "${runs[next_run]}" &
    active=$((active + 1)); next_run=$((next_run + 1))
  done
  wait -n || failed=$((failed + 1))
  active=$((active - 1))
done
[[ "$failed" -eq 0 ]] || exit 1
