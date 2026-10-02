#!/usr/bin/env bash
# Run 36 ImageNet E3b GroupDRO/IRO/INF-TASK jobs with 8 workers per GPU.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="/home/ra95tig/anaconda3/envs/domgen/bin/python"
results_root="/home/ra95tig/results_final_imagenet100c_groupdro_iro_inftask"
gpus=(0 1)
workers_per_gpu=8
cd "$repo_root"
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
mkdir -p "$results_root/logs"

runs=()
for seed in 0 1 2; do
  for condition in balanced long_tail scarce_tail missing; do
    for algorithm in groupdro iro inftask; do
      runs+=("$seed|$condition|$algorithm")
    done
  done
done

total_runs="${#runs[@]}"
run_one() {
  local gpu="$1" index="$2" seed condition algorithm output log
  IFS='|' read -r seed condition algorithm <<< "${runs[index]}"
  output="$results_root/seed${seed}/E3b_${condition}_${algorithm}"
  log="$results_root/logs/E3b_${condition}_${algorithm}_seed${seed}.log"
  if [[ -f "$output/checkpoints/final.pt" ]]; then
    printf 'SKIP [%02d/%d] GPU=%s seed=%s condition=%s algorithm=%s\n' "$((index+1))" "$total_runs" "$gpu" "$seed" "$condition" "$algorithm"
    return 0
  fi
  if [[ -e "$output" ]]; then
    printf 'ERROR incomplete output directory: %s\n' "$output" >&2
    return 1
  fi
  printf 'START [%02d/%d] GPU=%s seed=%s condition=%s algorithm=%s\n' "$((index+1))" "$total_runs" "$gpu" "$seed" "$condition" "$algorithm"
  if ! CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -m IMAGENET100C.train \
      --seed "$seed" --algorithm "$algorithm" --backbone_mode finetune_last_stage \
      --experiment E3b --condition "$condition" --steps 1000 --batch_size 64 \
      --num_lambda_samples 4 --eqrm_alpha 0.9 --penalty_weight 1000 \
      --erm_pretrain_iters 400 --lr_cos_sched --workers 1 --pin_memory \
      --persistent_workers --prefetch_factor 2 --checkpoint_selection final \
      --output_dir "$output" >"$log" 2>&1; then
    printf 'ERROR [%02d/%d] seed=%s condition=%s algorithm=%s; see %s\n' "$((index+1))" "$total_runs" "$seed" "$condition" "$algorithm" "$log" >&2
    return 1
  fi
  [[ -f "$output/checkpoints/final.pt" ]] || { echo "Missing checkpoint: $output" >&2; return 1; }
  printf 'DONE  [%02d/%d] GPU=%s seed=%s condition=%s algorithm=%s\n' "$((index+1))" "$total_runs" "$gpu" "$seed" "$condition" "$algorithm"
}

declare -A pid_gpu
next_run=0
for gpu in "${gpus[@]}"; do
  for ((worker=0; worker<workers_per_gpu && next_run<total_runs; worker++)); do
    run_one "$gpu" "$next_run" &
    pid_gpu[$!]="$gpu"
    ((next_run+=1))
  done
done

status=0
while [[ ${#pid_gpu[@]} -gt 0 ]]; do
  completed_pid=""
  if wait -n -p completed_pid "${!pid_gpu[@]}"; then
    gpu="${pid_gpu[$completed_pid]}"
    unset "pid_gpu[$completed_pid]"
    if [[ $next_run -lt $total_runs ]]; then
      run_one "$gpu" "$next_run" &
      pid_gpu[$!]="$gpu"
      ((next_run+=1))
    fi
  else
    status=1
    gpu="${pid_gpu[$completed_pid]:-unknown}"
    unset "pid_gpu[$completed_pid]"
    printf 'FAILED PID=%s GPU=%s; continuing queue\n' "$completed_pid" "$gpu" >&2
    if [[ $next_run -lt $total_runs ]]; then
      run_one "$gpu" "$next_run" &
      pid_gpu[$!]="$gpu"
      ((next_run+=1))
    fi
  fi
done

if [[ $status -ne 0 ]]; then
  echo "One or more ImageNet runs failed; completed runs are preserved." >&2
  exit 1
fi
printf 'ALL %d GROUPDRO/IRO/INF-TASK RUNS COMPLETE\n' "$total_runs"
