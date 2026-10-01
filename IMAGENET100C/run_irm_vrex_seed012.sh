#!/usr/bin/env bash
# Run the canonical ImageNet-100-C IRM/VREx seed 0-2 matrix.
set -euo pipefail

results_root="/home/ra95tig/results_final_imagenet100c"
python_bin="/home/ra95tig/anaconda3/envs/domgen/bin/python"
gpus=(0 1)
workers_per_gpu=4
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
mkdir -p "$results_root/logs"

runs=()
for seed in 0 1 2; do
    for condition in balanced long_tail scarce_tail missing; do
        for algorithm in irm vrex; do
            runs+=("$seed|$condition|$algorithm")
        done
    done
done

total_runs="${#runs[@]}"
run_one() {
    local gpu="$1" index="$2"
    local seed condition algorithm output_dir checkpoint log_file
    IFS='|' read -r seed condition algorithm <<< "${runs[index]}"
    output_dir="$results_root/seed${seed}/E3b_${condition}_${algorithm}"
    checkpoint="$output_dir/checkpoints/final.pt"
    log_file="$results_root/logs/E3b_${condition}_${algorithm}_seed${seed}.log"
    if [[ -f "$checkpoint" ]]; then
        printf 'SKIP  [%02d/%d] GPU %s %s\n' "$((index + 1))" "$total_runs" "$gpu" "$checkpoint"
        return 0
    fi
    if [[ -e "$output_dir" ]]; then
        printf 'ERROR incomplete output directory: %s\n' "$output_dir" >&2
        return 1
    fi
    printf 'START [%02d/%d] GPU %s seed=%s condition=%s algorithm=%s\n' \
        "$((index + 1))" "$total_runs" "$gpu" "$seed" "$condition" "$algorithm"
    if ! CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -m IMAGENET100C.train \
        --seed "$seed" \
        --algorithm "$algorithm" \
        --backbone_mode finetune_last_stage \
        --experiment E3b \
        --condition "$condition" \
        --steps 1000 \
        --batch_size 64 \
        --num_lambda_samples 4 \
        --alpha 0.9 \
        --penalty_weight 1000 \
        --erm_pretrain_iters 400 \
        --lr_cos_sched \
        --workers 1 \
        --pin_memory \
        --persistent_workers \
        --prefetch_factor 2 \
        --checkpoint_selection final \
        --output_dir "$output_dir" >"$log_file" 2>&1; then
        printf 'ERROR seed=%s condition=%s algorithm=%s; see %s\n' "$seed" "$condition" "$algorithm" "$log_file" >&2
        return 1
    fi
    [[ -f "$checkpoint" ]] || { echo "Missing checkpoint: $checkpoint" >&2; return 1; }
    printf 'DONE  [%02d/%d] GPU %s seed=%s condition=%s algorithm=%s\n' \
        "$((index + 1))" "$total_runs" "$gpu" "$seed" "$condition" "$algorithm"
}

declare -A pid_gpu
next_run=0
for gpu in "${gpus[@]}"; do
    for ((worker = 0; worker < workers_per_gpu && next_run < total_runs; worker++)); do
        run_one "$gpu" "$next_run" &
        pid_gpu[$!]="$gpu"
        ((next_run += 1))
    done
done

status=0
while [[ ${#pid_gpu[@]} -gt 0 ]]; do
    completed_pid=""
    if wait -n -p completed_pid "${!pid_gpu[@]}"; then
        gpu="${pid_gpu[$completed_pid]}"
        unset "pid_gpu[$completed_pid]"
        if [[ $next_run -lt total_runs ]]; then
            run_one "$gpu" "$next_run" &
            pid_gpu[$!]="$gpu"
            ((next_run += 1))
        fi
    else
        status=1
        break
    fi
done

if [[ $status -ne 0 ]]; then
    for worker_pid in "${!pid_gpu[@]}"; do wait "$worker_pid" || true; done
    echo "One or more ImageNet runs failed; completed runs are preserved." >&2
    exit 1
fi
printf 'ALL %d IRM/VREx TRAINING RUNS COMPLETE\n' "$total_runs"
