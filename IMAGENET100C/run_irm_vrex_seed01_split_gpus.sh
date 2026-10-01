#!/usr/bin/env bash
# Run IRM/VREx for seed 0 on GPU 0 and seed 1 on GPU 1.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="/home/ra95tig/anaconda3/envs/domgen/bin/python"
results_root="/home/ra95tig/results_final_imagenet100c_seed01_retry"
cd "$repo_root"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
mkdir -p "$results_root/logs"

run_seed() {
    local gpu="$1"
    local seed="$2"
    local condition algorithm output_dir checkpoint log_file
    for condition in balanced long_tail scarce_tail missing; do
        for algorithm in irm vrex; do
            output_dir="$results_root/seed${seed}/E3b_${condition}_${algorithm}"
            checkpoint="$output_dir/checkpoints/final.pt"
            log_file="$results_root/logs/E3b_${condition}_${algorithm}_seed${seed}.log"
            if [[ -f "$checkpoint" ]]; then
                printf 'SKIP GPU=%s seed=%s condition=%s algorithm=%s\n' "$gpu" "$seed" "$condition" "$algorithm"
                continue
            fi
            if [[ -e "$output_dir" ]]; then
                printf 'ERROR incomplete output directory: %s\n' "$output_dir" >&2
                return 1
            fi
            printf 'START GPU=%s seed=%s condition=%s algorithm=%s\n' "$gpu" "$seed" "$condition" "$algorithm"
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
            printf 'DONE GPU=%s seed=%s condition=%s algorithm=%s\n' "$gpu" "$seed" "$condition" "$algorithm"
        done
    done
}

run_seed 0 0 & pid0=$!
run_seed 1 1 & pid1=$!
status=0
wait "$pid0" || status=1
wait "$pid1" || status=1
if [[ $status -ne 0 ]]; then
    echo "One or more seed lanes failed; completed runs are preserved." >&2
    exit 1
fi
printf 'ALL SEED 0/1 IRM/VREx RUNS COMPLETE\n'
