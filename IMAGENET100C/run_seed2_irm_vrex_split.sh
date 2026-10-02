#!/usr/bin/env bash
# Run seed 2 IRM on GPU 0 and VREx on GPU 1 across all E3b conditions.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="/home/ra95tig/anaconda3/envs/domgen/bin/python"
results_root="/home/ra95tig/results_final_imagenet100c_seed2_irm_vrex"
cd "$repo_root"
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
mkdir -p "$results_root/logs"
run_lane() {
    local gpu="$1" algorithm="$2" condition output checkpoint log_file
    for condition in balanced long_tail scarce_tail missing; do
        output="$results_root/seed2/E3b_${condition}_${algorithm}"
        checkpoint="$output/checkpoints/final.pt"
        log_file="$results_root/logs/E3b_${condition}_${algorithm}_seed2.log"
        if [[ -f "$checkpoint" ]]; then continue; fi
        if [[ -e "$output" ]]; then echo "Incomplete output exists: $output" >&2; return 1; fi
        echo "START GPU=$gpu seed=2 condition=$condition algorithm=$algorithm"
        CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -m IMAGENET100C.train \
            --seed 2 --algorithm "$algorithm" --backbone_mode finetune_last_stage \
            --experiment E3b --condition "$condition" --steps 1000 --batch_size 64 \
            --num_lambda_samples 4 --alpha 0.9 --penalty_weight 1000 \
            --erm_pretrain_iters 400 --lr_cos_sched --workers 1 --pin_memory \
            --persistent_workers --prefetch_factor 2 --checkpoint_selection final \
            --output_dir "$output" >"$log_file" 2>&1
        [[ -f "$checkpoint" ]] || { echo "Missing checkpoint: $checkpoint" >&2; return 1; }
        echo "DONE GPU=$gpu seed=2 condition=$condition algorithm=$algorithm"
    done
}
run_lane 0 irm & p0=$!
run_lane 1 vrex & p1=$!
status=0; wait "$p0" || status=1; wait "$p1" || status=1
exit "$status"
