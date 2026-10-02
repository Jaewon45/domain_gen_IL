#!/usr/bin/env bash
# Evaluate seed 0 on GPU 0 and seed 1 on GPU 1, eight evaluators per GPU.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="/home/ra95tig/anaconda3/envs/domgen/bin/python"
results_root="/home/ra95tig/results_final_imagenet100c_seed01_8workers"
cd "$repo_root"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
mkdir -p "$results_root/logs"

run_one() {
    local gpu="$1" seed="$2" condition="$3" algorithm="$4"
    local checkpoint="$results_root/seed${seed}/E3b_${condition}_${algorithm}/checkpoints/final.pt"
    local output="$results_root/seed${seed}/E3b_${condition}_${algorithm}/evaluation_minimal_anchor1000"
    local log_file="$results_root/logs/eval_${condition}_${algorithm}_seed${seed}.log"
    if [[ -f "$output/evaluation.jsonl" ]]; then
        printf 'SKIP GPU=%s seed=%s condition=%s algorithm=%s\n' "$gpu" "$seed" "$condition" "$algorithm"
        return 0
    fi
    if [[ -e "$output" ]]; then
        printf 'ERROR incomplete evaluation directory: %s\n' "$output" >&2
        return 1
    fi
    printf 'START GPU=%s seed=%s condition=%s algorithm=%s\n' "$gpu" "$seed" "$condition" "$algorithm"
    if ! CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -m IMAGENET100C.evaluate "$checkpoint" \
        --output_dir "$output" --batch_size 64 --max_eval_images 1000 \
        --corruption_types gaussian_noise,defocus_blur,snow,contrast \
        --workers 1 --pin_memory >"$log_file" 2>&1; then
        printf 'ERROR seed=%s condition=%s algorithm=%s; see %s\n' "$seed" "$condition" "$algorithm" "$log_file" >&2
        return 1
    fi
    [[ -f "$output/evaluation.jsonl" ]] || { echo "Missing evaluation: $output" >&2; return 1; }
    printf 'DONE GPU=%s seed=%s condition=%s algorithm=%s\n' "$gpu" "$seed" "$condition" "$algorithm"
}

worker_lane() {
    local gpu="$1" seed="$2" worker_index="$3" index condition algorithm
    jobs=("balanced|irm" "balanced|vrex" "long_tail|irm" "long_tail|vrex" \
          "scarce_tail|irm" "scarce_tail|vrex" "missing|irm" "missing|vrex")
    for ((index = worker_index; index < ${#jobs[@]}; index += 8)); do
        IFS='|' read -r condition algorithm <<< "${jobs[index]}"
        run_one "$gpu" "$seed" "$condition" "$algorithm" || return 1
    done
}

status=0
for worker in $(seq 0 7); do worker_lane 0 0 "$worker" & done
for worker in $(seq 0 7); do worker_lane 1 1 "$worker" & done
for worker_pid in $(jobs -p); do wait "$worker_pid" || status=1; done
if [[ $status -ne 0 ]]; then
    echo "One or more evaluations failed; completed evaluations are preserved." >&2
    exit 1
fi
printf 'ALL 16 NEAR-MISSING EVALUATIONS COMPLETE\n'
