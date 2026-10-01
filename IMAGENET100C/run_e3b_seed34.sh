#!/usr/bin/env bash
# Run the 56 E3b ImageNet-100-C training jobs for seeds 3 and 4 across 2 GPUs (6 workers per GPU = 12 concurrent).
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: run_e3b_seed34.sh [options]

Options:
  --results-root PATH         Output root (default: /home/ra95tig/imagenet100c_results)
  --python PATH               Python executable (default: /home/ra95tig/anaconda3/envs/domgen/bin/python)
  --gpus IDS                  Comma-separated CUDA device IDs (default: 0,1)
  --workers-per-gpu N         Concurrent processes per GPU (default: 6)
  --backbone-mode MODE        finetune_last_stage (default) or frozen_feature_pilot
  --dry-run                   Print the planned 56 runs and exit
  -h, --help                  Show this message
EOF
}

results_root="/home/ra95tig/imagenet100c_results"
python_bin="/home/ra95tig/anaconda3/envs/domgen/bin/python"
gpu_csv="0,1"
workers_per_gpu=6
backbone_mode="finetune_last_stage"
dry_run=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --results-root) results_root="$2"; shift 2 ;;
        --python) python_bin="$2"; shift 2 ;;
        --gpus) gpu_csv="$2"; shift 2 ;;
        --workers-per-gpu) workers_per_gpu="$2"; shift 2 ;;
        --backbone-mode) backbone_mode="$2"; shift 2 ;;
        --dry-run) dry_run=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

IFS=',' read -r -a gpus <<< "$gpu_csv"
if [[ ${#gpus[@]} -eq 0 || -z "${gpus[0]}" ]]; then
    echo "--gpus must list at least one CUDA device ID" >&2
    exit 2
fi
for gpu in "${gpus[@]}"; do
    [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "Invalid CUDA device ID: $gpu" >&2; exit 2; }
done
[[ "$workers_per_gpu" =~ ^[1-9][0-9]*$ ]] || { echo "--workers-per-gpu must be a positive integer" >&2; exit 2; }

command -v "$python_bin" >/dev/null || { echo "Python executable not found: $python_bin" >&2; exit 2; }

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

seeds=(3 4)
conditions=(balanced long_tail near_missing missing)
algorithms=(erm irm vrex eqrm groupdro inftask iro)

runs=()
for seed in "${seeds[@]}"; do
    for condition in "${conditions[@]}"; do
        for algorithm in "${algorithms[@]}"; do
            runs+=("$seed|$condition|$algorithm")
        done
    done
done

total_runs="${#runs[@]}"
total_workers=$((${#gpus[@]} * workers_per_gpu))

if [[ $dry_run -eq 1 ]]; then
    printf 'Planned runs: %d across GPUs: %s (%s workers/GPU = %d concurrent)\n' \
        "$total_runs" "$gpu_csv" "$workers_per_gpu" "$total_workers"
    for i in "${!runs[@]}"; do
        IFS='|' read -r s cond algo <<< "${runs[i]}"
        printf '[%02d/%d] seed=%s condition=%s algorithm=%s\n' "$((i + 1))" "$total_runs" "$s" "$cond" "$algo"
    done
    exit 0
fi

mkdir -p "$results_root/seed3/logs" "$results_root/seed4/logs" "$repo_root/logs"

run_one() {
    local gpu="$1"
    local run_index="$2"
    local seed condition algorithm output_dir checkpoint log_file
    IFS='|' read -r seed condition algorithm <<< "${runs[run_index]}"

    output_dir="$results_root/seed${seed}/E3b_${condition}_${algorithm}"
    checkpoint="$output_dir/checkpoints/final.pt"
    log_file="$results_root/seed${seed}/logs/E3b_${condition}_${algorithm}.log"

    if [[ -f "$checkpoint" ]]; then
        printf 'SKIP  %s: [%02d/%d] GPU %s %s\n' "$(date --iso-8601=seconds)" "$((run_index + 1))" "$total_runs" "$gpu" "$checkpoint"
        return 0
    fi
    if [[ -e "$output_dir" ]]; then
        printf 'ERROR %s: incomplete output directory: %s\n' "$(date --iso-8601=seconds)" "$output_dir" >&2
        return 1
    fi

    printf 'START %s: [%02d/%d] GPU %s seed=%s condition=%s algorithm=%s\n' \
        "$(date --iso-8601=seconds)" "$((run_index + 1))" "$total_runs" "$gpu" "$seed" "$condition" "$algorithm"

    if ! CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -m IMAGENET100C.train \
        --experiment E3b \
        --condition "$condition" \
        --algorithm "$algorithm" \
        --backbone_mode "$backbone_mode" \
        --seed "$seed" \
        --steps 1000 \
        --batch_size 64 \
        --num_lambda_samples 4 \
        --alpha 0.75 \
        --checkpoint_selection final \
        --output_dir "$output_dir" >"$log_file" 2>&1; then
        printf 'ERROR %s: [%02d/%d] failed on GPU %s; see %s\n' \
            "$(date --iso-8601=seconds)" "$((run_index + 1))" "$total_runs" "$gpu" "$log_file" >&2
        return 1
    fi

    if [[ ! -f "$checkpoint" ]]; then
        printf 'ERROR %s: missing final checkpoint: %s\n' "$(date --iso-8601=seconds)" "$checkpoint" >&2
        return 1
    fi

    printf 'DONE  %s: [%02d/%d] GPU %s seed=%s condition=%s algorithm=%s\n' \
        "$(date --iso-8601=seconds)" "$((run_index + 1))" "$total_runs" "$gpu" "$seed" "$condition" "$algorithm"
}

printf '=== Launching %d runs with %d workers (%d per GPU on GPUs %s) ===\n' \
    "$total_runs" "$total_workers" "$workers_per_gpu" "$gpu_csv"

declare -A pid_gpu
next_run=0

# Seed initial workers
for gpu in "${gpus[@]}"; do
    for ((w = 0; w < workers_per_gpu && next_run < total_runs; w++)); do
        run_one "$gpu" "$next_run" &
        pid_gpu[$!]="$gpu"
        ((next_run += 1))
    done
done

status=0
while [[ ${#pid_gpu[@]} -gt 0 ]]; do
    completed_pid=""
    if wait -n -p completed_pid "${!pid_gpu[@]}"; then
        ret=$?
        gpu="${pid_gpu[$completed_pid]}"
        unset "pid_gpu[$completed_pid]"
        if [[ $ret -ne 0 ]]; then
            printf 'Process PID %s on GPU %s failed with exit code %s\n' "$completed_pid" "$gpu" "$ret" >&2
            status=1
        fi
        if [[ $next_run -lt total_runs ]]; then
            run_one "$gpu" "$next_run" &
            pid_gpu[$!]="$gpu"
            ((next_run += 1))
        fi
    else
        ret=$?
        gpu="${pid_gpu[$completed_pid]:-unknown}"
        unset "pid_gpu[$completed_pid]"
        printf 'Process PID %s on GPU %s failed with exit code %s\n' "$completed_pid" "$gpu" "$ret" >&2
        status=1
        if [[ $next_run -lt total_runs ]]; then
            run_one "$gpu" "$next_run" &
            pid_gpu[$!]="$gpu"
            ((next_run += 1))
        fi
    fi
done

if [[ $status -ne 0 ]]; then
    echo "One or more training runs failed. Check individual logs." >&2
    exit "$status"
fi

printf '=== ALL %d TRAINING RUNS COMPLETED SUCCESSFULLY: %s ===\n' "$total_runs" "$(date --iso-8601=seconds)"
