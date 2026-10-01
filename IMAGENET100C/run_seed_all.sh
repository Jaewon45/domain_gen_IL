#!/usr/bin/env bash
# Run the registered ImageNet-100-C training matrix across independent GPUs.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: run_seed_all.sh [options]

Options:
  --seed SEED                 Random seed (default: 0)
  --backbone-mode MODE        frozen_feature_pilot or finetune_last_stage
                              (default: finetune_last_stage)
  --results-root PATH         Output root (default: results/imagenet100c_seedSEED)
  --python PATH               Python executable (default: python)
  --gpus IDS                  Comma-separated CUDA device IDs (default: 0,1)
  --workers-per-gpu N         Concurrent processes per GPU (default: 1)
  --e3b-only                  Run only the four E3b support conditions
  --offline                   Require Hugging Face data and weights to be cached
    --dry-run                   Print the GPU assignment for every run and exit
  -h, --help                  Show this message

Example:
  nohup bash IMAGENET100C/run_seed_all.sh --seed 1 --gpus 0,1 \
    > logs/imagenet100c_seed1.log 2>&1 &
EOF
}

seed=0
backbone_mode="finetune_last_stage"
results_root=""
python_bin="python"
gpu_csv="0,1"
workers_per_gpu=1
e3b_only=0
offline=0
dry_run=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed) seed="$2"; shift 2 ;;
        --backbone-mode) backbone_mode="$2"; shift 2 ;;
        --results-root) results_root="$2"; shift 2 ;;
        --python) python_bin="$2"; shift 2 ;;
        --gpus) gpu_csv="$2"; shift 2 ;;
        --workers-per-gpu) workers_per_gpu="$2"; shift 2 ;;
        --e3b-only) e3b_only=1; shift ;;
        --offline) offline=1; shift ;;
        --dry-run) dry_run=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "$backbone_mode" in
    frozen_feature_pilot|finetune_last_stage) ;;
    *) echo "Invalid --backbone-mode: $backbone_mode" >&2; exit 2 ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
if [[ -z "$results_root" ]]; then
    results_root="results/imagenet100c_seed${seed}"
fi
if [[ "$results_root" != /* ]]; then
    results_root="$repo_root/$results_root"
fi

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
mkdir -p "$results_root/logs"
if [[ $offline -eq 1 ]]; then
    export HF_DATASETS_OFFLINE=1
    export HF_HUB_OFFLINE=1
fi

runs=()
add_run() {
    local tag="$1"
    local algorithm="$2"
    shift 2
    runs+=("$tag|$algorithm|$*")
}

algorithms=(erm irm vrex eqrm groupdro inftask iro)
if [[ $e3b_only -eq 0 ]]; then
    for algorithm in "${algorithms[@]}"; do
        add_run "E0" "$algorithm" "--experiment E0"
    done
    for source_count in 2 4 8 12; do
        for algorithm in "${algorithms[@]}"; do
            add_run "E1_${source_count}types" "$algorithm" "--experiment E1 --source_count $source_count"
        done
    done
    for samples_per_type in 1000 5000 10000; do
        for algorithm in "${algorithms[@]}"; do
            add_run "E2_${samples_per_type}pertype" "$algorithm" "--experiment E2 --samples_per_type $samples_per_type"
        done
    done
    for condition in balanced mild_imbalance strong_imbalance; do
        for algorithm in "${algorithms[@]}"; do
            add_run "E3_${condition}" "$algorithm" "--experiment E3 --condition $condition"
        done
    done
fi
for condition in balanced long_tail near_missing missing; do
    for algorithm in "${algorithms[@]}"; do
        add_run "E3b_${condition}" "$algorithm" "--experiment E3b --condition $condition"
    done
done
if [[ $e3b_only -eq 0 ]]; then
    for algorithm in "${algorithms[@]}"; do
        add_run "severity_support" "$algorithm" "--experiment severity_support"
    done
fi

if [[ $dry_run -eq 1 ]]; then
        printf 'Planned runs: %d across GPUs: %s (%s workers/GPU)\n' "${#runs[@]}" "$gpu_csv" "$workers_per_gpu"
    for run_index in "${!runs[@]}"; do
        IFS='|' read -r tag algorithm experiment_args <<< "${runs[run_index]}"
        printf 'GPU %s: %s_%s %s\n' "${gpus[run_index % ${#gpus[@]}]}" "$tag" "$algorithm" "$experiment_args"
    done
    exit 0
fi

run_one() {
    local gpu="$1"
    local run_index="$2"
    local tag algorithm experiment_args output_directory checkpoint log_file
    IFS='|' read -r tag algorithm experiment_args <<< "${runs[run_index]}"
    output_directory="$results_root/${tag}_${algorithm}"
    checkpoint="$output_directory/checkpoints/final.pt"
    log_file="$results_root/logs/${tag}_${algorithm}.log"
    if [[ -f "$checkpoint" ]]; then
        printf 'SKIP  %s: %s\n' "$(date --iso-8601=seconds)" "$checkpoint"
        return 0
    fi
    if [[ -e "$output_directory" ]]; then
        printf 'ERROR %s: incomplete output directory: %s\n' "$(date --iso-8601=seconds)" "$output_directory" >&2
        return 1
    fi
    printf 'START %s: GPU %s %s\n' "$(date --iso-8601=seconds)" "$gpu" "$output_directory"
    read -r -a experiment_arg_array <<< "$experiment_args"
    if ! CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -m IMAGENET100C.train \
        --seed "$seed" --algorithm "$algorithm" --backbone_mode "$backbone_mode" \
        --checkpoint_selection final --output_dir "$output_directory" \
        "${experiment_arg_array[@]}" >"$log_file" 2>&1; then
        printf 'ERROR %s: training failed; see %s\n' "$(date --iso-8601=seconds)" "$log_file" >&2
        return 1
    fi
    if [[ ! -f "$checkpoint" ]]; then
        printf 'ERROR %s: missing final checkpoint: %s\n' "$(date --iso-8601=seconds)" "$checkpoint" >&2
        return 1
    fi
    printf 'DONE  %s: %s\n' "$(date --iso-8601=seconds)" "$output_directory"
}

declare -A pid_gpu
next_run=0
for gpu in "${gpus[@]}"; do
    for ((worker = 0; worker < workers_per_gpu && next_run < ${#runs[@]}; worker++)); do
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
        unset 'pid_gpu[$completed_pid]'
        if [[ $next_run -lt ${#runs[@]} ]]; then
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
    for worker_pid in "${!pid_gpu[@]}"; do
        wait "$worker_pid" || true
    done
    echo "A GPU worker failed; completed runs can be resumed safely." >&2
    exit "$status"
fi
printf 'ALL TRAINING RUNS COMPLETE: %s\n' "$(date --iso-8601=seconds)"
