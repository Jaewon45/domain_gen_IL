#!/usr/bin/env bash
# Evaluate ImageNet-100-C final checkpoints with one evaluator per GPU.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: run_evaluations.sh --results-root PATH [options]

By default, evaluates all 64 final checkpoints at lambda 0. The script requires
all training checkpoints before it starts, resumes only completed evaluation
JSONL files, and stops if it finds a partial evaluation directory.

Options:
  --results-root PATH   Seed training root (required)
  --python PATH         Python executable (default: python)
  --gpus IDS            CUDA device IDs, comma-separated (default: 0,1)
  --batch-size SIZE     Evaluation batch size (default: 64)
  --workers COUNT       DataLoader workers per evaluator (default: 0)
  --expected-runs COUNT Required final checkpoint count (default: 64; 0 permits partial)
  --offline             Require cached Hugging Face artifacts
  --dry-run             Print work discovered and exit
  -h, --help            Show this message
EOF
}

results_root=""
python_bin="python"
gpu_csv="0,1"
batch_size=64
workers=0
expected_runs=64
offline=0
dry_run=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --results-root) results_root="$2"; shift 2 ;;
        --python) python_bin="$2"; shift 2 ;;
        --gpus) gpu_csv="$2"; shift 2 ;;
        --batch-size) batch_size="$2"; shift 2 ;;
        --workers) workers="$2"; shift 2 ;;
        --expected-runs) expected_runs="$2"; shift 2 ;;
        --offline) offline=1; shift ;;
        --dry-run) dry_run=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n "$results_root" ]] || { echo "--results-root is required" >&2; exit 2; }
[[ "$batch_size" =~ ^[1-9][0-9]*$ ]] || { echo "--batch-size must be positive" >&2; exit 2; }
[[ "$workers" =~ ^[0-9]+$ ]] || { echo "--workers must be non-negative" >&2; exit 2; }
[[ "$expected_runs" =~ ^[0-9]+$ ]] || { echo "--expected-runs must be non-negative" >&2; exit 2; }

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
if [[ "$results_root" != /* ]]; then
    results_root="$repo_root/$results_root"
fi
[[ -d "$results_root" ]] || { echo "Results root not found: $results_root" >&2; exit 2; }
command -v "$python_bin" >/dev/null || { echo "Python executable not found: $python_bin" >&2; exit 2; }

IFS=',' read -r -a gpus <<< "$gpu_csv"
[[ ${#gpus[@]} -gt 0 && -n "${gpus[0]}" ]] || { echo "--gpus must list at least one device" >&2; exit 2; }
for gpu in "${gpus[@]}"; do
    [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "Invalid CUDA device ID: $gpu" >&2; exit 2; }
done
if [[ $offline -eq 1 ]]; then
    export HF_DATASETS_OFFLINE=1
    export HF_HUB_OFFLINE=1
fi

mapfile -t checkpoints < <(find "$results_root" -mindepth 3 -maxdepth 3 -type f -path '*/checkpoints/final.pt' | sort)
if [[ ${#checkpoints[@]} -eq 0 ]]; then
    echo "No final checkpoints found under: $results_root" >&2
    exit 1
fi
if [[ $expected_runs -gt 0 && ${#checkpoints[@]} -ne $expected_runs ]]; then
    echo "Expected $expected_runs final checkpoints, found ${#checkpoints[@]}. Training may still be active." >&2
    exit 1
fi

run_names=()
for checkpoint in "${checkpoints[@]}"; do
    run_directory="$(dirname "$(dirname "$checkpoint")")"
    run_names+=("$(basename "$run_directory")")
done

if [[ $dry_run -eq 1 ]]; then
    printf 'Planned full lambda-0 evaluations: %d across GPUs: %s\n' "${#run_names[@]}" "$gpu_csv"
    for run_name in "${run_names[@]}"; do
        printf '%s\n' "$run_name"
    done
    exit 0
fi

mkdir -p "$results_root/logs"
run_one() {
    local gpu="$1"
    local run_name="$2"
    local run_directory="$results_root/$run_name"
    local checkpoint="$run_directory/checkpoints/final.pt"
    local output_directory="$run_directory/evaluation_lambda0"
    local result="$output_directory/evaluation.jsonl"
    local log_file="$results_root/logs/evaluation_lambda0_${run_name}.log"
    if [[ -f "$result" ]]; then
        printf 'SKIP  %s: %s\n' "$(date --iso-8601=seconds)" "$run_name"
        return 0
    fi
    if [[ -e "$output_directory" ]]; then
        printf 'ERROR %s: incomplete evaluation directory: %s\n' "$(date --iso-8601=seconds)" "$output_directory" >&2
        return 1
    fi
    printf 'START %s: GPU %s %s\n' "$(date --iso-8601=seconds)" "$gpu" "$run_name"
    if CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -m IMAGENET100C.evaluate "$checkpoint" \
        --output_dir "$output_directory" --batch_size "$batch_size" --workers "$workers" >"$log_file" 2>&1; then
        :
    else
        printf 'ERROR %s: evaluation failed; see %s\n' "$(date --iso-8601=seconds)" "$log_file" >&2
        return 1
    fi
    [[ -f "$result" ]] || { echo "Evaluation result missing: $result" >&2; return 1; }
    printf 'DONE  %s: %s\n' "$(date --iso-8601=seconds)" "$run_name"
}

run_worker() {
    local worker_index="$1"
    local gpu="$2"
    local run_index
    for ((run_index=worker_index; run_index<${#run_names[@]}; run_index+=${#gpus[@]})); do
        run_one "$gpu" "${run_names[run_index]}" || return 1
    done
}

worker_pids=()
for worker_index in "${!gpus[@]}"; do
    run_worker "$worker_index" "${gpus[worker_index]}" &
    worker_pids+=("$!")
done

status=0
for worker_pid in "${worker_pids[@]}"; do
    wait "$worker_pid" || status=1
done
if [[ $status -ne 0 ]]; then
    echo "An evaluator failed; completed evaluations can be resumed safely." >&2
    exit "$status"
fi
printf 'ALL FULL LAMBDA-0 EVALUATIONS COMPLETE: %s\n' "$(date --iso-8601=seconds)"
