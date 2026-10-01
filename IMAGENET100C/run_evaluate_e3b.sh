#!/usr/bin/env bash
# Evaluate one canonical ImageNet-100-C E3b seed across the four registered anchors.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: run_evaluate_e3b.sh --seed SEED [options]

Options:
  --seed SEED                Required E3b seed.
  --results-root PATH        Training root (default: results/imagenet100c).
  --python PATH              Python executable (default: python).
  --gpus IDS                 CUDA device IDs, comma-separated (default: 0,1).
  --batch-size SIZE          Evaluation batch size (default: 64).
  --workers COUNT            DataLoader workers (default: 0).
  --dry-run                  Print discovered runs without evaluating.
EOF
}

seed=""
results_root="results/imagenet100c"
python_bin="python"
gpu_csv="0,1"
batch_size=64
workers=0
dry_run=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed) seed="$2"; shift 2 ;;
        --results-root) results_root="$2"; shift 2 ;;
        --python) python_bin="$2"; shift 2 ;;
        --gpus) gpu_csv="$2"; shift 2 ;;
        --batch-size) batch_size="$2"; shift 2 ;;
        --workers) workers="$2"; shift 2 ;;
        --dry-run) dry_run=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ "$seed" =~ ^[0-9]+$ ]] || { echo "--seed must be a non-negative integer" >&2; exit 2; }
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
[[ "$results_root" = /* ]] || results_root="$repo_root/$results_root"
seed_root="$results_root/seed${seed}"
[[ -d "$seed_root" ]] || { echo "Canonical seed root not found: $seed_root" >&2; exit 1; }
command -v "$python_bin" >/dev/null || { echo "Python executable not found: $python_bin" >&2; exit 2; }
IFS=',' read -r -a gpus <<< "$gpu_csv"
[[ ${#gpus[@]} -gt 0 && -n "${gpus[0]}" ]] || { echo "--gpus must list at least one device" >&2; exit 2; }
for gpu in "${gpus[@]}"; do [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "Invalid CUDA device ID: $gpu" >&2; exit 2; }; done

mapfile -t checkpoints < <(find "$seed_root" -mindepth 3 -maxdepth 3 -type f -path '*/checkpoints/final.pt' | sort)
[[ ${#checkpoints[@]} -eq 28 ]] || { echo "Expected 28 canonical E3b checkpoints for seed ${seed}, found ${#checkpoints[@]}" >&2; exit 1; }
if [[ $dry_run -eq 1 ]]; then printf 'Planned focused evaluations: %d\n' "${#checkpoints[@]}"; printf '%s\n' "${checkpoints[@]}"; exit 0; fi

mkdir -p "$seed_root/logs"
run_one() {
    local gpu="$1" checkpoint="$2" run_directory output log_file
    run_directory="$(dirname "$(dirname "$checkpoint")")"
    output="$run_directory/evaluation_anchor1000"
    log_file="$seed_root/logs/$(basename "$run_directory")_evaluation.log"
    [[ -f "$output/evaluation.jsonl" ]] && return 0
    [[ ! -e "$output" ]] || { echo "Incomplete evaluation directory: $output" >&2; return 1; }
    CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -m IMAGENET100C.evaluate "$checkpoint" \
        --corruption_types gaussian_noise,defocus_blur,snow,contrast --max_eval_images 1000 \
        --batch_size "$batch_size" --workers "$workers" --output_dir "$output" >"$log_file" 2>&1
    [[ -f "$output/evaluation.jsonl" ]]
}

run_worker() {
    local worker_index="$1" gpu="$2" index
    for ((index = worker_index; index < ${#checkpoints[@]}; index += ${#gpus[@]})); do
        run_one "$gpu" "${checkpoints[index]}" || return 1
    done
}

pids=()
for index in "${!gpus[@]}"; do
    run_worker "$index" "${gpus[index]}" &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
