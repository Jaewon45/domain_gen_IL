#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="/home/ra95tig/anaconda3/envs/domgen/bin/python"
root="/home/ra95tig/results_final_imagenet100c_seed2_irm_vrex_4workers"
cd "$repo_root"
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
mkdir -p "$root/logs"
run_one() {
  local gpu="$1" algorithm="$2" condition="$3"
  local out="$root/seed2/E3b_${condition}_${algorithm}"
  local log="$root/logs/E3b_${condition}_${algorithm}_seed2.log"
  [[ -f "$out/checkpoints/final.pt" ]] && return 0
  [[ ! -e "$out" ]] || { echo "Incomplete output exists: $out" >&2; return 1; }
  CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -m IMAGENET100C.train \
      --seed 2 --algorithm "$algorithm" --backbone_mode finetune_last_stage \
      --experiment E3b --condition "$condition" --steps 1000 --batch_size 64 \
      --num_lambda_samples 4 --eqrm_alpha 0.9 --penalty_weight 1000 \
      --erm_pretrain_iters 400 --lr_cos_sched --workers 1 --pin_memory \
      --persistent_workers --prefetch_factor 2 --checkpoint_selection final \
      --output_dir "$out" >"$log" 2>&1
  [[ -f "$out/checkpoints/final.pt" ]]
}
worker_lane() {
  local gpu="$1" algorithm="$2" index="$3"
  local conditions=(balanced long_tail scarce_tail missing)
  run_one "$gpu" "$algorithm" "${conditions[index]}"
}
for index in 0 1 2 3; do worker_lane 0 irm "$index" & done
for index in 0 1 2 3; do worker_lane 1 vrex "$index" & done
status=0
for worker_pid in $(jobs -p); do wait "$worker_pid" || status=1; done
exit "$status"
