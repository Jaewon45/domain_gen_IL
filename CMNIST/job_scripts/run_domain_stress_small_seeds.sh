#!/usr/bin/env bash
set -euo pipefail

# Run CMNIST reduced sweep for seeds 0, 1, and 2.
# Each seed writes to its own output folder to keep artifacts isolated.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CMNIST_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CMD_FILE="${SCRIPT_DIR}/domain_stress_small.txt"

if [[ ! -f "${CMD_FILE}" ]]; then
  echo "Command file not found: ${CMD_FILE}" >&2
  exit 1
fi

cd "${CMNIST_DIR}"

for seed in 0 1 2; do
  out_dir="../results/cmnist_exp_small_seed${seed}"
  echo "============================================================"
  echo "Running reduced sweep for seed=${seed}"
  echo "Output directory: ${out_dir}"
  echo "============================================================"

  while IFS= read -r cmd || [[ -n "${cmd}" ]]; do
    # Skip empty lines and comments.
    if [[ -z "${cmd// /}" || "${cmd}" =~ ^[[:space:]]*# ]]; then
      continue
    fi

    full_cmd="${cmd} --seed ${seed} --deterministic --n_workers 0 --output_dir ${out_dir}"
    echo "RUNNING: ${full_cmd}"
    eval "${full_cmd}"
  done < "${CMD_FILE}"
done

echo "All seed runs completed."
