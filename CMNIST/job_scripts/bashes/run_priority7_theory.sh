#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON="${REPO_ROOT}/dgil_env/Scripts/python.exe"
OUTPUT_DIR="${REPO_ROOT}/results/cmnist_priority7_theory_v1"
SMOKE=""

if [[ "${1:-}" == "--smoke" ]]; then
  SMOKE="--smoke"
fi

mkdir -p "${OUTPUT_DIR}/runner_logs"
LOG="${OUTPUT_DIR}/runner_logs/priority7_runner.log"
START="$(date --iso-8601=seconds 2>/dev/null || date)"
echo "[${START}] START python=${PYTHON} output=${OUTPUT_DIR} ${SMOKE}" | tee "${LOG}"
"${PYTHON}" "${REPO_ROOT}/CMNIST/priority7_theory.py" --output_dir "${OUTPUT_DIR}" ${SMOKE} 2>&1 | tee -a "${LOG}"
END="$(date --iso-8601=seconds 2>/dev/null || date)"
echo "[${END}] END exit_code=$?" | tee -a "${LOG}"
