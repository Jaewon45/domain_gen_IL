#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON="${REPO_ROOT}/dgil_env/Scripts/python.exe"
LAMBDA_METRICS="${1:-${REPO_ROOT}/results/cmnist_lambda_prediction_eval_v2/prediction_lambda_metrics.csv}"
OUTPUT_DIR="${2:-${REPO_ROOT}/results/cmnist_lambda_pseudoregret_v2}"
LOG_DIR="${OUTPUT_DIR}/runner_logs"
LOG_FILE="${LOG_DIR}/regret_logging.log"

mkdir -p "${LOG_DIR}"
START="$(date --iso-8601=seconds 2>/dev/null || date)"
echo "[${START}] START regret logging" | tee "${LOG_FILE}"
echo "[${START}] lambda_metrics=${LAMBDA_METRICS}" | tee -a "${LOG_FILE}"
"${PYTHON}" "${REPO_ROOT}/CMNIST/analyze_regret.py" "${LAMBDA_METRICS}" \
  --output_dir "${OUTPUT_DIR}" 2>&1 | tee -a "${LOG_FILE}"
STATUS=${PIPESTATUS[0]}
END="$(date --iso-8601=seconds 2>/dev/null || date)"
echo "[${END}] END exit_code=${STATUS}" | tee -a "${LOG_FILE}"
exit "${STATUS}"
