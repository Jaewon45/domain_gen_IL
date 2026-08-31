#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON="${REPO_ROOT}/dgil_env/Scripts/python.exe"
CHECKPOINTS="${REPO_ROOT}/results/cmnist_exp/ckpts"
OUTPUT_DIR="${REPO_ROOT}/results/cmnist_lambda_prediction_eval_v1"
LOG_DIR="${OUTPUT_DIR}/runner_logs"
LOG_FILE="${LOG_DIR}/lambda_prediction_eval.log"

mkdir -p "${LOG_DIR}"
START="$(date --iso-8601=seconds 2>/dev/null || date)"
echo "[${START}] START lambda prediction evaluation" | tee "${LOG_FILE}"
echo "[${START}] checkpoint_glob=${CHECKPOINTS}" | tee -a "${LOG_FILE}"

cd "${REPO_ROOT}/CMNIST"
"${PYTHON}" evaluate_lambda_predictions.py "${CHECKPOINTS}" \
  --output_dir "${OUTPUT_DIR}" \
  --algorithms iro,inftask \
  --max_checkpoints 2 \
  --device cpu \
  --eval_envs 0.0,0.1,0.5,0.9,1.0 \
  --lambda_grid 0.0:1.0:0.1 2>&1 | tee -a "${LOG_FILE}"
STATUS=${PIPESTATUS[0]}
END="$(date --iso-8601=seconds 2>/dev/null || date)"
echo "[${END}] END exit_code=${STATUS}" | tee -a "${LOG_FILE}"
exit "${STATUS}"
