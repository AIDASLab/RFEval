#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${1:-}"
if [[ -z "$MODEL_NAME" ]]; then
  echo "Usage: $0 <model_name> [extra inference.py args]" >&2
  echo "Example: $0 Qwen/Qwen3-32B --num_gpu 2" >&2
  echo "Example: $0 gpt-5.1 --use-proprietary-api" >&2
  exit 1
fi
shift || true
EXTRA_ARGS=("$@")

TASKS="${TASKS:-code_generation context_understanding legal_decision logical_reasoning mathematical_reasoning paper_review table_reasoning}"
IFS=' ' read -r -a TASK_LIST <<< "$TASKS"

NUM_GPU="${NUM_GPU:-2}"
DATASET_REPO="${DATASET_REPO:-snu-aidas/RFEval}"
SLEEP="${SLEEP:-0}"

for TASK in "${TASK_LIST[@]}"; do
  DATASET_ARGS=(--dataset_repo "${DATASET_REPO}" --dataset_config "${TASK}")

  python inference.py --model_name "${MODEL_NAME}" --task "${TASK}" --num_gpu "${NUM_GPU}" "${DATASET_ARGS[@]}" "${EXTRA_ARGS[@]}"
  python inference.py --model_name "${MODEL_NAME}" --task "${TASK}" --num_gpu "${NUM_GPU}" "${DATASET_ARGS[@]}" "${EXTRA_ARGS[@]}" --apply_intervention

  if [[ "${SLEEP}" != "0" ]]; then
    sleep "${SLEEP}"
  fi
done
