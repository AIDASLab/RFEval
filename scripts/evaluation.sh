#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${1:-}"
if [[ -z "$MODEL_NAME" ]]; then
  echo "Usage: $0 <model_name> [extra evaluation.py args]" >&2
  echo "Example: $0 Qwen/Qwen3-32B" >&2
  echo "Example: $0 gpt-5.1 --evaluator_name o3 --batch_post" >&2
  exit 1
fi
shift || true
EXTRA_ARGS=("$@")

TASKS="${TASKS:-code_generation context_understanding legal_decision logical_reasoning mathematical_reasoning paper_review table_reasoning}"
IFS=' ' read -r -a TASK_LIST <<< "$TASKS"

EVALUATOR="${EVALUATOR:-o3}"
MODE="${MODE:-sync}"
SLEEP="${SLEEP:-1}"

BATCH_FLAG=""
if [[ "$MODE" == "post" || "$MODE" == "get" ]]; then
  if [[ "$EVALUATOR" == *"claude"* || "$EVALUATOR" == *"gemini"* ]]; then
    echo "[ERROR] Batch mode is only supported for OpenAI evaluators." >&2
    exit 1
  fi
  if [[ "$MODE" == "post" ]]; then
    BATCH_FLAG="--batch_post"
  else
    BATCH_FLAG="--batch_get"
  fi
fi

for TASK in "${TASK_LIST[@]}"; do
  python evaluation.py --model_name "${MODEL_NAME}" --evaluator_name "${EVALUATOR}" --task "${TASK}" $BATCH_FLAG "${EXTRA_ARGS[@]}"
  python evaluation.py --model_name "${MODEL_NAME}" --evaluator_name "${EVALUATOR}" --task "${TASK}" --apply_intervention $BATCH_FLAG "${EXTRA_ARGS[@]}"
  if [[ "${SLEEP}" != "0" ]]; then
    sleep "${SLEEP}"
  fi
done
