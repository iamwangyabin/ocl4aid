#!/usr/bin/env bash
set -Eeuo pipefail

: "${CAID_DATA_DIR:?Set CAID_DATA_DIR to the CAIDBenchmark root}"
: "${CAID_STREAM_NAME:?Set CAID_STREAM_NAME, e.g. mainblurry or hard}"
: "${CAID_STAGE_BLURRY_N:?Set CAID_STAGE_BLURRY_N}"
: "${CAID_STAGE_BLURRY_M:?Set CAID_STAGE_BLURRY_M}"
: "${CAID_SWANLAB_GROUP:?Set CAID_SWANLAB_GROUP}"
: "${CAID_PLAN_ID:?Set CAID_PLAN_ID}"
: "${CAID_METHODS:?Set CAID_METHODS as a space-separated method list}"

CAID_CONFIG="${CAID_CONFIG:-configs/framework/caidbench.yaml}"
CAID_SEEDS="${CAID_SEEDS:-1 2 3}"
CAID_BASE_STAGE_EPOCHS="${CAID_BASE_STAGE_EPOCHS:-5}"
CAID_N_WORKER="${CAID_N_WORKER:-8}"
CAID_SWANLAB_PROJECT="${CAID_SWANLAB_PROJECT:-ocl4aid}"
CAID_EXTRA_TAGS="${CAID_EXTRA_TAGS:-}"

commit="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
seed_label="$(printf '%s' "${CAID_SEEDS}" | tr ' ' '-')"
log_root="run_logs/${CAID_PLAN_ID}"
mkdir -p "${log_root}"

{
  echo "PLAN_START $(date -Is)"
  echo "commit=${commit}"
  echo "config=${CAID_CONFIG}"
  echo "data_dir=${CAID_DATA_DIR}"
  echo "stream=${CAID_STREAM_NAME}"
  echo "stage_blurry_n=${CAID_STAGE_BLURRY_N}"
  echo "stage_blurry_m=${CAID_STAGE_BLURRY_M}"
  echo "base_stage_epochs=${CAID_BASE_STAGE_EPOCHS}"
  echo "seeds=${CAID_SEEDS}"
  echo "methods=${CAID_METHODS}"
  echo "swanlab_group=${CAID_SWANLAB_GROUP}"
  echo
} | tee -a "${log_root}/plan.log"

for method in ${CAID_METHODS}; do
  note="caid_${CAID_STREAM_NAME}_${method}_base${CAID_BASE_STAGE_EPOCHS}_s${seed_label}_${commit}"
  experiment_name="${method}_${CAID_STREAM_NAME}_base${CAID_BASE_STAGE_EPOCHS}_s${seed_label}_${commit}"

  {
    echo "METHOD_START ${method} $(date -Is)"
    echo "note=${note}"
    echo "experiment_name=${experiment_name}"
  } | tee -a "${log_root}/plan.log"

  python -u main.py \
    --config "${CAID_CONFIG}" \
    --caidbench_data_dir "${CAID_DATA_DIR}" \
    --method "${method}" \
    --seeds ${CAID_SEEDS} \
    --base_stage_epochs "${CAID_BASE_STAGE_EPOCHS}" \
    --stage_blurry_n "${CAID_STAGE_BLURRY_N}" \
    --stage_blurry_m "${CAID_STAGE_BLURRY_M}" \
    --n_worker "${CAID_N_WORKER}" \
    --note "${note}" \
    --swanlab \
    --swanlab_mode offline \
    --swanlab_project "${CAID_SWANLAB_PROJECT}" \
    --swanlab_group "${CAID_SWANLAB_GROUP}" \
    --swanlab_experiment_name "${experiment_name}" \
    --swanlab_tags \
      caidbench \
      "stream_${CAID_STREAM_NAME}" \
      "base_stage${CAID_BASE_STAGE_EPOCHS}" \
      "stage_blurry_n${CAID_STAGE_BLURRY_N}" \
      "stage_blurry_m${CAID_STAGE_BLURRY_M}" \
      "seeds_${seed_label}" \
      "commit_${commit}" \
      "plan_${CAID_PLAN_ID}" \
      ${CAID_EXTRA_TAGS}

  echo "METHOD_END ${method} $(date -Is)" | tee -a "${log_root}/plan.log"
done

echo "PLAN_END $(date -Is)" | tee -a "${log_root}/plan.log"
