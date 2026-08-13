#!/bin/bash
# Check that the probe produced a runtime selection, then start the 12K rung.

#SBATCH --job-name=mlf-t6-probe-gate
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00

set -euo pipefail
: "${T6_CONFIG:?set T6_CONFIG}"
# shellcheck disable=SC1090
source "${T6_CONFIG}"

[[ -s "${T6_RUNTIME_ENV}" ]] || {
  echo "[fail] probe did not write ${T6_RUNTIME_ENV}" >&2
  echo FAILED_PROBE > "${T6_STATE_DIR}/STATUS"
  exit 1
}
# shellcheck disable=SC1090
source "${T6_RUNTIME_ENV}"
[[ -d "${ENV_SOURCE}" ]] || {
  echo "[fail] selected environment does not exist: ${ENV_SOURCE}" >&2
  exit 1
}

train_job=$(sbatch --parsable \
  --job-name=mlf-t6-12k \
  --output="${T6_STATE_DIR}/logs/train-12k-%j.out" \
  --error="${T6_STATE_DIR}/logs/train-12k-%j.out" \
  --time="${T6_FULL_TIME}" \
  --export=ALL,T6_RUNTIME_ENV="${T6_RUNTIME_ENV}",RUNG=12k,SEED=0,EPOCHS=25,RUN_ID=R2_12k_s0_e25 \
  "${PROJECT_ROOT}/scripts/watgpu/train_rung.sbatch")
printf '12k\ttrain\t%s\n' "${train_job}" >> "${T6_STATE_DIR}/jobs.tsv"

gate_job=$(sbatch --parsable \
  --dependency="afterany:${train_job}" \
  --job-name=mlf-t6-gate-12k \
  --output="${T6_STATE_DIR}/logs/gate-12k-%j.out" \
  --error="${T6_STATE_DIR}/logs/gate-12k-%j.out" \
  --export=ALL,T6_CONFIG="${T6_CONFIG}",T6_STAGE=12k,RUN_ID=R2_12k_s0_e25,EXPECT_N=12000,EXPECT_EPOCHS=25,TRAIN_JOB_ID="${train_job}" \
  "${PROJECT_ROOT}/scripts/watgpu/t6_gate_and_advance.sbatch")
printf '12k\tgate\t%s\n' "${gate_job}" >> "${T6_STATE_DIR}/jobs.tsv"

echo RUNNING_12K > "${T6_STATE_DIR}/STATUS"
echo "[submit] 12k train=${train_job} gate=${gate_job}"
