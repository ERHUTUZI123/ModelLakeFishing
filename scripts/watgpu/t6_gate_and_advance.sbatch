#!/bin/bash
# Validate one T6 run.  A passed gate submits the next rung, so no controller
# allocation sits idle while GPU jobs wait in the queue.

#SBATCH --job-name=mlf-t6-gate
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00

set -euo pipefail

: "${T6_CONFIG:?set T6_CONFIG}"
# shellcheck disable=SC1090
source "${T6_CONFIG}"

: "${T6_STAGE:?set T6_STAGE}"
: "${RUN_ID:?set RUN_ID}"
: "${EXPECT_N:?set EXPECT_N}"
: "${EXPECT_EPOCHS:?set EXPECT_EPOCHS}"
: "${TRAIN_JOB_ID:?set TRAIN_JOB_ID}"

RUN_DIR="${OUTPUT_ROOT}/runs/${RUN_ID}"
REPORT="${T6_STATE_DIR}/reports/gate_${T6_STAGE}.json"
mkdir -p "${T6_STATE_DIR}/reports" "${T6_STATE_DIR}/artifacts" "${T6_STATE_DIR}/logs"

set +e
python3 "${PROJECT_ROOT}/scale1m/validate_t6_run.py" \
  --run-dir "${RUN_DIR}" \
  --stage "${T6_STAGE}" \
  --expect-n "${EXPECT_N}" \
  --expect-epochs "${EXPECT_EPOCHS}" \
  --train-job-id "${TRAIN_JOB_ID}" \
  --full-time-seconds "${T6_FULL_TIME_SECONDS}" \
  --gold-gate-override "${T6_ALLOW_NO_GOLD_GATE:-0}" \
  --report "${REPORT}"
gate_rc=$?
set -e

# Preserve the small evidence bundle; checkpoints intentionally remain remote.
if [[ -f "${RUN_DIR}/MANIFEST.json" ]]; then
  tar -czf "${T6_STATE_DIR}/artifacts/t6_${RUN_ID}.tgz" \
    -C "${RUN_DIR}" MANIFEST.json metrics metadata stdout/train.log
fi
sacct -j "${TRAIN_JOB_ID}" \
  --format=JobID,Submit,Start,End,Elapsed,MaxRSS,AllocTRES,ExitCode \
  > "${T6_STATE_DIR}/reports/sacct_${T6_STAGE}.txt" || true

if (( gate_rc != 0 )); then
  printf 'FAILED_%s\n' "${T6_STAGE}" > "${T6_STATE_DIR}/STATUS"
  echo "[stop] ${T6_STAGE} gate failed; no downstream work submitted" >&2
  exit "${gate_rc}"
fi

record_job() {
  printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> "${T6_STATE_DIR}/jobs.tsv"
}

submit_train_and_gate() {
  local stage=$1 rung=$2 seed=$3 epochs=$4 expect_n=$5 run_id=$6 time_limit=$7
  local train_job gate_job
  train_job=$(sbatch --parsable \
    --job-name="mlf-t6-${stage}" \
    --output="${T6_STATE_DIR}/logs/train-${stage}-%j.out" \
    --error="${T6_STATE_DIR}/logs/train-${stage}-%j.out" \
    --time="${time_limit}" \
    --export=ALL,T6_RUNTIME_ENV="${T6_RUNTIME_ENV}",RUNG="${rung}",SEED="${seed}",EPOCHS="${epochs}",RUN_ID="${run_id}" \
    "${PROJECT_ROOT}/scripts/watgpu/train_rung.sbatch")
  record_job "${stage}" train "${train_job}"
  gate_job=$(sbatch --parsable \
    --dependency="afterany:${train_job}" \
    --job-name="mlf-t6-gate-${stage}" \
    --output="${T6_STATE_DIR}/logs/gate-${stage}-%j.out" \
    --error="${T6_STATE_DIR}/logs/gate-${stage}-%j.out" \
    --export=ALL,T6_CONFIG="${T6_CONFIG}",T6_STAGE="${stage}",RUN_ID="${run_id}",EXPECT_N="${expect_n}",EXPECT_EPOCHS="${epochs}",TRAIN_JOB_ID="${train_job}" \
    "${PROJECT_ROOT}/scripts/watgpu/t6_gate_and_advance.sbatch")
  record_job "${stage}" gate "${gate_job}"
  printf '%s' "${gate_job}"
}

case "${T6_STAGE}" in
  12k)
    submit_train_and_gate 30k 30k 0 25 30183 R2_30k_s0_e25 "${T6_FULL_TIME}"
    echo RUNNING_30K > "${T6_STATE_DIR}/STATUS"
    ;;
  30k)
    submit_train_and_gate 100k_smoke 100k 0 2 100000 R2_100k_s0_e2 "${T6_SMOKE_TIME}"
    echo RUNNING_100K_SMOKE > "${T6_STATE_DIR}/STATUS"
    ;;
  100k_smoke)
    if [[ "${T6_ALLOW_NO_GOLD_GATE}" != 1 ]]; then
      echo "PAUSED_NO_GOLD_GATE" > "${T6_STATE_DIR}/STATUS"
      echo "[pause] smoke passed, but T6_ALLOW_NO_GOLD_GATE is not 1"
      exit 0
    fi
    echo "[warning] proceeding without gold@10 validation; recorded by explicit opt-in"
    gate_ids=()
    for seed in 0 1 2; do
      gate_ids+=("$(submit_train_and_gate "100k_s${seed}" 100k "${seed}" 25 100000 "R2_100k_s${seed}_e25" "${T6_FULL_TIME}")")
    done
    dependency=$(IFS=:; echo "${gate_ids[*]}")
    summary_job=$(sbatch --parsable \
      --dependency="afterany:${dependency}" \
      --job-name=mlf-t6-summary \
      --output="${T6_STATE_DIR}/logs/summary-%j.out" \
      --error="${T6_STATE_DIR}/logs/summary-%j.out" \
      --export=ALL,T6_CONFIG="${T6_CONFIG}" \
      "${PROJECT_ROOT}/scripts/watgpu/t6_summary.sbatch")
    record_job summary summary "${summary_job}"
    echo RUNNING_100K_FULL > "${T6_STATE_DIR}/STATUS"
    ;;
  100k_s0|100k_s1|100k_s2)
    echo "[gate] ${T6_STAGE} complete; summary dependency will collect all seeds"
    ;;
  *)
    echo "[fail] unknown T6_STAGE=${T6_STAGE}" >&2
    exit 2
    ;;
esac
