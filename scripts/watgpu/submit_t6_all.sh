#!/bin/bash
# One-shot, unattended T6 workflow launcher.
#
# The scientific anchor cannot be fully accepted until T7/T8 implement
# gold@10.  To train all 100K seeds anyway, the caller must make that explicit:
#
#   T6_ALLOW_NO_GOLD_GATE=1 bash scripts/watgpu/submit_t6_all.sh
#
# Without that opt-in the workflow safely stops after the 100K smoke test.

set -euo pipefail

if [[ -f "${HOME}/.mlf_env" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/.mlf_env"
fi

: "${PROJECT_ROOT:?set PROJECT_ROOT (normally in ~/.mlf_env)}"
: "${DATA_ROOT:?set DATA_ROOT (normally in ~/.mlf_env)}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT (normally in ~/.mlf_env)}"
: "${ENV_SOURCE:?set ENV_SOURCE (normally in ~/.mlf_env)}"

T6_ALLOW_NO_GOLD_GATE=${T6_ALLOW_NO_GOLD_GATE:-0}
T6_FULL_TIME=${T6_FULL_TIME:-04:00:00}
T6_SMOKE_TIME=${T6_SMOKE_TIME:-04:00:00}
T6_FULL_TIME_SECONDS=${T6_FULL_TIME_SECONDS:-14400}
T6_WORKFLOW_ID=${T6_WORKFLOW_ID:-T6_$(date -u +%Y%m%dT%H%M%SZ)}
T6_STATE_DIR="${OUTPUT_ROOT}/t6_workflows/${T6_WORKFLOW_ID}"
T6_CONFIG="${T6_STATE_DIR}/workflow.env"
T6_RUNTIME_ENV="${T6_STATE_DIR}/runtime.env"

[[ "${T6_ALLOW_NO_GOLD_GATE}" == 0 || "${T6_ALLOW_NO_GOLD_GATE}" == 1 ]] || {
  echo "[fail] T6_ALLOW_NO_GOLD_GATE must be 0 or 1" >&2
  exit 2
}

[[ ! -e "${T6_STATE_DIR}" ]] || {
  echo "[fail] workflow state already exists: ${T6_STATE_DIR}" >&2
  echo "       choose a new T6_WORKFLOW_ID or inspect the existing workflow" >&2
  exit 2
}

for path in \
  "${PROJECT_ROOT}/scripts/watgpu/train_rung.sbatch" \
  "${PROJECT_ROOT}/scripts/watgpu/t6_probe_env.sbatch" \
  "${PROJECT_ROOT}/scripts/watgpu/t6_probe_advance.sbatch" \
  "${PROJECT_ROOT}/scripts/watgpu/t6_gate_and_advance.sbatch" \
  "${PROJECT_ROOT}/scripts/watgpu/t6_summary.sbatch" \
  "${PROJECT_ROOT}/scripts/watgpu/stage_env.sh" \
  "${PROJECT_ROOT}/stage1BuildTransferGraph/hgraph_ml_v2_sub.pt" \
  "${PROJECT_ROOT}/stage1BuildTransferGraph/hgraph_ml_v2.pt" \
  "${DATA_ROOT}/data1m/graphs/hgraph_100k.pt" \
  "${DATA_ROOT}/data1m/feats/100k/family_vocab.csv" \
  "${ENV_SOURCE}"; do
  [[ -e "${path}" ]] || { echo "[fail] required input missing: ${path}" >&2; exit 2; }
done

for script in \
  "${PROJECT_ROOT}/scripts/watgpu/train_rung.sbatch" \
  "${PROJECT_ROOT}/scripts/watgpu/t6_probe_env.sbatch" \
  "${PROJECT_ROOT}/scripts/watgpu/t6_probe_advance.sbatch" \
  "${PROJECT_ROOT}/scripts/watgpu/t6_gate_and_advance.sbatch" \
  "${PROJECT_ROOT}/scripts/watgpu/t6_summary.sbatch" \
  "${PROJECT_ROOT}/scripts/watgpu/stage_env.sh"; do
  bash -n "${script}"
done
python3 -m py_compile "${PROJECT_ROOT}/scale1m/validate_t6_run.py"

mkdir -p "${DATA_ROOT}/data1m/graphs" \
  "${T6_STATE_DIR}/logs" "${T6_STATE_DIR}/reports" "${T6_STATE_DIR}/artifacts"
ln -sfn "${PROJECT_ROOT}/stage1BuildTransferGraph/hgraph_ml_v2_sub.pt" \
  "${DATA_ROOT}/data1m/graphs/hgraph_12k.pt"
ln -sfn "${PROJECT_ROOT}/stage1BuildTransferGraph/hgraph_ml_v2.pt" \
  "${DATA_ROOT}/data1m/graphs/hgraph_30k.pt"

{
  printf 'export PROJECT_ROOT=%q\n' "${PROJECT_ROOT}"
  printf 'export DATA_ROOT=%q\n' "${DATA_ROOT}"
  printf 'export OUTPUT_ROOT=%q\n' "${OUTPUT_ROOT}"
  printf 'export ENV_SOURCE=%q\n' "${ENV_SOURCE}"
  printf 'export T6_STATE_DIR=%q\n' "${T6_STATE_DIR}"
  printf 'export T6_CONFIG=%q\n' "${T6_CONFIG}"
  printf 'export T6_RUNTIME_ENV=%q\n' "${T6_RUNTIME_ENV}"
  printf 'export T6_ALLOW_NO_GOLD_GATE=%q\n' "${T6_ALLOW_NO_GOLD_GATE}"
  printf 'export T6_FULL_TIME=%q\n' "${T6_FULL_TIME}"
  printf 'export T6_SMOKE_TIME=%q\n' "${T6_SMOKE_TIME}"
  printf 'export T6_FULL_TIME_SECONDS=%q\n' "${T6_FULL_TIME_SECONDS}"
  printf 'export T6_WORKFLOW_ID=%q\n' "${T6_WORKFLOW_ID}"
} > "${T6_CONFIG}"

{
  printf 'workflow\tmeta\t%s\n' "${T6_WORKFLOW_ID}"
  printf 'gold_gate_override\tmeta\t%s\n' "${T6_ALLOW_NO_GOLD_GATE}"
} > "${T6_STATE_DIR}/jobs.tsv"
echo SUBMITTED_PROBE > "${T6_STATE_DIR}/STATUS"

probe_job=$(sbatch --parsable \
  --output="${T6_STATE_DIR}/logs/probe-%j.out" \
  --error="${T6_STATE_DIR}/logs/probe-%j.out" \
  --export=ALL,T6_STATE_DIR="${T6_STATE_DIR}" \
  "${PROJECT_ROOT}/scripts/watgpu/t6_probe_env.sbatch")
printf 'probe\tprobe\t%s\n' "${probe_job}" >> "${T6_STATE_DIR}/jobs.tsv"

probe_gate_job=$(sbatch --parsable \
  --dependency="afterany:${probe_job}" \
  --output="${T6_STATE_DIR}/logs/probe-gate-%j.out" \
  --error="${T6_STATE_DIR}/logs/probe-gate-%j.out" \
  --export=ALL,T6_CONFIG="${T6_CONFIG}" \
  "${PROJECT_ROOT}/scripts/watgpu/t6_probe_advance.sbatch")
printf 'probe\tgate\t%s\n' "${probe_gate_job}" >> "${T6_STATE_DIR}/jobs.tsv"

cat <<EOF
[submitted] ${T6_WORKFLOW_ID}
  state:      ${T6_STATE_DIR}
  probe:      ${probe_job}
  probe gate: ${probe_gate_job}
  full seeds: $([[ "${T6_ALLOW_NO_GOLD_GATE}" == 1 ]] && echo enabled-without-gold-validation || echo paused-after-smoke)

You may disconnect now.  Later inspect with:
  cat '${T6_STATE_DIR}/STATUS'
  cat '${T6_STATE_DIR}/jobs.tsv'
  squeue -u '${USER}'
EOF
