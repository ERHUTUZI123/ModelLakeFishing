#!/bin/bash
# T6 (docs/1M/100kplan.md §9): train the champion config on one rung.
#
# Submit from <REMOTE_PROJECT_ROOT>:
#     mkdir -p logs
#     RUNG=12k SEED=0 EPOCHS=25 sbatch --export=ALL,RUNG,SEED,EPOCHS \
#         scripts/watgpu/train_rung.sbatch
#
# RUN_ID is derived from RUNG/SEED/EPOCHS unless you set it. Keep the same
# RUN_ID when a job is requeued: the driver finds <run>/ckpt/last.pt by itself
# and continues from there, and mixing RUN_IDs across a requeue splits one
# logical run's metrics across two directories.
#
# Inputs beyond the repo:
#   $DATA_ROOT/data1m/graphs/hgraph_<RUNG>.pt        (T5; 12k/30k come from CORE)
#   $DATA_ROOT/data1m/feats/<RUNG>/family_vocab.csv  (T4; checkpoint binding)

#SBATCH --job-name=mlf-t6-train
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
# Storage preflight: H200 NVL 140 GB plus an executable /tmp with >500 GB free.
#SBATCH --nodelist=watgpu808
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
# 100kplan.md 4.5 set 04:00:00 as the pre-measurement placeholder and said to
# correct it once a real epoch had been timed. It has been: the slowest T6 run
# was 2m44s wall including env staging (100k, 25 epochs, 3 seeds all under 3m).
#
# Oversizing this is not free. Slurm reserves the *requested* wall time, so at
# 4h each, six 2-minute jobs behind one busy GPU get scheduled 4h apart -- a
# 20-hour queue. At 00:30:00 the same six fit in a single evening.
#
# R3/R4 must raise this: pass --time on the sbatch command line, which
# overrides the directive below.
#SBATCH --time=00:30:00

set -euo pipefail

# A T6 workflow probe can select a persisted environment at runtime.  This is
# sourced by the compute job (rather than by the submit shell) because the
# probe and this job are separated by a Slurm dependency.
if [[ -n "${T6_RUNTIME_ENV:-}" ]]; then
  [[ -f "${T6_RUNTIME_ENV}" ]] || {
    echo "[fail] no runtime environment selection at ${T6_RUNTIME_ENV}"
    exit 2
  }
  # shellcheck disable=SC1090
  source "${T6_RUNTIME_ENV}"
fi

: "${PROJECT_ROOT:?set PROJECT_ROOT}"
: "${DATA_ROOT:?set DATA_ROOT}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT}"
: "${ENV_SOURCE:?set ENV_SOURCE to the stored Python environment}"
: "${RUNG:=100k}"
: "${SEED:=0}"
: "${EPOCHS:=25}"
: "${RUN_ID:=R2_${RUNG}_s${SEED}_e${EPOCHS}}"

RUN_DIR="${OUTPUT_ROOT}/runs/${RUN_ID}"
mkdir -p "${RUN_DIR}/stdout"

cd "${PROJECT_ROOT}"

# $HOME is mounted noexec on WATGPU. Stage compiled Python packages onto the
# compute node's executable local filesystem for this job.
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/watgpu/stage_env.sh"
STAGE_BASE=$(mlf_choose_stage_base "${ENV_SOURCE}")
MLF_JOB_TMP=$(mktemp -d "${STAGE_BASE%/}/mlf-${USER}-${SLURM_JOB_ID}.XXXXXX")
trap 'rm -rf -- "${MLF_JOB_TMP}"' EXIT

echo "[env] staging ${ENV_SOURCE} -> ${MLF_JOB_TMP}/mlf"
# Copy contents rather than a possible top-level symlink.  Preserve the venv's
# internal links, but not NFS metadata that some node-local filesystems reject.
time mlf_copy_env "${ENV_SOURCE}" "${MLF_JOB_TMP}/mlf"

export VIRTUAL_ENV="${MLF_JOB_TMP}/mlf"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
unset PYTHONHOME
hash -r

# Fail before loading the graph if staging selected the system interpreter or
# lost a required package.
python -c "import torch, matplotlib; print('[env] torch', torch.__version__, '| matplotlib', matplotlib.__version__)"

GRAPH="${DATA_ROOT}/data1m/graphs/hgraph_${RUNG}.pt"
VOCAB="${DATA_ROOT}/data1m/feats/${RUNG}/family_vocab.csv"
[ -f "${GRAPH}" ] || { echo "[fail] no graph at ${GRAPH}"; exit 2; }

# The vocab binding is not optional (CLAUDE.md Step 6: it is the only credential
# for embedding-row identity). This used to be a silent skip, which is how the
# 12K and 30K runs of 2026-08-11 shipped with family_vocab_sha256: null. For the
# CORE rungs the file is dumped straight out of the frozen graph:
#     python -m scale1m.dump_family_vocab --graph "${GRAPH}" --out "${VOCAB}"
[ -f "${VOCAB}" ] || {
  echo "[fail] no family_vocab.csv at ${VOCAB}"
  echo "       dump it first: python -m scale1m.dump_family_vocab \\"
  echo "         --graph ${GRAPH} --out ${VOCAB}"
  exit 2
}
VOCAB_ARG=(--family-vocab "${VOCAB}")

# The T0 scale switches. They are ON here and OFF in the historical runs, which
# is exactly what G-A1 checked on 12K: with them on, the 12K anchor still has to
# come out at gold@10 = 0.4159.
srun python -m scale1m.train_rung \
  --rung "${RUNG}" \
  --graph "${GRAPH}" \
  --out   "${RUN_DIR}" \
  --seed "${SEED}" --epochs "${EPOCHS}" \
  "${VOCAB_ARG[@]}" \
  --fanout --sparse-M --contrast-n-neg 256 --chunked-infer 50000 \
  2>&1 | tee -a "${RUN_DIR}/stdout/train.log"

echo "[done] ${RUN_DIR} ; MANIFEST.json + ckpt/ + metrics/"
