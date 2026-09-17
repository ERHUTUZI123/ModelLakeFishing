#!/bin/bash
#SBATCH --job-name=mlf-f4-embed
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail
: "${PROJECT_ROOT:?}"; : "${DATA_ROOT:?}"; : "${OUTPUT_ROOT:?}"
: "${ENV_SOURCE:?}"; : "${RUN_ID:=F4_rf_manual}"

RUN_DIR="${OUTPUT_ROOT}/runs/${RUN_ID}"
mkdir -p "${RUN_DIR}/metadata" "${RUN_DIR}/stdout"
cd "${PROJECT_ROOT}"

# $HOME is noexec on watGPU: stage the venv onto the node's local disk.
MLF_JOB_TMP=$(mktemp -d "/tmp/mlf-${USER}-${SLURM_JOB_ID}.XXXXXX")
trap 'rm -rf -- "${MLF_JOB_TMP}"' EXIT
if [[ "${MLF_STAGE_PORTABLE:-0}" == "1" ]]; then
  # Some newer nodes expose /tmp without metadata-preservation support.
  # Recursive copy still preserves executable mode bits and symlinks, which
  # are the properties the staged venv requires.
  time cp -R "${ENV_SOURCE}" "${MLF_JOB_TMP}/mlf"
else
  time cp -a "${ENV_SOURCE}" "${MLF_JOB_TMP}/mlf"
fi
export VIRTUAL_ENV="${MLF_JOB_TMP}/mlf"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
unset PYTHONHOME; hash -r

{
  echo "UTC_TIME=$(date -u --iso-8601=seconds)"
  echo "RUN_ID=${RUN_ID}"
  echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}"
  echo "HOSTNAME=$(hostname)"
  echo "GIT_HEAD=$(git rev-parse HEAD)"
  echo "GIT_DIRTY<<<"; git status --short; echo ">>>"
  python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
} | tee "${RUN_DIR}/metadata/run.txt"
nvidia-smi > "${RUN_DIR}/metadata/nvidia-smi.txt" 2>&1 || true
python -m pip freeze > "${RUN_DIR}/metadata/pip-freeze.txt" 2>&1 || true

srun python -m scale1m.embed_lake_rf \
  --ladder "${DATA_ROOT}/data1m/ladder_rf/full_model_ids.parquet" \
  --out    "${DATA_ROOT}/data1m/feats_rf" \
  2>&1 | tee "${RUN_DIR}/stdout/embed_lake_rf.log"

cp "${DATA_ROOT}/data1m/feats_rf/FEATS_REPORT.json" "${RUN_DIR}/" || true
cp "${DATA_ROOT}/data1m/feats_rf/family_vocab.csv"  "${RUN_DIR}/" || true
echo "[done] artifacts in ${DATA_ROOT}/data1m/feats_rf"
