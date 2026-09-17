#!/bin/bash
# X4: train proposal-distribution / global-dataset-count ablations on RF full lake.

#SBATCH --job-name=mlf-x4
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
# Verified fallback with enough GPU and host memory.  Submitters may override
# --nodelist after probing another compatible host (X4 used watgpu608 too).
#SBATCH --nodelist=watgpu308
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:30:00

set -euo pipefail

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
: "${ARM:?set ARM to G, D or GD}"
: "${RUNG:=full}"
: "${SEED:=0}"
: "${EPOCHS:=25}"

case "${ARM}" in
  G)  EXTRA=(--lake-gamma 0.5) ;;
  D)  EXTRA=(--global-n-datasets 128) ;;
  GD) EXTRA=(--lake-gamma 0.5 --global-n-datasets 128) ;;
  *)  echo "[fail] unsupported ARM: ${ARM}" >&2; exit 2 ;;
esac
: "${RUN_ID:=X4${ARM}_full_s${SEED}_e${EPOCHS}}"

[[ "${RUNG}" == "full" ]] || { echo "[fail] X4 only supports full rung"; exit 2; }
GRAPH="${DATA_ROOT}/data1m/graphs/hgraph_rf"
VOCAB="${DATA_ROOT}/data1m/feats_rf/family_vocab.csv"
RUN_DIR="${OUTPUT_ROOT}/runs/${RUN_ID}"

[[ -d "${GRAPH}" ]] || { echo "[fail] no sharded graph at ${GRAPH}"; exit 2; }
[[ -f "${VOCAB}" ]] || { echo "[fail] no family_vocab.csv at ${VOCAB}"; exit 2; }
mkdir -p "${RUN_DIR}/stdout"
cd "${PROJECT_ROOT}"

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/watgpu/stage_env.sh"
STAGE_BASE=$(mlf_choose_stage_base "${ENV_SOURCE}")
MLF_JOB_TMP=$(mktemp -d "${STAGE_BASE%/}/mlf-${USER}-${SLURM_JOB_ID}.XXXXXX")
trap 'rm -rf -- "${MLF_JOB_TMP}"' EXIT
echo "[env] staging ${ENV_SOURCE} -> ${MLF_JOB_TMP}/mlf"
time mlf_copy_env "${ENV_SOURCE}" "${MLF_JOB_TMP}/mlf"

export VIRTUAL_ENV="${MLF_JOB_TMP}/mlf"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
unset PYTHONHOME
hash -r

python -c "import torch, matplotlib; from torch_geometric.typing import WITH_PYG_LIB, WITH_TORCH_SPARSE; print('[env] torch',torch.__version__,'cuda',torch.cuda.is_available(),'pyg_lib',WITH_PYG_LIB,'torch_sparse',WITH_TORCH_SPARSE)"

TRAIN_CMD=(python -m scale1m.train_rung
  --rung "${RUNG}" \
  --graph "${GRAPH}" \
  --out "${RUN_DIR}" \
  --seed "${SEED}" --epochs "${EPOCHS}" \
  --family-vocab "${VOCAB}" \
  --fanout --sparse-M --contrast-n-neg 256 --chunked-infer 50000 \
  --skip-diagnostics \
  "${EXTRA[@]}")

if [[ "${X4_DIRECT:-0}" == "1" ]]; then
  echo "[launch] direct within batch allocation (srun unavailable on this host)"
  "${TRAIN_CMD[@]}" 2>&1 | tee -a "${RUN_DIR}/stdout/train.log"
else
  srun "${TRAIN_CMD[@]}" 2>&1 | tee -a "${RUN_DIR}/stdout/train.log"
fi

echo "[done] ${RUN_DIR} ; MANIFEST.json + ckpt/ + metrics/"
