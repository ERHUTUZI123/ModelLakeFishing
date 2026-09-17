#!/bin/bash
# X6 T baseline: X4-GD training recipe with message passing removed.

#SBATCH --job-name=mlf-x6-ng
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00

set -euo pipefail

source ~/.mlf_env
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
: "${DATA_ROOT:?set DATA_ROOT}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT}"
: "${ENV_SOURCE:?set ENV_SOURCE}"
: "${SEED:=0}"
: "${EPOCHS:=25}"
: "${RUN_ID:=X6NoGraph_full_s${SEED}_e${EPOCHS}}"
RESUME_ARG=()
if [[ -n "${RESUME:-}" ]]; then
  RESUME_ARG=(--resume "${RESUME}")
fi

GRAPH="${DATA_ROOT}/data1m/graphs/hgraph_rf"
VOCAB="${DATA_ROOT}/data1m/feats_rf/family_vocab.csv"
RUN_DIR="${OUTPUT_ROOT}/runs/${RUN_ID}"
[[ -d "${GRAPH}" ]] || { echo "[fail] no graph at ${GRAPH}"; exit 2; }
[[ -f "${VOCAB}" ]] || { echo "[fail] no vocab at ${VOCAB}"; exit 2; }
mkdir -p "${RUN_DIR}/stdout" "${PROJECT_ROOT}/logs"
cd "${PROJECT_ROOT}"

source "${PROJECT_ROOT}/scripts/watgpu/stage_env.sh"
STAGE_BASE=$(mlf_choose_stage_base "${ENV_SOURCE}")
MLF_JOB_TMP=$(mktemp -d "${STAGE_BASE%/}/mlf-${USER}-${SLURM_JOB_ID}.XXXXXX")
trap 'rm -rf -- "${MLF_JOB_TMP}"' EXIT
mlf_copy_env "${ENV_SOURCE}" "${MLF_JOB_TMP}/mlf"
export VIRTUAL_ENV="${MLF_JOB_TMP}/mlf"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
unset PYTHONHOME
hash -r

python -c "import torch; from torch_geometric.typing import WITH_PYG_LIB; print('[env]', torch.__version__, torch.cuda.get_device_name(0), 'pyg_lib', WITH_PYG_LIB)"

python -m scale1m.train_rung \
  --rung full --graph "${GRAPH}" --out "${RUN_DIR}" \
  --seed "${SEED}" --epochs "${EPOCHS}" --family-vocab "${VOCAB}" \
  --fanout --sparse-M --contrast-n-neg 256 --chunked-infer 50000 \
  --skip-diagnostics --lake-gamma 0.5 --global-n-datasets 128 --num-layers 0 \
  "${RESUME_ARG[@]}" \
  2>&1 | tee -a "${RUN_DIR}/stdout/train.log"

echo "[done] ${RUN_DIR}"
