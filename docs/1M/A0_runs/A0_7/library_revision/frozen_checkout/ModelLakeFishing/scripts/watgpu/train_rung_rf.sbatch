#!/bin/bash
# F6: train the RF full-lake graph (or the later 500K graph).

#SBATCH --job-name=mlf-f6-rf
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
# The full graph needs an 80 GB+ GPU; this is the H200 node used by T6.
#SBATCH --nodelist=watgpu808
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
# The 1-epoch smoke measures the final request; formal runs override --time.
#SBATCH --time=01:00:00

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
: "${RUNG:=full}"
: "${SEED:=0}"
: "${EPOCHS:=25}"
: "${RUN_ID:=RF_${RUNG}_s${SEED}_e${EPOCHS}}"

case "${RUNG}" in
  full)
    GRAPH="${DATA_ROOT}/data1m/graphs/hgraph_rf"
    EXPECT_ARG=()
    ;;
  500k)
    GRAPH="${DATA_ROOT}/data1m/graphs/hgraph_rf_500k"
    : "${EXPECT_N:?set EXPECT_N for the 500k graph}"
    EXPECT_ARG=(--expect-n "${EXPECT_N}")
    ;;
  *)
    echo "[fail] unsupported RF rung: ${RUNG}" >&2
    exit 2
    ;;
esac
VOCAB="${DATA_ROOT}/data1m/feats_rf/family_vocab.csv"
RUN_DIR="${OUTPUT_ROOT}/runs/${RUN_ID}"

[[ -d "${GRAPH}" ]] || { echo "[fail] no sharded graph at ${GRAPH}"; exit 2; }
[[ -f "${VOCAB}" ]] || { echo "[fail] no family_vocab.csv at ${VOCAB}"; exit 2; }
mkdir -p "${RUN_DIR}/stdout"
cd "${PROJECT_ROOT}"

# $HOME is noexec. Use the shared selector because node-local capacity and
# metadata semantics vary across watGPU hosts.
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

srun python -m scale1m.train_rung \
  --rung "${RUNG}" \
  --graph "${GRAPH}" \
  --out "${RUN_DIR}" \
  --seed "${SEED}" --epochs "${EPOCHS}" \
  --family-vocab "${VOCAB}" \
  "${EXPECT_ARG[@]}" \
  --fanout --sparse-M --contrast-n-neg 256 --chunked-infer 50000 \
  --skip-diagnostics \
  2>&1 | tee -a "${RUN_DIR}/stdout/train.log"

echo "[done] ${RUN_DIR} ; MANIFEST.json + ckpt/ + metrics/"
