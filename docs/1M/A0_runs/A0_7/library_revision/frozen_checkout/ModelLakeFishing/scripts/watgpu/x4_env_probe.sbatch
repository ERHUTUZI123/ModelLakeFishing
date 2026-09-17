#!/bin/bash
# X4 H200/PyG environment gate.

#SBATCH --job-name=mlf-x4-env
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#SBATCH --constraint=H200
# The free H200 on 1103 cannot stage the persisted environment locally.
#SBATCH --exclude=watgpu1103
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00

set -euo pipefail

source ~/.mlf_env
: "${ENV_SOURCE:?set ENV_SOURCE in ~/.mlf_env}"
: "${PROJECT_ROOT:?set PROJECT_ROOT in ~/.mlf_env}"
cd "${PROJECT_ROOT}"

# Use the same relocatable environment staging path as the training job.  The
# shared source is storage, not the runtime environment used on compute nodes.
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/watgpu/stage_env.sh"
STAGE_BASE=$(mlf_choose_stage_base "${ENV_SOURCE}")
MLF_JOB_TMP=$(mktemp -d "${STAGE_BASE%/}/mlf-${USER}-${SLURM_JOB_ID}.XXXXXX")
trap 'rm -rf -- "${MLF_JOB_TMP}"' EXIT
mlf_copy_env "${ENV_SOURCE}" "${MLF_JOB_TMP}/mlf"

export VIRTUAL_ENV="${MLF_JOB_TMP}/mlf"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
unset PYTHONHOME
hash -r

python -c "import torch, matplotlib; from torch_geometric.typing import WITH_PYG_LIB, WITH_TORCH_SPARSE; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| pyg_lib', WITH_PYG_LIB, '| torch_sparse', WITH_TORCH_SPARSE); assert WITH_PYG_LIB or WITH_TORCH_SPARSE"
