#!/bin/bash
# Generic X4 GPU/runtime probe; target a node with sbatch --nodelist.

#SBATCH --job-name=mlf-x4-gpu-probe
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:05:00

set -euo pipefail

source ~/.mlf_env
: "${ENV_SOURCE:?set ENV_SOURCE in ~/.mlf_env}"
: "${PROJECT_ROOT:?set PROJECT_ROOT in ~/.mlf_env}"
cd "${PROJECT_ROOT}"

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

python -c "import torch; ok=torch.cuda.is_available(); print('torch',torch.__version__,'cuda',ok,'count',torch.cuda.device_count()); assert ok; p=torch.cuda.get_device_properties(0); print('device',p.name,'total_gb',round(p.total_memory/2**30,3))"
