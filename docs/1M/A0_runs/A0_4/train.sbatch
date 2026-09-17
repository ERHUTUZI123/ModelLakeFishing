#!/bin/bash
#SBATCH --job-name=a04-gd
#SBATCH --output=/u801/x98liu/model_lake/a0_formal_20260913/slurm-%A_%a.log
#SBATCH --error=/u801/x98liu/model_lake/a0_formal_20260913/slurm-%A_%a.log
#SBATCH --nodelist=watgpu308
#SBATCH --gres=gpu:schoolgpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --array=0-2
set -euo pipefail
source ~/.mlf_env
A04_ROOT=/u801/x98liu/model_lake/a0_formal_20260913
PROJECT_ROOT="$A04_ROOT/ModelLakeFishing"
cd "$PROJECT_ROOT"
source scripts/watgpu/stage_env.sh
STAGE_BASE=$(mlf_choose_stage_base "$ENV_SOURCE")
MLF_JOB_TMP=$(mktemp -d "${STAGE_BASE%/}/mlf-a04-${USER}-${SLURM_JOB_ID}.XXXXXX")
trap 'rm -rf -- "${MLF_JOB_TMP}"' EXIT
mlf_copy_env "$ENV_SOURCE" "$MLF_JOB_TMP/mlf"
export VIRTUAL_ENV="$MLF_JOB_TMP/mlf"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export PYTHONUNBUFFERED=1
unset PYTHONHOME
hash -r
python -c "import torch; from torch_geometric.typing import WITH_PYG_LIB; print('[env]',torch.__version__,torch.cuda.get_device_name(0),torch.cuda.get_device_properties(0).total_memory,'pyg_lib',WITH_PYG_LIB); assert torch.cuda.is_available() and WITH_PYG_LIB"
python -m pip freeze > "$A04_ROOT/pip_freeze_s${SLURM_ARRAY_TASK_ID}.txt"
python "$A04_ROOT/monitor_train.py" "$SLURM_ARRAY_TASK_ID"
