#!/bin/bash
#SBATCH --job-name=a03-smoke
#SBATCH --output=/u801/x98liu/model_lake/a0_smoke_20260913/slurm-%j.log
#SBATCH --error=/u801/x98liu/model_lake/a0_smoke_20260913/slurm-%j.log
#SBATCH --nodelist=watgpu308
#SBATCH --gres=gpu:schoolgpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:30:00
set -euo pipefail
source ~/.mlf_env
PROJECT_ROOT=/u801/x98liu/model_lake/a0_smoke_20260913/ModelLakeFishing
cd "$PROJECT_ROOT"
source scripts/watgpu/stage_env.sh
STAGE_BASE=$(mlf_choose_stage_base "$ENV_SOURCE")
MLF_JOB_TMP=$(mktemp -d "${STAGE_BASE%/}/mlf-a03-${USER}-${SLURM_JOB_ID}.XXXXXX")
trap 'rm -rf -- "${MLF_JOB_TMP}"' EXIT
mlf_copy_env "$ENV_SOURCE" "$MLF_JOB_TMP/mlf"
export VIRTUAL_ENV="$MLF_JOB_TMP/mlf"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export PYTHONUNBUFFERED=1
unset PYTHONHOME
hash -r
python -c "import torch; from torch_geometric.typing import WITH_PYG_LIB; print('[env]',torch.__version__,torch.cuda.get_device_name(0),torch.cuda.get_device_properties(0).total_memory,'pyg_lib',WITH_PYG_LIB); assert torch.cuda.is_available() and WITH_PYG_LIB"
python -m pip install pytest==9.1.1
python -m pip freeze > ../pip_freeze.txt
python -m pytest scale1m/tests/test_a03_information_boundary.py -q > ../boundary.log 2>&1
python a0_ops/monitor_smoke.py
