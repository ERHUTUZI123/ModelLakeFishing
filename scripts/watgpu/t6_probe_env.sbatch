#!/bin/bash
# Probe pyg-lib on a compute node without damaging the stored T4 environment.
# A usable installed environment is persisted below T6_STATE_DIR; otherwise
# runtime.env points back to the original environment and training uses the
# bounded Python LightLinkLoader fallback.

#SBATCH --job-name=mlf-t6-probe
# Storage preflight: H200 NVL 140 GB plus an executable /tmp with >500 GB free.
#SBATCH --nodelist=watgpu808
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail

: "${T6_STATE_DIR:?set T6_STATE_DIR}"
: "${ENV_SOURCE:?set ENV_SOURCE}"
: "${PROJECT_ROOT:?set PROJECT_ROOT}"

mkdir -p "${T6_STATE_DIR}/env" "${T6_STATE_DIR}/reports"
BASE_ENV=${ENV_SOURCE}
# shellcheck disable=SC1090
source "${PROJECT_ROOT}/scripts/watgpu/stage_env.sh"
STAGE_BASE=$(mlf_choose_stage_base "${BASE_ENV}")
PROBE_TMP=$(mktemp -d "${STAGE_BASE%/}/mlf-pyg-${USER}-${SLURM_JOB_ID}.XXXXXX")
trap 'rm -rf -- "${PROBE_TMP}"' EXIT

RUNTIME_FILE="${T6_STATE_DIR}/runtime.env"
STATUS_FILE="${T6_STATE_DIR}/reports/pyg_probe.txt"
ENV_CHANGED=0

echo "[probe] host=$(hostname) job=${SLURM_JOB_ID} base=${BASE_ENV}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true

# Copy the directory contents (not a possible top-level symlink) and retain the
# venv's internal symlink layout.  watGPU node-local filesystems reject some
# NFS metadata operations, so disable metadata preservation explicitly.
mlf_copy_env "${BASE_ENV}" "${PROBE_TMP}/mlf"
export VIRTUAL_ENV="${PROBE_TMP}/mlf"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
unset PYTHONHOME
hash -r

python -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda)"

# T4's stored environment may be sufficient for embedding but not for the
# stage2 training import graph.  Install this declared project dependency into
# the disposable copy, never into the known-good T4 environment.
if ! python -c "import matplotlib; print('matplotlib', matplotlib.__version__)"; then
  echo "[probe] matplotlib missing; installing into the disposable T6 copy"
  python -m pip install matplotlib
  ENV_CHANGED=1
fi

backend_flags() {
  python -c "from torch_geometric.typing import WITH_PYG_LIB, WITH_TORCH_SPARSE; print('pyg_lib', WITH_PYG_LIB, '| torch_sparse', WITH_TORCH_SPARSE); raise SystemExit(0 if (WITH_PYG_LIB or WITH_TORCH_SPARSE) else 1)"
}

SELECTED_ENV=${BASE_ENV}
PYG_RESULT=base_backend
if ! backend_flags; then
  PYG_RESULT=wheel_unavailable
  if python -m pip install --no-deps pyg-lib \
      -f https://data.pyg.org/whl/torch-2.12.0+cu130.html; then
    if backend_flags; then
      PYG_RESULT=wheel_installed
      ENV_CHANGED=1
    else
      PYG_RESULT=wheel_import_failed
    fi
  fi
fi

# Import the real training entry point before requesting the 96G/GPU job.  A
# missing Python dependency now fails the cheap probe rather than the 12K run.
export PYTHONPATH="$(dirname "${PROJECT_ROOT}"):${PYTHONPATH:-}"
python -c "from ModelLakeFishing.stage2TrainGraphSAGE.ablation import train_eval_one; print('training_import', 'ok')"

if (( ENV_CHANGED )); then
  SELECTED_ENV="${T6_STATE_DIR}/env/training"
  mlf_copy_env "${PROBE_TMP}/mlf" "${SELECTED_ENV}"
fi
python -m pip freeze > "${T6_STATE_DIR}/reports/pip-freeze-selected.txt"

RUNTIME_TMP="${RUNTIME_FILE}.tmp.${SLURM_JOB_ID}"
{
  printf 'export ENV_SOURCE=%q\n' "${SELECTED_ENV}"
  printf 'export T6_PYG_RESULT=%q\n' "${PYG_RESULT}"
} > "${RUNTIME_TMP}"
mv -f "${RUNTIME_TMP}" "${RUNTIME_FILE}"

{
  echo "job_id=${SLURM_JOB_ID}"
  echo "host=$(hostname)"
  echo "result=${PYG_RESULT}"
  echo "selected_env=${SELECTED_ENV}"
} > "${STATUS_FILE}"

echo "[probe] result=${PYG_RESULT} selected_env=${SELECTED_ENV}"
echo "[probe] runtime selection written to ${RUNTIME_FILE}"
