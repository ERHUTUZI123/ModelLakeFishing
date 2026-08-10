#!/bin/bash
# T4 (docs/1M/100kplan.md §7): build the 100K frozen model-feature matrix.
#
# Submit from <REMOTE_PROJECT_ROOT>:
#     mkdir -p logs
#     sbatch --export=ALL,RUN_ID=T4_100k_$(date -u +%Y%m%dT%H%M%SZ) \
#            scripts/watgpu/embed_lake.sbatch
#
# Everything the job needs beyond the repo:
#   $DATA_ROOT/data1m/ladder/100k_model_ids.csv          (6.9 MB, from T3)
#   $PROJECT_ROOT/stage1BuildTransferGraph/hgraph_ml_v2.pt (92 MB, gitignored)
#   $HF_HOME/  with sentence-transformers/all-MiniLM-L6-v2 already downloaded
#
# GPU work is ~70k MiniLM sentence encodes. It took 40 s end to end on an
# RTX 4060 laptop, so the resource request below is deliberately generous
# rather than tuned -- do not read it as a size estimate for R3/R4.

#SBATCH --job-name=mlf-t4-embed
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -euo pipefail

: "${PROJECT_ROOT:?set PROJECT_ROOT}"
: "${DATA_ROOT:?set DATA_ROOT}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT}"
: "${RUN_ID:=T4_100k_manual}"

RUN_DIR="${OUTPUT_ROOT}/runs/${RUN_ID}"
mkdir -p "${RUN_DIR}/metadata" "${RUN_DIR}/stdout"

cd "${PROJECT_ROOT}"
# shellcheck disable=SC1090
source "${ENV_ACTIVATE:?set ENV_ACTIVATE to your venv/conda activate line}"

{
  echo "UTC_TIME=$(date -u --iso-8601=seconds)"
  echo "RUN_ID=${RUN_ID}"
  echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}"
  echo "HOSTNAME=$(hostname)"
  echo "GIT_HEAD=$(git rev-parse HEAD 2>/dev/null || echo 'not-a-git-repo')"
  echo "GIT_DIRTY<<<"; git status --short 2>/dev/null || true; echo ">>>"
  echo "PYTHON=$(which python)"
  python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
} | tee "${RUN_DIR}/metadata/run.txt"

nvidia-smi > "${RUN_DIR}/metadata/nvidia-smi.txt" 2>&1 || true
python -m pip freeze > "${RUN_DIR}/metadata/pip-freeze.txt" 2>&1 || true

srun python -m scale1m.embed_lake \
  --rung 100k \
  --core   "${PROJECT_ROOT}/stage1BuildTransferGraph/hgraph_ml_v2.pt" \
  --ladder "${DATA_ROOT}/data1m/ladder/100k_model_ids.csv" \
  --out    "${DATA_ROOT}/data1m/feats" \
  --ablation 6000 \
  2>&1 | tee "${RUN_DIR}/stdout/embed_lake.log"

cp "${DATA_ROOT}/data1m/feats/100k/FEATS_REPORT.json" "${RUN_DIR}/" || true
cp "${DATA_ROOT}/data1m/feats/100k/family_vocab.csv" "${RUN_DIR}/" || true
echo "[done] artifacts in ${DATA_ROOT}/data1m/feats/100k ; report copied to ${RUN_DIR}"
