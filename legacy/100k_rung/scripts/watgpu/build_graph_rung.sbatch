#!/bin/bash
# T5 (docs/1M/100kplan.md §8): assemble hgraph_100k.pt and run the exit gates.
#
# Submit from <REMOTE_PROJECT_ROOT>:
#     mkdir -p logs
#     sbatch --export=ALL,RUN_ID=T5_100k_$(date -u +%Y%m%dT%H%M%SZ) \
#            scripts/watgpu/build_graph_rung.sbatch
#
# No GPU. The build is a few tensor copies plus one hashmap join; the gates
# include one full-graph forward/backward through HeteroGraphSAGE on CPU, which
# is the slowest part and still finished in 24 s locally.
#
# Inputs beyond the repo:
#   $DATA_ROOT/data1m/ladder/100k_model_ids.csv               (from T3)
#   $DATA_ROOT/data1m/feats/100k/{x_m,size_bucket_id,family_id}.npy,
#                                 family_vocab.csv            (from T4)
#   $DATA_ROOT/data1m/candidates_v2/canon/hf_canon.parquet    (lineage_relation)
#   $PROJECT_ROOT/stage1BuildTransferGraph/hgraph_ml_v2.pt
#
# hf_canon.parquet is optional: without it every lineage edge lands in the
# `unknown` relation bucket. The edge weights are 1.0 either way, but the
# relation ids are the input r_mm' needs later, and recovering them means
# rebuilding the graph.

#SBATCH --job-name=mlf-t5-graph
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -euo pipefail

: "${PROJECT_ROOT:?set PROJECT_ROOT}"
: "${DATA_ROOT:?set DATA_ROOT}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT}"
: "${RUN_ID:=T5_100k_manual}"
: "${RUNG:=100k}"

RUN_DIR="${OUTPUT_ROOT}/runs/${RUN_ID}"
mkdir -p "${RUN_DIR}/metadata" "${RUN_DIR}/stdout"

cd "${PROJECT_ROOT}"

: "${ENV_SOURCE:?set ENV_SOURCE to the stored Python environment}"

# $HOME is mounted noexec on WATGPU. Stage compiled Python packages onto the
# compute node's executable local filesystem for this job.
MLF_JOB_TMP=$(mktemp -d "/tmp/mlf-${USER}-${SLURM_JOB_ID}.XXXXXX")
trap 'rm -rf -- "${MLF_JOB_TMP}"' EXIT

echo "[env] staging ${ENV_SOURCE} -> ${MLF_JOB_TMP}/mlf"
time cp -a "${ENV_SOURCE}" "${MLF_JOB_TMP}/mlf"

export VIRTUAL_ENV="${MLF_JOB_TMP}/mlf"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
unset PYTHONHOME
hash -r

GRAPHS="${DATA_ROOT}/data1m/graphs"
mkdir -p "${GRAPHS}"

{
  echo "UTC_TIME=$(date -u --iso-8601=seconds)"
  echo "RUN_ID=${RUN_ID}"
  echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}"
  echo "HOSTNAME=$(hostname)"
  echo "GIT_HEAD=$(git rev-parse HEAD 2>/dev/null || echo 'not-a-git-repo')"
  echo "GIT_DIRTY<<<"; git status --short 2>/dev/null || true; echo ">>>"
  echo "PYTHON=$(which python)"
  python -c "import torch;print('torch',torch.__version__)"
} | tee "${RUN_DIR}/metadata/run.txt"

python -m pip freeze > "${RUN_DIR}/metadata/pip-freeze.txt" 2>&1 || true

CANON="${DATA_ROOT}/data1m/candidates_v2/canon/hf_canon.parquet"
CANON_ARG=()
if [ -f "${CANON}" ]; then
  CANON_ARG=(--canon "${CANON}")
else
  echo "[warn] ${CANON} missing -- lineage relations will all be 'unknown'"
fi

srun python -m scale1m.build_graph_rung \
  --rung "${RUNG}" \
  --core   "${PROJECT_ROOT}/stage1BuildTransferGraph/hgraph_ml_v2.pt" \
  --ladder "${DATA_ROOT}/data1m/ladder/${RUNG}_model_ids.csv" \
  --feats  "${DATA_ROOT}/data1m/feats/${RUNG}" \
  "${CANON_ARG[@]}" \
  --out    "${GRAPHS}/hgraph_${RUNG}.pt" \
  2>&1 | tee "${RUN_DIR}/stdout/build_graph.log"

srun python -m scale1m.verify_rung_graph \
  --rung "${RUNG}" \
  --graph "${GRAPHS}/hgraph_${RUNG}.pt" \
  --core  "${PROJECT_ROOT}/stage1BuildTransferGraph/hgraph_ml_v2.pt" \
  --out   "${GRAPHS}/T5_GATES_${RUNG}.json" \
  2>&1 | tee "${RUN_DIR}/stdout/verify_graph.log"

cp "${GRAPHS}/GRAPH_REPORT_${RUNG}.json" "${GRAPHS}/lineage_stats.json" \
   "${GRAPHS}/T5_GATES_${RUNG}.json" "${RUN_DIR}/" || true
echo "[done] graph in ${GRAPHS}/hgraph_${RUNG}.pt ; reports copied to ${RUN_DIR}"
