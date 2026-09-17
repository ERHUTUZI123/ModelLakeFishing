#!/bin/bash
#SBATCH --job-name=a06-eval
#SBATCH --output=/u801/x98liu/model_lake/a0_eval_20260914/slurm-%j.log
#SBATCH --error=/u801/x98liu/model_lake/a0_eval_20260914/slurm-%j.log
#SBATCH --nodelist=watgpu308
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
set -euo pipefail
source ~/.mlf_env
PROJECT_ROOT=/u801/x98liu/model_lake/a0_formal_20260913/ModelLakeFishing
A06_BASE=/u801/x98liu/model_lake/a0_eval_20260914
cd "$PROJECT_ROOT"
source scripts/watgpu/stage_env.sh
STAGE_BASE=$(mlf_choose_stage_base "$ENV_SOURCE")
MLF_JOB_TMP=$(mktemp -d "${STAGE_BASE%/}/mlf-a06-${USER}-${SLURM_JOB_ID}.XXXXXX")
trap 'rm -rf -- "${MLF_JOB_TMP}"' EXIT
mlf_copy_env "$ENV_SOURCE" "$MLF_JOB_TMP/mlf"
export VIRTUAL_ENV="$MLF_JOB_TMP/mlf"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export PYTHONUNBUFFERED=1
unset PYTHONHOME
hash -r
python -c "import sys,torch,hnswlib;print('[env]',sys.version,torch.__version__,torch.cuda.get_device_name(0));assert sys.version_info[:2]==(3,11) and torch.cuda.is_available()"
python -m pip freeze > "$A06_BASE/pip_freeze.txt"
export_dir=/u801/x98liu/model_lake/data1m/a0_20260912/exports
metrics=/u801/x98liu/model_lake/data1m/a0_20260912/metrics
test ! -e "$metrics"
common=(--protocol a0 --exports "$export_dir" --run-fmt 'A0GD_full_s%d_e25'
  --sidecar-exports "$export_dir" --sidecar-run-fmt 'A0GD_full_s%d_e25'
  --dataset-nodes /u801/x98liu/model_lake/data1m/rf/canon/dataset_nodes_merged.parquet
  --out "$metrics" --device cuda --query-chunk 16 --model-chunk 50000
  --hnsw-M 32 --ef-construction 200 --hnsw-threads 8 --ef-search 1000 1500 2000 3000 5000
  --a0-protocol-file "$A06_BASE/inputs/A0_PROTOCOL.linux.json"
  --a0-query-identity "$A06_BASE/inputs/audit/A0_QUERY_IDENTITY.jsonl")
for stage in exact hnsw finalize; do
  echo "[A0.6] $stage started"
  /usr/bin/time -v -o "$A06_BASE/${stage}.time" python -m scale1m.eval_y2 --stage "$stage" "${common[@]}" > "$A06_BASE/${stage}.log" 2>&1
  echo "[A0.6] $stage completed"
done
python "$A06_BASE/validate.py"
echo '[A0.6] all stages and delivery checks completed'
