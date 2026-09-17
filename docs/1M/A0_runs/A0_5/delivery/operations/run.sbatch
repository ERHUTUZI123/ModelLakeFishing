#!/bin/bash
#SBATCH --job-name=a05-export
#SBATCH --output=/u801/x98liu/model_lake/a0_export_20260913/slurm-%j.log
#SBATCH --error=/u801/x98liu/model_lake/a0_export_20260913/slurm-%j.log
#SBATCH --nodelist=watgpu308
#SBATCH --gres=gpu:schoolgpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
set -euo pipefail
source ~/.mlf_env
PROJECT_ROOT=/u801/x98liu/model_lake/a0_formal_20260913/ModelLakeFishing
A05_BASE=/u801/x98liu/model_lake/a0_export_20260913
cd "$PROJECT_ROOT"
source scripts/watgpu/stage_env.sh
STAGE_BASE=$(mlf_choose_stage_base "$ENV_SOURCE")
MLF_JOB_TMP=$(mktemp -d "${STAGE_BASE%/}/mlf-a05-${USER}-${SLURM_JOB_ID}.XXXXXX")
trap 'rm -rf -- "${MLF_JOB_TMP}"' EXIT
mlf_copy_env "$ENV_SOURCE" "$MLF_JOB_TMP/mlf"
export VIRTUAL_ENV="$MLF_JOB_TMP/mlf"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export PYTHONUNBUFFERED=1
unset PYTHONHOME
hash -r
python -m pip freeze > "$A05_BASE/pip_freeze.txt"
python -c "import torch;print(torch.__version__,torch.cuda.get_device_name(0));assert torch.cuda.is_available()"
for seed in 0 1 2; do
  run="/u801/x98liu/model_lake/runs/A0_20260912/A0GD_full_s${seed}_e25"
  export_dir="/u801/x98liu/model_lake/data1m/a0_20260912/exports/A0GD_full_s${seed}_e25"
  test ! -e "$export_dir"
  echo "[A0.5] seed=$seed export"
  /usr/bin/time -v -o "$A05_BASE/export_s${seed}.time" python -m scale1m.export_rf \
    --run "$run" --stage embed --ckpt last \
    --graph /u801/x98liu/model_lake/data1m/a0_20260912/graph \
    --ladder /u801/x98liu/model_lake/data1m/ladder_rf/full_model_ids.parquet \
    --chunk 50000 --out "$export_dir" > "$A05_BASE/export_s${seed}.log" 2>&1
  echo "[A0.5] seed=$seed prior"
  /usr/bin/time -v -o "$A05_BASE/prior_s${seed}.time" python -m stage3HNSW.build_prior_sidecar \
    --graph-store /u801/x98liu/model_lake/data1m/a0_20260912/graph \
    --export "$export_dir" --split-seed "$seed" \
    --task-nodes /u801/x98liu/model_lake/data1m/rf/canon/dataset_nodes_merged.parquet \
    > "$A05_BASE/prior_s${seed}.log" 2>&1
  python "$A05_BASE/validate.py" "$seed" > "$A05_BASE/validation_s${seed}.log" 2>&1
  cat "$A05_BASE/validation_s${seed}.log"
done
echo '[A0.5] all three seeds PASS'
